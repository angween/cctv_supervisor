"""
stream_manager.py — Multi-threaded RTSP Stream Capture

Manages concurrent RTSP stream readers using dedicated threads
with thread-safe queues for frame delivery.
"""

import time
import logging
import threading
from typing import Dict, Optional

import cv2
import numpy as np

logger = logging.getLogger(__name__)


class StreamCapture(threading.Thread):
    """Dedicated thread for capturing frames from a single RTSP stream.

    Reads frames continuously and keeps only the latest frame in a
    thread-safe manner. Supports auto-reconnect on stream failure.
    """

    def __init__(self, channel: int, url: str, frame_skip: int = 3):
        super().__init__(daemon=True)
        self.channel = channel
        self.url = url
        self.frame_skip = frame_skip

        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._latest_frame: Optional[np.ndarray] = None
        self._connected = False
        self._frame_count = 0
        self._reconnect_delay = 5  # seconds

    @property
    def connected(self) -> bool:
        return self._connected

    def get_frame(self) -> Optional[np.ndarray]:
        """Get the latest captured frame (thread-safe).

        Returns:
            The latest frame as numpy array, or None if no frame available.
        """
        with self._lock:
            return self._latest_frame.copy() if self._latest_frame is not None else None

    def stop(self):
        """Signal the thread to stop."""
        self._stop_event.set()

    def run(self):
        """Main thread loop: capture frames with auto-reconnect."""
        while not self._stop_event.is_set():
            cap = None
            try:
                logger.info(f"[Channel {self.channel}] Connecting to Channel {self.channel}")
                cap = cv2.VideoCapture(self.url, cv2.CAP_FFMPEG)

                # Set buffer size to minimize latency
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)

                if not cap.isOpened():
                    logger.warning(
                        f"[Channel {self.channel}] Failed to open stream. "
                        f"Retrying in {self._reconnect_delay}s..."
                    )
                    self._connected = False
                    self._wait_or_stop(self._reconnect_delay)
                    continue

                self._connected = True
                logger.info(f"[Channel {self.channel}] Connected successfully")
                self._frame_count = 0

                while not self._stop_event.is_set():
                    ret, frame = cap.read()

                    if not ret:
                        logger.warning(
                            f"[Channel {self.channel}] Frame read failed. Reconnecting..."
                        )
                        self._connected = False
                        break

                    self._frame_count += 1

                    # Frame skipping: only keep every Nth frame
                    if self._frame_count % self.frame_skip != 0:
                        continue

                    with self._lock:
                        self._latest_frame = frame

            except Exception as e:
                logger.error(f"[Channel {self.channel}] Error: {e}")
                self._connected = False

            finally:
                if cap is not None:
                    cap.release()

            # Wait before reconnecting
            if not self._stop_event.is_set():
                self._wait_or_stop(self._reconnect_delay)

        logger.info(f"[Channel {self.channel}] Capture thread stopped")

    def _wait_or_stop(self, seconds: float):
        """Wait for specified seconds, but return early if stop is signaled."""
        self._stop_event.wait(timeout=seconds)


class StreamManager:
    """Manages multiple StreamCapture instances for concurrent RTSP processing with batch rotation."""

    def __init__(self, cameras: list, rtsp_url_builder, batch_size: int, duration_loop_sec: int, frame_skip: int = 3, rotation_callback=None):
        """Initialize stream manager.

        Args:
            cameras: List of camera configuration dicts.
            rtsp_url_builder: Function that takes camera dict and returns RTSP URL.
            batch_size: Number of cameras to process simultaneously.
            duration_loop_sec: How long to monitor a batch before rotating (in seconds).
            frame_skip: Process every Nth frame (default: 3).
            rotation_callback: Function to call when rotating batches.
        """
        self.all_cameras = cameras
        self.rtsp_url_builder = rtsp_url_builder
        self.batch_size = batch_size
        self.duration_loop_sec = duration_loop_sec
        self.frame_skip = frame_skip
        self.rotation_callback = rotation_callback
        
        self.streams: Dict[int, StreamCapture] = {}
        self.active_batch_indices = []
        self.current_batch_start_index = 0
        self.last_rotation_time = 0
        
        # Lock for thread-safe access to streams dict during rotation
        self._streams_lock = threading.Lock()

    def update(self):
        """Check if it's time to rotate the batch, and perform rotation if needed.
        Should be called regularly in the main loop.
        """
        if not self.all_cameras:
            return

        current_time = time.time()
        # If duration_loop_sec is 0 or less, don't rotate
        if self.duration_loop_sec > 0 and (current_time - self.last_rotation_time >= self.duration_loop_sec):
            self.rotate_batch()

    def rotate_batch(self):
        """Rotate to the next batch of cameras."""
        # Stop current streams
        self.stop_all()

        if not self.all_cameras:
            return

        # Calculate next batch indices
        n_cameras = len(self.all_cameras)
        batch = []
        for i in range(self.batch_size):
            idx = (self.current_batch_start_index + i) % n_cameras
            batch.append(self.all_cameras[idx])
            # If we've looped through all cameras and batch_size > total cameras
            if len(batch) == n_cameras:
                break
        
        self.current_batch_start_index = (self.current_batch_start_index + len(batch)) % n_cameras
        self.last_rotation_time = time.time()

        # Start new streams
        with self._streams_lock:
            self.streams.clear()
            for cam in batch:
                ch = cam.get("channel")
                url = self.rtsp_url_builder(cam)
                self.streams[ch] = StreamCapture(
                    channel=ch,
                    url=url,
                    frame_skip=self.frame_skip
                )
        
        self.start_all()
        
        if self.rotation_callback:
            self.rotation_callback(batch)

    def start_all(self):
        """Start current batch of stream capture threads."""
        with self._streams_lock:
            for ch, stream in self.streams.items():
                logger.info(f"Starting capture for channel {ch}")
                stream.start()

    def stop_all(self):
        """Signal all streams to stop and wait for threads to finish."""
        with self._streams_lock:
            streams_copy = list(self.streams.values())
            
        for stream in streams_copy:
            stream.stop()

        for stream in streams_copy:
            stream.join(timeout=5)

        logger.info("Current stream captures stopped")

    def get_latest_frames(self) -> Dict[int, np.ndarray]:
        """Get the latest frame from each connected stream.

        Returns:
            Dictionary mapping channel number to its latest frame.
            Only includes channels with available frames.
        """
        frames = {}
        with self._streams_lock:
            for ch, stream in self.streams.items():
                frame = stream.get_frame()
                if frame is not None:
                    frames[ch] = frame
        return frames

    def get_status(self) -> Dict[int, bool]:
        """Get connection status for all streams.

        Returns:
            Dictionary mapping channel number to connection status.
        """
        with self._streams_lock:
            return {ch: stream.connected for ch, stream in self.streams.items()}
