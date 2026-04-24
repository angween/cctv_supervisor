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
    """Manages multiple StreamCapture instances for concurrent RTSP processing."""

    def __init__(self, channels: list, rtsp_url_builder, frame_skip: int = 3):
        """Initialize stream manager.

        Args:
            channels: List of camera channel numbers.
            rtsp_url_builder: Function that takes channel number and returns RTSP URL.
            frame_skip: Process every Nth frame (default: 3).
        """
        self.streams: Dict[int, StreamCapture] = {}

        for ch in channels:
            url = rtsp_url_builder(ch)
            self.streams[ch] = StreamCapture(
                channel=ch,
                url=url,
                frame_skip=frame_skip
            )

    def start_all(self):
        """Start all stream capture threads."""
        for ch, stream in self.streams.items():
            logger.info(f"Starting capture for channel {ch}")
            stream.start()

    def stop_all(self):
        """Signal all streams to stop and wait for threads to finish."""
        for stream in self.streams.values():
            stream.stop()

        for stream in self.streams.values():
            stream.join(timeout=5)

        logger.info("All stream captures stopped")

    def get_latest_frames(self) -> Dict[int, np.ndarray]:
        """Get the latest frame from each connected stream.

        Returns:
            Dictionary mapping channel number to its latest frame.
            Only includes channels with available frames.
        """
        frames = {}
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
        return {ch: stream.connected for ch, stream in self.streams.items()}
