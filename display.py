"""
display.py — Optional Video Display Manager

Displays a grid view of all camera streams with detection overlays
when the --display flag is enabled.
"""

import logging
from typing import Dict, List, Tuple

import cv2
import numpy as np

from detector import ChannelDetections, PersonDetection, PhoneDetection
from activity_analyzer import ActivityEvent

logger = logging.getLogger(__name__)

# Colors (BGR format)
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)
COLOR_WHITE = (255, 255, 255)
COLOR_ORANGE = (0, 165, 255)
COLOR_BG = (40, 40, 40)

# Activity display config
ACTIVITY_COLORS = {
    "sleeping": COLOR_RED,
    "chatting": COLOR_YELLOW,
    "phone_usage": COLOR_ORANGE,
}

ACTIVITY_LABELS = {
    "sleeping": "TIDUR",
    "chatting": "MENGOBROL",
    "phone_usage": "MAIN HP",
}

# COCO skeleton connections for drawing
SKELETON_CONNECTIONS = [
    (0, 1), (0, 2), (1, 3), (2, 4),        # Head
    (5, 6),                                   # Shoulders
    (5, 7), (7, 9),                          # Left arm
    (6, 8), (8, 10),                         # Right arm
    (5, 11), (6, 12),                        # Torso
    (11, 12),                                 # Hips
    (11, 13), (13, 15),                      # Left leg
    (12, 14), (14, 16),                      # Right leg
]


class DisplayManager:
    """Manages real-time video display with detection overlays."""

    WINDOW_NAME = "CCTV Supervisor"

    def __init__(self, grid_width: int = 960, grid_height: int = 540):
        """Initialize display manager.

        Args:
            grid_width: Target width for each cell in the grid.
            grid_height: Target height for each cell in the grid.
        """
        self.cell_width = grid_width
        self.cell_height = grid_height
        self._window_created = False

    def _ensure_window(self):
        """Create the display window if not already created."""
        if not self._window_created:
            cv2.namedWindow(self.WINDOW_NAME, cv2.WINDOW_NORMAL)
            self._window_created = True

    def update(
        self,
        frames: Dict[int, np.ndarray],
        detections: Dict[int, ChannelDetections],
        events: List[ActivityEvent],
        active_durations: Dict[Tuple[int, str], float],
        stream_status: Dict[int, bool]
    ) -> bool:
        """Update the display with current frames and detections.

        Args:
            frames: Channel-to-frame mapping.
            detections: Channel-to-detections mapping.
            events: Current activity events.
            active_durations: Currently tracked activity durations.
            stream_status: Connection status per channel.

        Returns:
            False if user pressed 'q' to quit, True otherwise.
        """
        self._ensure_window()

        if not frames and not stream_status:
            return True

        # Get all channels (from status, which includes disconnected ones)
        all_channels = sorted(stream_status.keys())
        n = len(all_channels)

        # Calculate grid layout
        if n <= 1:
            rows, cols = 1, 1
        elif n <= 2:
            rows, cols = 1, 2
        elif n <= 4:
            rows, cols = 2, 2
        else:
            cols = 3
            rows = (n + cols - 1) // cols

        # Build event lookup by channel
        events_by_channel: Dict[int, List[ActivityEvent]] = {}
        for event in events:
            events_by_channel.setdefault(event.channel, []).append(event)

        # Render each cell
        cells = []
        for ch in all_channels:
            if ch in frames:
                cell = self._render_channel(
                    channel=ch,
                    frame=frames[ch],
                    channel_detections=detections.get(ch),
                    channel_events=events_by_channel.get(ch, []),
                    active_durations=active_durations,
                    connected=stream_status.get(ch, False)
                )
            else:
                cell = self._render_disconnected(ch)

            cells.append(cell)

        # Pad with blank cells if needed
        while len(cells) < rows * cols:
            cells.append(np.zeros(
                (self.cell_height, self.cell_width, 3), dtype=np.uint8
            ))

        # Assemble grid
        grid_rows = []
        for r in range(rows):
            row_cells = cells[r * cols: (r + 1) * cols]
            grid_rows.append(np.hstack(row_cells))

        grid = np.vstack(grid_rows)

        cv2.imshow(self.WINDOW_NAME, grid)

        # Check for 'q' key press (1ms wait)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            logger.info("User pressed 'q' — quitting display")
            return False

        return True

    def _render_channel(
        self,
        channel: int,
        frame: np.ndarray,
        channel_detections: ChannelDetections,
        channel_events: List[ActivityEvent],
        active_durations: Dict[Tuple[int, str], float],
        connected: bool
    ) -> np.ndarray:
        """Render a single channel's frame with overlays."""
        # Resize frame to cell dimensions
        display = cv2.resize(frame, (self.cell_width, self.cell_height))
        scale_x = self.cell_width / frame.shape[1]
        scale_y = self.cell_height / frame.shape[0]

        if channel_detections:
            # Draw person bounding boxes and skeletons
            for person in channel_detections.persons:
                self._draw_person(display, person, scale_x, scale_y)

            # Draw phone bounding boxes
            for phone in channel_detections.phones:
                self._draw_phone(display, phone, scale_x, scale_y)

        # Draw activity labels
        active_types = set()
        for event in channel_events:
            active_types.add(event.activity_type)

        y_offset = 60
        for activity_type in active_types:
            color = ACTIVITY_COLORS.get(activity_type, COLOR_RED)
            label = ACTIVITY_LABELS.get(activity_type, activity_type)

            # Get duration if being tracked
            duration = active_durations.get((channel, activity_type), 0)
            duration_str = f" ({duration:.0f}s)" if duration > 0 else ""

            cv2.putText(
                display, f"⚠ {label}{duration_str}",
                (10, y_offset), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, color, 2
            )
            y_offset += 30

        # Draw channel header
        status_color = COLOR_GREEN if connected else COLOR_RED
        status_text = "LIVE" if connected else "DISCONNECTED"
        header = f"CH {channel} | {status_text}"

        # Header background
        cv2.rectangle(display, (0, 0), (self.cell_width, 30), (0, 0, 0), -1)
        cv2.putText(
            display, header,
            (10, 22), cv2.FONT_HERSHEY_SIMPLEX,
            0.6, status_color, 2
        )

        return display

    def _render_disconnected(self, channel: int) -> np.ndarray:
        """Render a placeholder for a disconnected channel."""
        cell = np.zeros((self.cell_height, self.cell_width, 3), dtype=np.uint8)
        cell[:] = COLOR_BG

        text = f"CH {channel} — DISCONNECTED"
        text_size = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)[0]
        x = (self.cell_width - text_size[0]) // 2
        y = (self.cell_height + text_size[1]) // 2

        cv2.putText(
            cell, text,
            (x, y), cv2.FONT_HERSHEY_SIMPLEX,
            0.8, COLOR_RED, 2
        )

        return cell

    def _draw_person(
        self, frame: np.ndarray,
        person: PersonDetection,
        sx: float, sy: float
    ):
        """Draw person bounding box and skeleton on frame."""
        bbox = person.bbox
        x1, y1 = int(bbox[0] * sx), int(bbox[1] * sy)
        x2, y2 = int(bbox[2] * sx), int(bbox[3] * sy)

        # Bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_GREEN, 2)

        # Keypoints and skeleton
        kps = person.keypoints
        points = []

        for i in range(17):
            if kps[i, 2] > 0.3:  # Confidence threshold
                px = int(kps[i, 0] * sx)
                py = int(kps[i, 1] * sy)
                points.append((px, py))
                cv2.circle(frame, (px, py), 3, COLOR_CYAN, -1)
            else:
                points.append(None)

        # Draw skeleton connections
        for (i, j) in SKELETON_CONNECTIONS:
            if points[i] is not None and points[j] is not None:
                cv2.line(frame, points[i], points[j], COLOR_GREEN, 1, cv2.LINE_AA)

    def _draw_phone(
        self, frame: np.ndarray,
        phone: PhoneDetection,
        sx: float, sy: float
    ):
        """Draw phone bounding box on frame."""
        bbox = phone.bbox
        x1, y1 = int(bbox[0] * sx), int(bbox[1] * sy)
        x2, y2 = int(bbox[2] * sx), int(bbox[3] * sy)

        cv2.rectangle(frame, (x1, y1), (x2, y2), COLOR_ORANGE, 2)
        cv2.putText(
            frame, "PHONE",
            (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, COLOR_ORANGE, 1
        )

    def close(self):
        """Close the display window."""
        if self._window_created:
            cv2.destroyAllWindows()
            self._window_created = False
            logger.info("Display closed")
