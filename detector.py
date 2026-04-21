"""
detector.py — YOLO Inference Manager

Handles loading YOLO11 models and running batch inference
for pose estimation and object detection (cell phone).
"""

import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class PersonDetection:
    """Detected person with pose keypoints."""
    keypoints: np.ndarray      # Shape: (17, 3) — x, y, confidence per keypoint
    bbox: np.ndarray           # Shape: (4,) — x1, y1, x2, y2
    confidence: float


@dataclass
class PhoneDetection:
    """Detected cell phone."""
    bbox: np.ndarray           # Shape: (4,) — x1, y1, x2, y2
    confidence: float


@dataclass
class ChannelDetections:
    """All detections for a single camera channel."""
    persons: List[PersonDetection] = field(default_factory=list)
    phones: List[PhoneDetection] = field(default_factory=list)


class YOLODetector:
    """Manages YOLO11 models for pose estimation and object detection.

    Loads two models:
    - yolo11s-pose.pt for human pose estimation (17 keypoints)
    - yolo11s.pt for object detection (filtering for cell phones)
    """

    # COCO class index for cell phone
    PHONE_CLASS_ID = 67

    def __init__(self, confidence_threshold: float = 0.5, device: str = None):
        """Initialize YOLO detector.

        Args:
            confidence_threshold: Minimum confidence for detections.
            device: CUDA device ID or 'cpu'. Auto-detects if None.
        """
        self.confidence_threshold = confidence_threshold
        
        if device is None:
            import torch
            self.device = "0" if torch.cuda.is_available() else "cpu"
            if self.device == "cpu":
                logger.warning("CUDA not available. Falling back to CPU.")
        else:
            self.device = device
            
        self.pose_model = None
        self.detect_model = None

    def load_models(self):
        """Load YOLO11 models. Downloads automatically if not present."""
        from ultralytics import YOLO

        logger.info("Loading YOLO11s-pose model for pose estimation...")
        self.pose_model = YOLO("yolo11s-pose.pt")
        logger.info("Pose model loaded successfully")

        logger.info("Loading YOLO11s model for object detection...")
        self.detect_model = YOLO("yolo11s.pt")
        logger.info("Detection model loaded successfully")

    def detect(self, frames: Dict[int, np.ndarray]) -> Dict[int, ChannelDetections]:
        """Run inference on frames from multiple channels.

        Processes all frames through both pose estimation and object detection.

        Args:
            frames: Dictionary mapping channel number to frame (numpy array).

        Returns:
            Dictionary mapping channel number to ChannelDetections.
        """
        if not frames:
            return {}

        channels = list(frames.keys())
        frame_list = [frames[ch] for ch in channels]

        # Run pose estimation on all frames
        pose_results = self.pose_model(
            frame_list,
            conf=self.confidence_threshold,
            device=self.device,
            verbose=False
        )

        # Run object detection on all frames (cell phone only)
        detect_results = self.detect_model(
            frame_list,
            conf=self.confidence_threshold,
            classes=[self.PHONE_CLASS_ID],
            device=self.device,
            verbose=False
        )

        # Assemble results per channel
        results: Dict[int, ChannelDetections] = {}

        for i, ch in enumerate(channels):
            detections = ChannelDetections()

            # Extract pose detections
            pose_result = pose_results[i]
            if pose_result.keypoints is not None and len(pose_result.keypoints) > 0:
                keypoints_data = pose_result.keypoints.data.cpu().numpy()
                boxes_data = pose_result.boxes

                for j in range(len(keypoints_data)):
                    kps = keypoints_data[j]             # (17, 3)
                    bbox = boxes_data.xyxy[j].cpu().numpy()  # (4,)
                    conf = float(boxes_data.conf[j].cpu())

                    detections.persons.append(PersonDetection(
                        keypoints=kps,
                        bbox=bbox,
                        confidence=conf
                    ))

            # Extract phone detections
            detect_result = detect_results[i]
            if detect_result.boxes is not None and len(detect_result.boxes) > 0:
                for j in range(len(detect_result.boxes)):
                    bbox = detect_result.boxes.xyxy[j].cpu().numpy()
                    conf = float(detect_result.boxes.conf[j].cpu())

                    detections.phones.append(PhoneDetection(
                        bbox=bbox,
                        confidence=conf
                    ))

            results[ch] = detections

        return results
