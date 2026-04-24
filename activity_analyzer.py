"""
activity_analyzer.py — Heuristic Activity Detection

Analyzes pose estimation and object detection results to identify
undesirable employee activities: sleeping, chatting, phone usage.
"""

import logging
from dataclasses import dataclass
from datetime import datetime
from math import atan2, degrees, sqrt
from typing import List, Optional

import numpy as np

from detector import ChannelDetections, PersonDetection

logger = logging.getLogger(__name__)


# -------------------------------------------------------------------
# COCO Pose Keypoint Indices
# -------------------------------------------------------------------
NOSE = 0
LEFT_EYE = 1
RIGHT_EYE = 2
LEFT_EAR = 3
RIGHT_EAR = 4
LEFT_SHOULDER = 5
RIGHT_SHOULDER = 6
LEFT_ELBOW = 7
RIGHT_ELBOW = 8
LEFT_WRIST = 9
RIGHT_WRIST = 10
LEFT_HIP = 11
RIGHT_HIP = 12
LEFT_KNEE = 13
RIGHT_KNEE = 14
LEFT_ANKLE = 15
RIGHT_ANKLE = 16


@dataclass
class ActivityEvent:
    """A detected activity occurrence in a single frame."""
    channel: int
    activity_type: str      # "sleeping" | "chatting" | "phone_usage"
    persons_involved: int
    confidence: float
    timestamp: datetime
    frame: Optional[np.ndarray] = None  # Screenshot for Telegram


def _kp_valid(keypoints: np.ndarray, idx: int, min_conf: float = 0.3) -> bool:
    """Check if a keypoint has sufficient confidence."""
    return keypoints[idx, 2] >= min_conf


def _kp_xy(keypoints: np.ndarray, idx: int) -> tuple:
    """Extract (x, y) from keypoint array."""
    return float(keypoints[idx, 0]), float(keypoints[idx, 1])


def _distance(p1: tuple, p2: tuple) -> float:
    """Euclidean distance between two (x, y) points."""
    return sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def _bbox_center(bbox: np.ndarray) -> tuple:
    """Get center point of a bounding box [x1, y1, x2, y2]."""
    cx = (bbox[0] + bbox[2]) / 2
    cy = (bbox[1] + bbox[3]) / 2
    return float(cx), float(cy)


# ===================================================================
# SLEEP DETECTOR
# ===================================================================

class SleepDetector:
    """Detects sleeping posture based on head tilt and head drop.

    Heuristics:
    1. Head tilt: Angle of eye-to-eye line relative to horizontal axis.
       If > threshold, head is tilted sideways (resting on desk/shoulder).
    2. Head drop: Nose position relative to shoulder midpoint.
       If nose is close to or below shoulders, person is slumped over.
    """

    def __init__(self, tilt_threshold: float = 30.0, drop_threshold: float = -0.3):
        self.tilt_threshold = tilt_threshold
        self.drop_threshold = drop_threshold

    def detect(self, person: PersonDetection) -> Optional[float]:
        """Check if person appears to be sleeping.

        Args:
            person: Detected person with keypoints.

        Returns:
            Confidence score (0-1) if sleeping detected, None otherwise.
        """
        kps = person.keypoints
        score = 0.0
        checks = 0

        # Check 1: Head tilt via eye angle
        if _kp_valid(kps, LEFT_EYE) and _kp_valid(kps, RIGHT_EYE):
            left_eye = _kp_xy(kps, LEFT_EYE)
            right_eye = _kp_xy(kps, RIGHT_EYE)

            delta_y = right_eye[1] - left_eye[1]
            delta_x = right_eye[0] - left_eye[0]

            if abs(delta_x) > 1e-6:  # Avoid division by zero
                angle = abs(degrees(atan2(delta_y, delta_x)))
                if angle > self.tilt_threshold:
                    score += 0.6
            checks += 1

        # Check 2: Head drop — nose relative to shoulders
        if (_kp_valid(kps, NOSE) and
                _kp_valid(kps, LEFT_SHOULDER) and
                _kp_valid(kps, RIGHT_SHOULDER)):

            nose = _kp_xy(kps, NOSE)
            left_sh = _kp_xy(kps, LEFT_SHOULDER)
            right_sh = _kp_xy(kps, RIGHT_SHOULDER)

            shoulder_mid_y = (left_sh[1] + right_sh[1]) / 2
            shoulder_span = abs(right_sh[0] - left_sh[0])

            if shoulder_span > 1e-6:
                # Negative = nose above shoulders, positive = nose below
                head_drop = (nose[1] - shoulder_mid_y) / shoulder_span
                if head_drop > self.drop_threshold:
                    score += 0.5
            checks += 1

        # Check 3: Body horizontal — lying down (shoulders at same level as hips)
        if (_kp_valid(kps, LEFT_SHOULDER) and _kp_valid(kps, RIGHT_SHOULDER) and
                _kp_valid(kps, LEFT_HIP) and _kp_valid(kps, RIGHT_HIP)):

            sh_mid_y = (kps[LEFT_SHOULDER, 1] + kps[RIGHT_SHOULDER, 1]) / 2
            hip_mid_y = (kps[LEFT_HIP, 1] + kps[RIGHT_HIP, 1]) / 2
            body_height = abs(hip_mid_y - sh_mid_y)

            sh_mid_x = (kps[LEFT_SHOULDER, 0] + kps[RIGHT_SHOULDER, 0]) / 2
            hip_mid_x = (kps[LEFT_HIP, 0] + kps[RIGHT_HIP, 0]) / 2
            body_width = abs(hip_mid_x - sh_mid_x)

            # If body is more horizontal than vertical
            if body_height > 1e-6 and body_width / body_height > 1.5:
                score += 0.7
            checks += 1

        if checks == 0:
            return None

        final_score = min(score, 1.0)
        return final_score if final_score >= 0.4 else None


# ===================================================================
# CHAT DETECTOR
# ===================================================================

class ChatDetector:
    """Detects chatting by analyzing proximity between multiple persons.

    Heuristics:
    1. Two or more persons within close proximity (< 30% frame width).
    2. Persons are facing each other based on nose/shoulder orientation.
    """

    def __init__(self, proximity_ratio: float = 0.30):
        self.proximity_ratio = proximity_ratio

    def detect(
        self,
        persons: List[PersonDetection],
        frame_width: int
    ) -> Optional[tuple]:
        """Check if two or more persons appear to be chatting.

        Args:
            persons: List of detected persons.
            frame_width: Width of the frame in pixels.

        Returns:
            Tuple of (num_persons, confidence) if chatting detected, None otherwise.
        """
        if len(persons) < 2:
            return None

        proximity_threshold = frame_width * self.proximity_ratio
        chatting_pairs = 0
        total_conf = 0.0

        for i in range(len(persons)):
            for j in range(i + 1, len(persons)):
                p1_kps = persons[i].keypoints
                p2_kps = persons[j].keypoints

                # Get body centers from shoulder/hip midpoints
                c1 = self._body_center(p1_kps)
                c2 = self._body_center(p2_kps)

                if c1 is None or c2 is None:
                    continue

                dist = _distance(c1, c2)

                if dist < proximity_threshold:
                    facing_score = self._facing_score(p1_kps, p2_kps)
                    if facing_score > 0.3:
                        chatting_pairs += 1
                        total_conf += facing_score

        if chatting_pairs > 0:
            avg_conf = min(total_conf / chatting_pairs, 1.0)
            num_involved = min(chatting_pairs + 1, len(persons))
            return num_involved, avg_conf

        return None

    def _body_center(self, kps: np.ndarray) -> Optional[tuple]:
        """Calculate approximate body center from shoulders and hips."""
        valid_points = []

        for idx in [LEFT_SHOULDER, RIGHT_SHOULDER, LEFT_HIP, RIGHT_HIP]:
            if _kp_valid(kps, idx):
                valid_points.append(_kp_xy(kps, idx))

        if len(valid_points) < 2:
            # Fallback: use bbox or available upper body points
            for idx in [NOSE, LEFT_EYE, RIGHT_EYE]:
                if _kp_valid(kps, idx):
                    valid_points.append(_kp_xy(kps, idx))

        if not valid_points:
            return None

        cx = sum(p[0] for p in valid_points) / len(valid_points)
        cy = sum(p[1] for p in valid_points) / len(valid_points)
        return cx, cy

    def _facing_score(self, kps1: np.ndarray, kps2: np.ndarray) -> float:
        """Estimate how much two persons face each other.

        Uses nose position relative to shoulder midpoint to infer facing direction.
        If person 1's nose points toward person 2 and vice versa, they face each other.
        """
        score = 0.0

        # For each person, check if nose is on the side facing the other person
        for src_kps, tgt_kps in [(kps1, kps2), (kps2, kps1)]:
            if not (_kp_valid(src_kps, NOSE) and
                    _kp_valid(src_kps, LEFT_SHOULDER) and
                    _kp_valid(src_kps, RIGHT_SHOULDER)):
                score += 0.25  # Can't determine, give partial credit
                continue

            nose = _kp_xy(src_kps, NOSE)
            l_sh = _kp_xy(src_kps, LEFT_SHOULDER)
            r_sh = _kp_xy(src_kps, RIGHT_SHOULDER)
            sh_mid_x = (l_sh[0] + r_sh[0]) / 2

            # Get target body center
            tgt_center = self._body_center(tgt_kps)
            if tgt_center is None:
                score += 0.25
                continue

            # Nose offset from shoulder midpoint
            nose_offset = nose[0] - sh_mid_x
            # Direction to target
            target_dir = tgt_center[0] - sh_mid_x

            # Same sign = nose pointing toward target
            if nose_offset * target_dir > 0:
                score += 0.5

        return score


# ===================================================================
# PHONE USAGE DETECTOR
# ===================================================================

class PhoneDetector:
    """Detects phone usage by combining phone object detection with pose analysis.

    Heuristics:
    1. Cell phone detected near a person's hands (wrist keypoints).
    2. Person's head is tilted downward (looking at phone).
    """

    def __init__(self, proximity_px: float = 100.0, require_phone_object: bool = True, pose_threshold: float = 0.5):
        self.proximity_px = proximity_px
        self.require_phone_object = require_phone_object
        self.pose_threshold = pose_threshold

    def detect(self, person: PersonDetection, phone_bboxes: list) -> Optional[float]:
        """Check if person is using a phone.

        Args:
            person: Detected person with keypoints.
            phone_bboxes: List of detected phone bounding boxes [(x1,y1,x2,y2), ...].

        Returns:
            Confidence score if phone usage detected, None otherwise.
        """
        kps = person.keypoints
        score = 0.0

        phone_near_hand = False

        if phone_bboxes:
            # Check if any phone is near person's hands
            wrist_indices = [LEFT_WRIST, RIGHT_WRIST]

            for phone_bbox in phone_bboxes:
                phone_center = _bbox_center(phone_bbox)

                # First check: is the phone within the person's bounding box area?
                p_bbox = person.bbox
                # Expand person bbox by 20%
                expand = 0.2
                w = p_bbox[2] - p_bbox[0]
                h = p_bbox[3] - p_bbox[1]
                expanded_bbox = [
                    p_bbox[0] - w * expand,
                    p_bbox[1] - h * expand,
                    p_bbox[2] + w * expand,
                    p_bbox[3] + h * expand
                ]

                if not (expanded_bbox[0] <= phone_center[0] <= expanded_bbox[2] and
                        expanded_bbox[1] <= phone_center[1] <= expanded_bbox[3]):
                    continue  # Phone not near this person

                # Check wrist proximity
                for wrist_idx in wrist_indices:
                    if _kp_valid(kps, wrist_idx):
                        wrist = _kp_xy(kps, wrist_idx)
                        dist = _distance(wrist, phone_center)
                        if dist < self.proximity_px:
                            phone_near_hand = True
                            score += 0.5
                            break

                if phone_near_hand:
                    break

            if not phone_near_hand:
                # Fallback: phone inside person bbox even without wrist detection
                for phone_bbox in phone_bboxes:
                    phone_center = _bbox_center(phone_bbox)
                    p_bbox = person.bbox
                    if (p_bbox[0] <= phone_center[0] <= p_bbox[2] and
                            p_bbox[1] <= phone_center[1] <= p_bbox[3]):
                        score += 0.3
                        phone_near_hand = True
                        break

        # --- POSE HEURISTICS (Fallback/Supplement) ---
        # Even if YOLO didn't detect a phone object, the person might be holding one
        
        # Need a scale reference to make distances relative to person size
        scale = 100.0
        if _kp_valid(kps, LEFT_SHOULDER) and _kp_valid(kps, RIGHT_SHOULDER):
            l_sh = _kp_xy(kps, LEFT_SHOULDER)
            r_sh = _kp_xy(kps, RIGHT_SHOULDER)
            scale_val = _distance(l_sh, r_sh)
            if scale_val > 10: 
                scale = scale_val
                
        wrist_pts = []
        if _kp_valid(kps, LEFT_WRIST): wrist_pts.append(_kp_xy(kps, LEFT_WRIST))
        if _kp_valid(kps, RIGHT_WRIST): wrist_pts.append(_kp_xy(kps, RIGHT_WRIST))
        
        head_pts = []
        for idx in [NOSE, LEFT_EAR, RIGHT_EAR, LEFT_EYE, RIGHT_EYE]:
            if _kp_valid(kps, idx):
                head_pts.append(_kp_xy(kps, idx))
                
        # Heuristic 1: Calling (Hand to ear/face)
        if wrist_pts and head_pts:
            min_dist = float('inf')
            for w in wrist_pts:
                for h in head_pts:
                    d = _distance(w, h)
                    if d < min_dist:
                        min_dist = d
                        
            # If wrist is close to head 
            if min_dist < scale * 0.8:
                score += 0.4
                if min_dist < scale * 0.5:
                    score += 0.2

        # Heuristic 2: Texting / Browsing
        # Check head down posture (looking at phone)
        head_down = False
        if (_kp_valid(kps, NOSE) and
                _kp_valid(kps, LEFT_SHOULDER) and
                _kp_valid(kps, RIGHT_SHOULDER)):

            nose = _kp_xy(kps, NOSE)
            l_sh = _kp_xy(kps, LEFT_SHOULDER)
            r_sh = _kp_xy(kps, RIGHT_SHOULDER)

            # Midpoint between ears, or fallback to eyes
            ref_y = None
            if _kp_valid(kps, LEFT_EAR) and _kp_valid(kps, RIGHT_EAR):
                ref_y = (kps[LEFT_EAR, 1] + kps[RIGHT_EAR, 1]) / 2
            elif _kp_valid(kps, LEFT_EYE) and _kp_valid(kps, RIGHT_EYE):
                ref_y = (kps[LEFT_EYE, 1] + kps[RIGHT_EYE, 1]) / 2

            shoulder_mid_y = (l_sh[1] + r_sh[1]) / 2

            # Nose significantly below eye/ear level = looking down
            if ref_y is not None and nose[1] > ref_y + 10:
                head_down = True
                score += 0.2

            # Nose getting close to shoulder level
            shoulder_span = abs(r_sh[0] - l_sh[0])
            if shoulder_span > 1e-6:
                head_drop = (nose[1] - shoulder_mid_y) / shoulder_span
                if head_drop > -0.5:
                    head_down = True
                    score += 0.1
                    
        # If head is down, check if hands are in front of body (texting)
        if head_down and _kp_valid(kps, LEFT_WRIST) and _kp_valid(kps, RIGHT_WRIST):
            lw = _kp_xy(kps, LEFT_WRIST)
            rw = _kp_xy(kps, RIGHT_WRIST)
            hands_dist = _distance(lw, rw)
            
            # Hands are close together
            if hands_dist < scale * 1.5:
                if _kp_valid(kps, LEFT_HIP) and _kp_valid(kps, LEFT_SHOULDER):
                    hip_y = kps[LEFT_HIP, 1]
                    sh_y = kps[LEFT_SHOULDER, 1]
                    hands_y = (lw[1] + rw[1]) / 2
                    
                    # Hands are between shoulders and hips height-wise
                    if sh_y < hands_y < hip_y:
                        score += 0.3

        # If we require a phone object but none was detected, return None
        if self.require_phone_object and not phone_near_hand:
            return None

        final_score = min(score, 1.0)
        
        # Use pose_threshold if no phone object was detected (and require_phone_object is False)
        # Otherwise use the default 0.5 threshold
        effective_threshold = self.pose_threshold if not phone_near_hand else 0.5
        
        return final_score if final_score >= effective_threshold else None


# ===================================================================
# COMBINED ANALYZER
# ===================================================================

class ActivityAnalyzer:
    """Combines all detectors to analyze frames for undesirable activities."""

    def __init__(self, config, activity_toggles: dict = None):
        """Initialize activity analyzer with optional activity toggles.

        Args:
            config: Configuration object with detection thresholds.
            activity_toggles: Dict of activity toggles, e.g.
                {"sleep": False, "chat": True, "phone": True}.
                Defaults: sleep=off, chat=on, phone=on.
        """
        self.activity_toggles = activity_toggles or {
            "sleep": False,
            "chat": True,
            "phone": True,
        }

        self.sleep_detector = SleepDetector(
            tilt_threshold=config.HEAD_TILT_THRESHOLD,
            drop_threshold=config.HEAD_DROP_THRESHOLD
        )
        self.chat_detector = ChatDetector(
            proximity_ratio=config.CHAT_PROXIMITY_RATIO
        )
        self.phone_detector = PhoneDetector(
            proximity_px=config.PHONE_PROXIMITY_PX,
            require_phone_object=config.REQUIRE_PHONE_OBJECT,
            pose_threshold=config.PHONE_POSE_THRESHOLD
        )

        logger.info(
            f"Activity toggles: sleep={'ON' if self.activity_toggles['sleep'] else 'OFF'}, "
            f"chat={'ON' if self.activity_toggles['chat'] else 'OFF'}, "
            f"phone={'ON' if self.activity_toggles['phone'] else 'OFF'}"
        )

    def analyze(
        self,
        channel: int,
        detections: ChannelDetections,
        frame: np.ndarray
    ) -> List[ActivityEvent]:
        """Analyze detections for a single channel.

        Args:
            channel: Camera channel number.
            detections: All detections for this channel.
            frame: Original frame for screenshot.

        Returns:
            List of detected ActivityEvents.
        """
        events = []
        now = datetime.now()
        frame_h, frame_w = frame.shape[:2]

        # --- Sleeping Detection ---
        if self.activity_toggles.get("sleep", False):
            for person in detections.persons:
                sleep_conf = self.sleep_detector.detect(person)
                if sleep_conf is not None:
                    events.append(ActivityEvent(
                        channel=channel,
                        activity_type="sleeping",
                        persons_involved=1,
                        confidence=sleep_conf,
                        timestamp=now,
                        frame=frame
                    ))

        # --- Chatting Detection ---
        if self.activity_toggles.get("chat", True):
            if len(detections.persons) >= 2:
                chat_result = self.chat_detector.detect(detections.persons, frame_w)
                if chat_result is not None:
                    num_persons, chat_conf = chat_result
                    events.append(ActivityEvent(
                        channel=channel,
                        activity_type="chatting",
                        persons_involved=num_persons,
                        confidence=chat_conf,
                        timestamp=now,
                        frame=frame
                    ))

        # --- Phone Usage Detection ---
        if self.activity_toggles.get("phone", True):
            phone_bboxes = [p.bbox for p in detections.phones]
            for person in detections.persons:
                phone_conf = self.phone_detector.detect(person, phone_bboxes)
                if phone_conf is not None:
                    events.append(ActivityEvent(
                        channel=channel,
                        activity_type="phone_usage",
                        persons_involved=1,
                        confidence=phone_conf,
                        timestamp=now,
                        frame=frame
                    ))

        return events

