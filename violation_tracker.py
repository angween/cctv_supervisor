"""
violation_tracker.py — State Machine for Violation Duration Tracking

Tracks how long each activity has been detected continuously per channel.
If an activity persists for >= VIOLATION_DURATION seconds, it triggers a violation.
"""

import logging
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np

from activity_analyzer import ActivityEvent

logger = logging.getLogger(__name__)


@dataclass
class Violation:
    """A confirmed violation that exceeded the duration threshold."""
    channel: int
    activity_type: str
    duration: float             # seconds
    start_time: datetime
    end_time: datetime
    screenshot: Optional[np.ndarray] = None


@dataclass
class _ActivityState:
    """Internal state for tracking a single activity on a single channel."""
    first_seen: Optional[datetime] = None
    last_seen: Optional[datetime] = None
    duration: float = 0.0        # accumulated duration in seconds
    reported: bool = False       # whether violation has been reported
    last_screenshot: Optional[np.ndarray] = None
    last_confidence: float = 0.0


class ViolationTracker:
    """Tracks activity durations and triggers violations when thresholds are exceeded.

    State machine per (channel, activity_type):
    - IDLE: No activity detected
    - TRACKING: Activity detected, accumulating duration
    - VIOLATED: Duration exceeded threshold, violation reported
    - COOLDOWN: After violation, suppress re-reporting for cooldown period
    """

    # Activity types to track
    ACTIVITY_TYPES = ["sleeping", "chatting", "phone_usage"]

    def __init__(
        self,
        violation_duration: float = 60.0,
        gap_tolerance: float = 5.0,
        cooldown: float = 300.0
    ):
        """Initialize violation tracker.

        Args:
            violation_duration: Seconds of continuous activity to trigger violation.
            gap_tolerance: Max gap (seconds) before resetting the activity counter.
            cooldown: Seconds to suppress duplicate violations after reporting.
        """
        self.violation_duration = violation_duration
        self.gap_tolerance = gap_tolerance
        self.cooldown = cooldown

        # State storage: (channel, activity_type) -> _ActivityState
        self._states: Dict[Tuple[int, str], _ActivityState] = {}

        # Cooldown tracking: (channel, activity_type) -> last_violation_time
        self._cooldowns: Dict[Tuple[int, str], datetime] = {}

    def _get_state(self, channel: int, activity_type: str) -> _ActivityState:
        """Get or create state for a channel/activity combination."""
        key = (channel, activity_type)
        if key not in self._states:
            self._states[key] = _ActivityState()
        return self._states[key]

    def _is_in_cooldown(self, channel: int, activity_type: str, now: datetime) -> bool:
        """Check if a channel/activity is in cooldown period."""
        key = (channel, activity_type)
        if key not in self._cooldowns:
            return False

        elapsed = (now - self._cooldowns[key]).total_seconds()
        return elapsed < self.cooldown

    def update(self, events: List[ActivityEvent]) -> List[Violation]:
        """Update tracker with new activity events and return any new violations.

        Args:
            events: List of ActivityEvents detected in the current frame.

        Returns:
            List of new Violations that exceeded the duration threshold.
        """
        now = datetime.now()
        violations = []

        # Collect which (channel, activity_type) pairs are active this frame
        active_pairs = set()
        event_map: Dict[Tuple[int, str], ActivityEvent] = {}

        for event in events:
            key = (event.channel, event.activity_type)
            active_pairs.add(key)
            # Keep the event with highest confidence if multiple
            if key not in event_map or event.confidence > event_map[key].confidence:
                event_map[key] = event

        # Update active states
        for key in active_pairs:
            channel, activity_type = key
            state = self._get_state(channel, activity_type)
            event = event_map[key]

            if state.first_seen is None:
                # Start tracking
                state.first_seen = now
                state.last_seen = now
                state.duration = 0.0
                state.reported = False
                logger.debug(
                    f"[Ch {channel}] Started tracking: {activity_type}"
                )
            else:
                # Continue tracking — check gap
                gap = (now - state.last_seen).total_seconds()
                if gap > self.gap_tolerance:
                    # Gap too large, reset
                    state.first_seen = now
                    state.duration = 0.0
                    state.reported = False
                    logger.debug(
                        f"[Ch {channel}] Reset tracking (gap {gap:.1f}s): {activity_type}"
                    )
                else:
                    state.duration += gap

            state.last_seen = now
            state.last_screenshot = event.frame
            state.last_confidence = event.confidence

            # Check if violation threshold exceeded
            if (state.duration >= self.violation_duration and
                    not state.reported and
                    not self._is_in_cooldown(channel, activity_type, now)):

                violation = Violation(
                    channel=channel,
                    activity_type=activity_type,
                    duration=state.duration,
                    start_time=state.first_seen,
                    end_time=now,
                    screenshot=state.last_screenshot
                )
                violations.append(violation)
                state.reported = True
                self._cooldowns[key] = now

                logger.warning(
                    f"[Ch {channel}] VIOLATION: {activity_type} "
                    f"for {state.duration:.0f}s"
                )

        # Decay inactive states (check gap tolerance)
        for key, state in list(self._states.items()):
            if key not in active_pairs and state.first_seen is not None:
                gap = (now - state.last_seen).total_seconds() if state.last_seen else 0
                if gap > self.gap_tolerance:
                    # Reset state
                    channel, activity_type = key
                    if state.duration > 0:
                        logger.debug(
                            f"[Ch {channel}] Activity ended after {state.duration:.1f}s: "
                            f"{activity_type}"
                        )
                    self._states[key] = _ActivityState()

        return violations

    def get_active_durations(self) -> Dict[Tuple[int, str], float]:
        """Get current duration of all actively tracked activities.

        Returns:
            Dictionary of (channel, activity_type) -> duration in seconds.
        """
        result = {}
        now = datetime.now()

        for key, state in self._states.items():
            if state.first_seen is not None and state.last_seen is not None:
                gap = (now - state.last_seen).total_seconds()
                if gap <= self.gap_tolerance:
                    result[key] = state.duration + gap

        return result
