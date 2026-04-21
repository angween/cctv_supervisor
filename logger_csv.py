"""
logger_csv.py — CSV Violation Logger

Logs violations to daily CSV files in the logs/ directory.
Thread-safe writing with file locking.
"""

import csv
import logging
import os
import threading
from datetime import datetime

from violation_tracker import Violation

logger = logging.getLogger(__name__)


class CSVLogger:
    """Logs violations to date-stamped CSV files.

    Generates one CSV file per day: logs/violations_YYYY-MM-DD.csv
    """

    FIELDNAMES = [
        "timestamp",
        "channel",
        "activity_type",
        "duration_seconds",
        "persons_involved",
        "confidence",
        "start_time",
        "end_time",
    ]

    def __init__(self, log_dir: str = "logs"):
        """Initialize CSV logger.

        Args:
            log_dir: Directory to store CSV log files.
        """
        self.log_dir = log_dir
        self._lock = threading.Lock()
        self._current_date: str = ""
        self._current_file = None
        self._writer = None

        # Ensure log directory exists
        os.makedirs(self.log_dir, exist_ok=True)

    def _get_filepath(self, date_str: str) -> str:
        """Get CSV file path for a given date."""
        return os.path.join(self.log_dir, f"violations_{date_str}.csv")

    def _ensure_file(self, date_str: str):
        """Ensure the correct CSV file is open for the current date."""
        if date_str != self._current_date:
            # Close previous file
            if self._current_file is not None:
                self._current_file.close()

            filepath = self._get_filepath(date_str)
            file_exists = os.path.exists(filepath)

            self._current_file = open(filepath, "a", newline="", encoding="utf-8")
            self._writer = csv.DictWriter(
                self._current_file, fieldnames=self.FIELDNAMES
            )

            # Write header if new file
            if not file_exists:
                self._writer.writeheader()
                self._current_file.flush()
                logger.info(f"Created new log file: {filepath}")

            self._current_date = date_str

    def log_violation(self, violation: Violation, persons_involved: int = 1,
                      confidence: float = 0.0):
        """Write a violation record to the CSV log.

        Args:
            violation: The Violation to log.
            persons_involved: Number of persons involved.
            confidence: Detection confidence score.
        """
        with self._lock:
            now = datetime.now()
            date_str = now.strftime("%Y-%m-%d")

            try:
                self._ensure_file(date_str)

                row = {
                    "timestamp": now.strftime("%Y-%m-%d %H:%M:%S"),
                    "channel": violation.channel,
                    "activity_type": violation.activity_type,
                    "duration_seconds": round(violation.duration, 1),
                    "persons_involved": persons_involved,
                    "confidence": round(confidence, 3),
                    "start_time": violation.start_time.strftime("%H:%M:%S"),
                    "end_time": violation.end_time.strftime("%H:%M:%S"),
                }

                self._writer.writerow(row)
                self._current_file.flush()

                logger.info(
                    f"Logged violation: Ch{violation.channel} "
                    f"{violation.activity_type} ({violation.duration:.0f}s)"
                )

            except Exception as e:
                logger.error(f"Failed to write CSV log: {e}")

    def close(self):
        """Close the current CSV file."""
        with self._lock:
            if self._current_file is not None:
                self._current_file.close()
                self._current_file = None
                self._writer = None
                logger.info("CSV logger closed")
