"""
scheduler.py — Work Hours Scheduler

Determines whether the current time falls within configured work hours.
Provides wait-until-active functionality for the main loop.
"""

import logging
from datetime import datetime, timedelta, time

from config import Config

logger = logging.getLogger(__name__)

# Day names in Indonesian for logging
DAY_NAMES = {
    0: "Senin",
    1: "Selasa",
    2: "Rabu",
    3: "Kamis",
    4: "Jumat",
    5: "Sabtu",
    6: "Minggu",
}


class WorkScheduler:
    """Manages work hour schedule for monitoring activation."""

    def __init__(self):
        self.schedule = Config.WORK_SCHEDULE
        self._last_status: bool = None

    def is_active(self) -> bool:
        """Check if current time is within work hours.

        Returns:
            True if monitoring should be active, False otherwise.
        """
        active = Config.is_work_hours()

        # Log status changes
        if self._last_status is not None and active != self._last_status:
            if active:
                logger.info("[INFO] Jam kerja dimulai -- monitoring AKTIF")
            else:
                logger.info("[INFO] Jam kerja selesai -- monitoring NONAKTIF")

        self._last_status = active
        return active

    def next_active_time(self) -> datetime:
        """Calculate when the next work period starts.

        Returns:
            Datetime of the next work period start.
        """
        now = datetime.now()

        # Check up to 7 days ahead
        for day_offset in range(8):
            check_date = now + timedelta(days=day_offset)
            day = check_date.weekday()
            schedule = self.schedule.get(day)

            if schedule is None:
                continue  # Day off

            start_time, end_time = schedule

            # Construct the start datetime for this day
            start_dt = check_date.replace(
                hour=start_time.hour,
                minute=start_time.minute,
                second=0,
                microsecond=0
            )

            # If it's today and we haven't passed the end time yet
            if day_offset == 0:
                end_dt = check_date.replace(
                    hour=end_time.hour,
                    minute=end_time.minute,
                    second=0,
                    microsecond=0
                )
                if now < end_dt:
                    # If before start, return start time
                    if now < start_dt:
                        return start_dt
                    else:
                        # We're currently in work hours
                        return now
            else:
                return start_dt

        # Fallback: next Monday at 07:00
        days_until_monday = (7 - now.weekday()) % 7
        if days_until_monday == 0:
            days_until_monday = 7
        next_monday = now + timedelta(days=days_until_monday)
        return next_monday.replace(hour=7, minute=0, second=0, microsecond=0)

    def seconds_until_active(self) -> float:
        """Calculate seconds until the next work period.

        Returns:
            Number of seconds to wait. 0 if currently active.
        """
        if self.is_active():
            return 0

        next_time = self.next_active_time()
        delta = (next_time - datetime.now()).total_seconds()
        return max(delta, 0)

    def get_status_message(self) -> str:
        """Get a human-readable status message about the schedule.

        Returns:
            Status string describing current state and next active time.
        """
        now = datetime.now()
        day_name = DAY_NAMES.get(now.weekday(), "?")

        if self.is_active():
            day = now.weekday()
            schedule = self.schedule.get(day)
            if schedule:
                end_time = schedule[1]
                return (
                    f"[AKTIF] Monitoring AKTIF | {day_name} | "
                    f"Selesai jam {end_time.strftime('%H:%M')}"
                )
            return f"[AKTIF] Monitoring AKTIF | {day_name}"

        next_time = self.next_active_time()
        next_day_name = DAY_NAMES.get(next_time.weekday(), "?")
        wait_seconds = self.seconds_until_active()
        hours = int(wait_seconds // 3600)
        minutes = int((wait_seconds % 3600) // 60)

        return (
            f"[PAUSE] Monitoring NONAKTIF | {day_name} | "
            f"Mulai lagi: {next_day_name} {next_time.strftime('%H:%M')} "
            f"({hours}j {minutes}m lagi)"
        )
