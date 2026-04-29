"""
config.py — Configuration & Environment Loader

Loads environment variables from .env file and provides
centralized configuration for the CCTV Supervisor application.
"""

import json
import os
from datetime import datetime, time
from dotenv import load_dotenv

# Load .env file
load_dotenv()


class Config:
    """Centralized configuration loaded from environment variables."""

    # RTSP Configuration
    RTSP_USER = os.getenv("RTSP_USER", "admin")
    RTSP_PASSWORD = os.getenv("RTSP_PASSWORD", "admin123")
    RTSP_HOST = os.getenv("RTSP_HOST", "172.16.0.187:554")
    RTSP_QUALITY = os.getenv("RTSP_QUALITY", "02")

    # Telegram Configuration
    TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
    TELEGRAM_CHAT_IDS = [
        id_str.strip() 
        for id_str in os.getenv("TELEGRAM_CHAT_ID", "").split(",") 
        if id_str.strip()
    ]
    TELEGRAM_COOLDOWN = int(os.getenv("TELEGRAM_COOLDOWN", "300"))  # 5 minutes

    # Detection Settings
    VIOLATION_DURATION = int(os.getenv("VIOLATION_DURATION", "60"))  # seconds
    FRAME_SKIP = int(os.getenv("FRAME_SKIP", "3"))
    CONFIDENCE_THRESHOLD = float(os.getenv("CONFIDENCE_THRESHOLD", "0.5"))

    # Work Schedule: {day_of_week: (start_time, end_time)} — Monday=0, Sunday=6
    WORK_SCHEDULE = {
        0: (time(7, 0), time(16, 0)),   # Senin
        1: (time(7, 0), time(16, 0)),   # Selasa
        2: (time(7, 0), time(16, 0)),   # Rabu
        3: (time(7, 0), time(16, 0)),   # Kamis
        4: (time(7, 0), time(16, 0)),   # Jumat
        5: (time(7, 0), time(12, 0)),   # Sabtu
        # 6 (Minggu) — not included = day off
    }

    # Activity Detection Thresholds
    HEAD_TILT_THRESHOLD = 30.0          # degrees — head tilt for sleeping
    HEAD_DROP_THRESHOLD = -0.3          # normalized — nose relative to shoulders
    CHAT_PROXIMITY_RATIO = 0.30         # fraction of frame width
    PHONE_PROXIMITY_PX = 100            # pixels — wrist to phone distance
    CHAT_PRE_DURATION = 5.0             # seconds before counting as chatting
    
    # Phone Detection Specifics
    REQUIRE_PHONE_OBJECT = True         # If True, must detect 'cell phone' object
    PHONE_POSE_THRESHOLD = 0.8          # threshold if REQUIRE_PHONE_OBJECT is False

    # Violation Tracking
    ACTIVITY_GAP_TOLERANCE = 5.0        # seconds — max gap before resetting counter

    @staticmethod
    def load_cameras() -> list:
        """Load camera configuration from cameras.json."""
        cameras_file = "cameras.json"
        if not os.path.exists(cameras_file):
            return []
        
        try:
            with open(cameras_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"Error loading {cameras_file}: {e}")
            return []

    @staticmethod
    def build_rtsp_url(camera_info: dict) -> str:
        """Build RTSP URL for a given camera configuration.

        Format: rtsp://{user}:{password}@{host}/Streaming/Channels/{channel}{quality}
        """
        # Allow complete URL override
        if "rtsp_url" in camera_info:
            return camera_info["rtsp_url"]
            
        # Get individual components, with fallbacks
        channel_id = camera_info.get("channel")
        # rtsp_channel allows overriding the physical channel number in the URL (e.g. 1 instead of 18)
        rtsp_channel = camera_info.get("rtsp_channel", channel_id)
        
        host = camera_info.get("host", Config.RTSP_HOST)
        user = camera_info.get("user", Config.RTSP_USER)
        password = camera_info.get("password", Config.RTSP_PASSWORD)
        
        # Channel format: first digits = channel number, last digit = quality
        channel_path = f"{rtsp_channel}{Config.RTSP_QUALITY}"
        return (
            f"rtsp://{user}:{password}"
            f"@{host}/Streaming/Channels/{channel_path}"
        )

    @staticmethod
    def is_work_hours(dt: datetime = None) -> bool:
        """Check if the given datetime falls within configured work hours.

        Args:
            dt: Datetime to check. Defaults to current time.

        Returns:
            True if within work hours, False otherwise.
        """
        if dt is None:
            dt = datetime.now()

        day = dt.weekday()  # Monday=0, Sunday=6
        schedule = Config.WORK_SCHEDULE.get(day)

        if schedule is None:
            return False  # Day off (e.g., Sunday)

        start, end = schedule
        current_time = dt.time()
        return start <= current_time <= end
