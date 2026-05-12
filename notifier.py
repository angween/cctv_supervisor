"""
notifier.py — Telegram Notification Sender

Sends violation alerts with screenshots to Telegram using the Bot API.
Uses simple HTTP POST via `requests` library.
"""

import io
import logging
from datetime import datetime
from typing import Optional

import cv2
import numpy as np
import requests

from violation_tracker import Violation

logger = logging.getLogger(__name__)

# Human-readable activity names (Indonesian)
ACTIVITY_NAMES = {
    "sleeping": "Tidur",
    "chatting": "Mengobrol",
    "phone_usage": "Bermain HP",
}


class TelegramNotifier:
    """Sends violation notifications to Telegram with screenshots."""

    BASE_URL = "https://api.telegram.org/bot{token}"

    def __init__(self, bot_token: str, chat_ids: list, max_retries: int = 3, cameras: list = None, webhook_url: str = "", webhook_auth: str = ""):
        """Initialize Telegram and Webhook notifier.

        Args:
            bot_token: Telegram Bot API token.
            chat_ids: List of target chat/user IDs for notifications.
            max_retries: Maximum retry attempts for failed sends.
            cameras: Optional list of camera configurations for name lookup.
            webhook_url: Optional URL for webhook notifications.
            webhook_auth: Optional Authorization header for webhook.
        """
        self.bot_token = bot_token
        self.chat_ids = chat_ids
        self.max_retries = max_retries
        self.base_url = self.BASE_URL.format(token=bot_token)
        self.cameras = cameras or []
        self.camera_names = {cam.get("channel"): cam.get("name", f"Channel {cam.get('channel')}") for cam in self.cameras}
        self.webhook_url = webhook_url
        self.webhook_auth = webhook_auth

    def send_violation(self, violation: Violation) -> bool:
        """Send a violation alert to Telegram with screenshot.

        Args:
            violation: The Violation object to report.

        Returns:
            True if message sent successfully, False otherwise.
        """
        activity_name = ACTIVITY_NAMES.get(
            violation.activity_type, violation.activity_type
        )
        
        camera_name = self.camera_names.get(violation.channel, f"Channel {violation.channel}")

        caption = (
            f"🚨 *PELANGGARAN TERDETEKSI*\n\n"
            f"📹 Kamera: {camera_name}\n"
            f"⚠️ Aktivitas: *{activity_name}*\n"
            f"⏱ Durasi: {violation.duration:.0f} detik\n"
            f"🕐 Waktu mulai: {violation.start_time.strftime('%H:%M:%S')}\n"
            f"🕐 Waktu akhir: {violation.end_time.strftime('%H:%M:%S')}\n"
            f"📅 Tanggal: {violation.end_time.strftime('%d/%m/%Y')}"
        )

        telegram_success = False
        if violation.screenshot is not None:
            telegram_success = self._send_photo(violation.screenshot, caption)
        else:
            telegram_success = self._send_message(caption)

        webhook_success = self._send_webhook(violation, camera_name, activity_name, caption)
        
        return telegram_success or webhook_success

    def send_status(self, message: str) -> bool:
        """Send a simple status message.

        Args:
            message: Status message text.

        Returns:
            True if sent successfully.
        """
        return self._send_message(message)

    def test_connection(self) -> bool:
        """Test Telegram bot connection by sending a test message.

        Returns:
            True if bot is reachable and message sent successfully.
        """
        try:
            url = f"{self.base_url}/getMe"
            response = requests.get(url, timeout=10)

            if response.status_code == 200 and response.json().get("ok"):
                bot_info = response.json()["result"]
                logger.info(
                    f"Telegram bot connected: @{bot_info.get('username', 'unknown')}"
                )
                # return self._send_message(
                #     "✅ *CCTV Supervisor Online*\n\n"
                #     f"Sistem monitoring aktif.\n"
                #     f"🕐 {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
                # )
            else:
                logger.error(f"Telegram bot auth failed: {response.text}")
                return False

        except Exception as e:
            logger.error(f"Telegram connection test failed: {e}")
            return False

    def _send_message(self, text: str) -> bool:
        """Send a text message via Telegram Bot API to all chat IDs."""
        if not self.chat_ids:
            logger.warning("No Telegram chat IDs configured")
            return False

        url = f"{self.base_url}/sendMessage"
        any_success = False

        for chat_id in self.chat_ids:
            payload = {
                "chat_id": chat_id,
                "text": text,
                "parse_mode": "Markdown"
            }

            for attempt in range(1, self.max_retries + 1):
                try:
                    response = requests.post(url, json=payload, timeout=15)

                    if response.status_code == 200 and response.json().get("ok"):
                        logger.debug(f"Telegram message sent successfully to {chat_id}")
                        any_success = True
                        break
                    else:
                        logger.warning(
                            f"Telegram send failed to {chat_id} (attempt {attempt}): {response.text}"
                        )

                except requests.RequestException as e:
                    logger.warning(
                        f"Telegram send error to {chat_id} (attempt {attempt}): {e}"
                    )

        if not any_success:
            logger.error("Telegram message send failed for all chat IDs")
        
        return any_success

    def _send_photo(self, frame: np.ndarray, caption: str) -> bool:
        """Send a photo with caption via Telegram Bot API to all chat IDs.

        Args:
            frame: OpenCV frame (numpy array) to send as photo.
            caption: Caption text for the photo.
        """
        if not self.chat_ids:
            logger.warning("No Telegram chat IDs configured")
            return False

        url = f"{self.base_url}/sendPhoto"

        # Encode frame as JPEG
        success, buffer = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
        if not success:
            logger.error("Failed to encode frame to JPEG")
            return self._send_message(caption)  # Fallback to text

        photo_bytes_data = buffer.tobytes()
        any_success = False

        for chat_id in self.chat_ids:
            photo_bytes = io.BytesIO(photo_bytes_data)
            photo_bytes.name = "violation.jpg"

            for attempt in range(1, self.max_retries + 1):
                try:
                    photo_bytes.seek(0)
                    response = requests.post(
                        url,
                        data={
                            "chat_id": chat_id,
                            "caption": caption,
                            "parse_mode": "Markdown"
                        },
                        files={"photo": photo_bytes},
                        timeout=30
                    )

                    if response.status_code == 200 and response.json().get("ok"):
                        logger.debug(f"Telegram photo sent successfully to {chat_id}")
                        any_success = True
                        break
                    else:
                        logger.warning(
                            f"Telegram photo send failed to {chat_id} (attempt {attempt}): {response.text}"
                        )

                except requests.RequestException as e:
                    logger.warning(
                        f"Telegram photo send error to {chat_id} (attempt {attempt}): {e}"
                    )

        if not any_success:
            logger.error("Telegram photo send failed for all chat IDs")
            
        return any_success

    def _send_webhook(self, violation: Violation, camera_name: str, activity_name: str, caption: str) -> bool:
        """Send a violation alert to the configured Webhook."""
        if not self.webhook_url:
            return False

        headers = {}
        if self.webhook_auth:
            headers["Authorization"] = self.webhook_auth

        data = {
            "camera_name": camera_name,
            "activity": activity_name,
            "duration": str(int(violation.duration)),
            "start_time": violation.start_time.strftime('%H:%M:%S'),
            "end_time": violation.end_time.strftime('%H:%M:%S'),
            "date": violation.end_time.strftime('%d/%m/%Y'),
            "caption": caption
        }

        files = None
        if violation.screenshot is not None:
            success, buffer = cv2.imencode(".jpg", violation.screenshot, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if success:
                # Provide a dummy name, bytes, and content type
                files = {"photo": ("violation.jpg", buffer.tobytes(), "image/jpeg")}

        for attempt in range(1, self.max_retries + 1):
            try:
                # Re-create files dict each attempt because requests reads the stream
                if files and violation.screenshot is not None:
                    files = {"photo": ("violation.jpg", buffer.tobytes(), "image/jpeg")}
                    response = requests.post(self.webhook_url, headers=headers, data=data, files=files, timeout=30)
                else:
                    response = requests.post(self.webhook_url, headers=headers, data=data, timeout=15)

                if response.status_code in (200, 201, 202, 204):
                    logger.debug("Webhook message sent successfully")
                    return True
                else:
                    logger.warning(
                        f"Webhook send failed (attempt {attempt}): [{response.status_code}] {response.text}"
                    )

            except requests.RequestException as e:
                logger.warning(
                    f"Webhook send error (attempt {attempt}): {e}"
                )

        logger.error("Webhook send failed after all retries")
        return False
