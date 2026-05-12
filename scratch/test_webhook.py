import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
# Load .env explicitly
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '.env'))

import numpy as np
import cv2
from datetime import datetime, timedelta
from config import Config
from notifier import TelegramNotifier
from violation_tracker import Violation

def test():
    # Create dummy image
    frame = np.zeros((480, 640, 3), dtype=np.uint8)
    # Give it some color
    frame[:] = (50, 100, 200)
    cv2.putText(frame, "TEST WEBHOOK", (150, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 3)

    violation = Violation(
        channel=1,
        activity_type="phone_usage",
        duration=125.0,
        start_time=datetime.now() - timedelta(seconds=125),
        end_time=datetime.now()
    )
    violation.screenshot = frame

    notifier = TelegramNotifier(
        bot_token=Config.TELEGRAM_BOT_TOKEN,
        chat_ids=Config.TELEGRAM_CHAT_IDS,
        cameras=[{"channel": 1, "name": "Kamera Test"}],
        webhook_url='https://n8n.tjahaja-baru.com/webhook-test/cctv',
        webhook_auth=Config.WEBHOOK_AUTH
    )

    print(f"Webhook URL: https://n8n.tjahaja-baru.com/webhook-test/cctv")
    print(f"Webhook Auth: {Config.WEBHOOK_AUTH[:10]}...")

    print("Sending payload...")
    caption = (
        f"🚨 *PELANGGARAN TERDETEKSI (TEST)*\n\n"
        f"📹 Kamera: Kamera Test\n"
        f"⚠️ Aktivitas: *Test Webhook*\n"
        f"⏱ Durasi: 125 detik"
    )

    success = notifier._send_webhook(
        violation=violation,
        camera_name="Kamera Test",
        activity_name="Test Webhook",
        caption=caption
    )
    
    print(f"Webhook send success: {success}")

if __name__ == "__main__":
    test()
