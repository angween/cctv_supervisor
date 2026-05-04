"""
main.py — CCTV Supervisor Entry Point

Monitors multiple CCTV RTSP streams to detect employee activities
(sleeping, chatting, phone usage) during work hours.
Logs violations to CSV and sends alerts via Telegram.

Usage:
    python main.py --display true
    python main.py --batch_size 4 --duration_loop 5
    python main.py --activity sleep on --activity phone off
    python main.py  (defaults: no display, batch size 4, duration 5m)
"""

import argparse
import logging
import signal
import sys
import time
from datetime import datetime

from config import Config
from stream_manager import StreamManager
from detector import YOLODetector
from activity_analyzer import ActivityAnalyzer
from violation_tracker import ViolationTracker
from notifier import TelegramNotifier
from logger_csv import CSVLogger
from scheduler import WorkScheduler
from display import DisplayManager

# ───────────────────────────────────────────────────────────────────
# Logging Setup
# ───────────────────────────────────────────────────────────────────

LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)-20s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def setup_logging():
    """Configure logging to console and file."""
    logging.basicConfig(
        level=logging.INFO,
        format=LOG_FORMAT,
        datefmt=LOG_DATE_FORMAT,
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler("cctv_supervisor.log", encoding="utf-8"),
        ]
    )


# ───────────────────────────────────────────────────────────────────
# CLI Arguments
# ───────────────────────────────────────────────────────────────────

def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="CCTV Supervisor — Employee Activity Monitor",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Default batch processing, no display
  python main.py --display true           # With display
  python main.py --batch_size 2 --duration_loop 10
  python main.py --activity sleep on      # Enable sleep detection
  python main.py --activity phone off     # Disable phone detection
  python main.py --activity sleep on --activity chat off
        """
    )

    parser.add_argument(
        "--display",
        type=str,
        default="false",
        choices=["true", "false"],
        help="Show video stream display (default: false)"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Number of cameras to process simultaneously (default: 4)"
    )

    parser.add_argument(
        "--duration_loop",
        type=int,
        default=5,
        help="Duration in minutes to monitor each batch before rotating (default: 5)"
    )

    parser.add_argument(
        "--activity",
        nargs=2,
        action="append",
        metavar=("TYPE", "STATE"),
        help="Toggle activity detection: --activity [sleep|chat|phone] [on|off]. "
             "Defaults: sleep=off, chat=on, phone=on. Can be specified multiple times."
    )

    return parser.parse_args()


def parse_activity_toggles(activity_args) -> dict:
    """Parse --activity arguments into a dict of activity toggles.

    Defaults: sleep=False (off), chat=True (on), phone=True (on).
    """
    toggles = {
        "sleep": False,   # OFF by default
        "chat": True,     # ON by default
        "phone": True,    # ON by default
    }

    if activity_args is None:
        return toggles

    valid_types = {"sleep", "chat", "phone"}
    valid_states = {"on", "off"}

    for activity_type, state in activity_args:
        activity_type = activity_type.lower()
        state = state.lower()

        if activity_type not in valid_types:
            logging.error(
                f"Invalid activity type: '{activity_type}'. "
                f"Valid types: {', '.join(sorted(valid_types))}"
            )
            sys.exit(1)

        if state not in valid_states:
            logging.error(
                f"Invalid state: '{state}'. Valid states: on, off"
            )
            sys.exit(1)

        toggles[activity_type] = (state == "on")

    return toggles


# ───────────────────────────────────────────────────────────────────
# Main Application
# ───────────────────────────────────────────────────────────────────

class CCTVSupervisor:
    """Main application orchestrating all components."""

    def __init__(self, cameras: list, batch_size: int, duration_loop: int, enable_display: bool, activity_toggles: dict):
        self.cameras = cameras
        self.batch_size = batch_size
        self.duration_loop_sec = duration_loop * 60
        self.activity_toggles = activity_toggles
        self.enable_display = enable_display
        self.running = False
        self.logger = logging.getLogger("CCTVSupervisor")

        # Components (initialized in setup)
        self.stream_manager = None
        self.detector = None
        self.analyzer = None
        self.tracker = None
        self.notifier = None
        self.csv_logger = None
        self.scheduler = None
        self.display = None

    def setup(self):
        """Initialize all components."""
        self.logger.info("=" * 60)
        self.logger.info("  CCTV Supervisor — Initializing")
        self.logger.info("=" * 60)
        self.logger.info(f"Total Cameras: {len(self.cameras)}")
        self.logger.info(f"Batch Size: {self.batch_size}")
        self.logger.info(f"Rotation Duration: {self.duration_loop_sec}s")
        self.logger.info(f"Display: {self.enable_display}")
        self.logger.info(f"Activity detection: sleep={'ON' if self.activity_toggles['sleep'] else 'OFF'}, "
                         f"chat={'ON' if self.activity_toggles['chat'] else 'OFF'}, "
                         f"phone={'ON' if self.activity_toggles['phone'] else 'OFF'}")
        self.logger.info(f"RTSP Host: {Config.RTSP_HOST}")
        self.logger.info(f"Frame skip: {Config.FRAME_SKIP}")
        self.logger.info(f"Violation duration: {Config.VIOLATION_DURATION}s")
        self.logger.info(f"Confidence threshold: {Config.CONFIDENCE_THRESHOLD}")

        # 1. Work scheduler
        self.scheduler = WorkScheduler()
        self.logger.info(f"Schedule: {self.scheduler.get_status_message()}")

        # 2. Stream manager
        self.stream_manager = StreamManager(
            cameras=self.cameras,
            rtsp_url_builder=Config.build_rtsp_url,
            batch_size=self.batch_size,
            duration_loop_sec=self.duration_loop_sec,
            frame_skip=Config.FRAME_SKIP,
            rotation_callback=self.on_batch_rotated
        )

        # 3. YOLO detector
        self.detector = YOLODetector(
            confidence_threshold=Config.CONFIDENCE_THRESHOLD,
            device=None
        )
        self.logger.info("Loading YOLO models...")
        self.detector.load_models()
        self.logger.info("YOLO models loaded ✓")

        # 4. Activity analyzer
        self.analyzer = ActivityAnalyzer(Config, activity_toggles=self.activity_toggles)

        # 5. Violation tracker
        self.tracker = ViolationTracker(
            violation_duration=Config.VIOLATION_DURATION,
            gap_tolerance=Config.ACTIVITY_GAP_TOLERANCE,
            cooldown=Config.TELEGRAM_COOLDOWN
        )

        # 6. Telegram notifier
        self.notifier = TelegramNotifier(
            bot_token=Config.TELEGRAM_BOT_TOKEN,
            chat_ids=Config.TELEGRAM_CHAT_IDS,
            cameras=self.cameras
        )

        # 7. CSV logger
        self.csv_logger = CSVLogger(log_dir="logs")

        # 8. Display (optional)
        if self.enable_display:
            self.display = DisplayManager()

        self.logger.info("All components initialized ✓")

    def on_batch_rotated(self, current_batch: list):
        """Callback when stream manager rotates to a new batch of cameras."""
        # Cleanup tracker for cameras no longer in the active batch
        if self.tracker:
            active_channels = [cam.get("channel") for cam in current_batch]
            self.tracker.cleanup_inactive_channels(active_channels)
            
        # Notify via Telegram
        # if self.notifier:
        #     cam_list_text = "\n".join([f"• Ch {c.get('channel')}: {c.get('name', 'Unknown')}" for c in current_batch])
        #     self.notifier.send_status(
        #         f"🔄 *Rotasi Kamera*\n\n"
        #         f"Memantau batch kamera berikutnya:\n{cam_list_text}\n\n"
        #         f"🕐 {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
        #     )

    def start(self):
        """Start the monitoring system."""
        self.running = True

        # Test Telegram connection
        self.logger.info("Testing Telegram connection...")
        if self.notifier.test_connection():
            self.logger.info("Telegram connected ✓")
        else:
            self.logger.warning("Telegram connection failed — notifications may not work")

        # Start initial batch of RTSP streams
        self.logger.info("Starting RTSP stream capture...")
        self.stream_manager.rotate_batch()

        # Wait for initial connection
        self.logger.info("Waiting for streams to connect (5s)...")
        time.sleep(5)

        # Log connection status
        status = self.stream_manager.get_status()
        connected = sum(1 for v in status.values() if v)
        self.logger.info(
            f"Streams connected: {connected}/{len(status)} — "
            f"{status}"
        )

        self.logger.info("=" * 60)
        self.logger.info("  Monitoring Started")
        self.logger.info("=" * 60)

        # Main processing loop
        self._run_loop()

    def _run_loop(self):
        """Main processing loop."""
        loop_count = 0
        fps_timer = time.time()

        while self.running:
            try:
                # Check work schedule
                if not self.scheduler.is_active():
                    wait_seconds = self.scheduler.seconds_until_active()
                    self.logger.info(
                        f"Di luar jam kerja. {self.scheduler.get_status_message()}"
                    )

                    # If display is active, still show frames but skip detection
                    if self.enable_display:
                        # Show idle state
                        pass

                    # Sleep in short intervals to allow graceful shutdown
                    sleep_interval = min(wait_seconds, 60)
                    for _ in range(int(sleep_interval)):
                        if not self.running:
                            return
                        time.sleep(1)
                    continue
                    
                # Check for batch rotation
                self.stream_manager.update()

                # Get latest frames from all streams
                frames = self.stream_manager.get_latest_frames()

                if not frames:
                    time.sleep(0.1)
                    continue

                # Run YOLO inference
                detections = self.detector.detect(frames)

                # Analyze activities per channel
                all_events = []
                for ch, channel_detections in detections.items():
                    if ch in frames:
                        events = self.analyzer.analyze(
                            channel=ch,
                            detections=channel_detections,
                            frame=frames[ch]
                        )
                        all_events.extend(events)

                # Track violations
                violations = self.tracker.update(all_events)
                active_durations = self.tracker.get_active_durations()

                # Handle new violations
                for violation in violations:
                    # Log to CSV
                    self.csv_logger.log_violation(violation)

                    # Send Telegram notification
                    self.notifier.send_violation(violation)

                    self.logger.warning(
                        f"[VIOLATION] Ch{violation.channel} -- "
                        f"{violation.activity_type} ({violation.duration:.0f}s)"
                    )

                # Update display
                if self.enable_display and self.display:
                    stream_status = self.stream_manager.get_status()
                    should_continue = self.display.update(
                        frames=frames,
                        detections=detections,
                        events=all_events,
                        active_durations=active_durations,
                        stream_status=stream_status
                    )
                    if not should_continue:
                        self.logger.info("Display closed by user")
                        self.running = False
                        break

                # FPS logging (every 100 loops)
                loop_count += 1
                if loop_count % 100 == 0:
                    elapsed = time.time() - fps_timer
                    fps = 100 / elapsed if elapsed > 0 else 0
                    fps_timer = time.time()

                    self.logger.info(
                        f"Processing: {fps:.1f} loops/s | "
                        f"Streams: {len(frames)}/{min(self.batch_size, len(self.cameras))} | "
                        f"Events: {len(all_events)} | "
                        f"Active: {len(active_durations)}"
                    )

                # Small sleep to prevent CPU spin
                time.sleep(0.01)

            except Exception as e:
                self.logger.error(f"Error in main loop: {e}", exc_info=True)
                time.sleep(1)

    def shutdown(self):
        """Gracefully shut down all components."""
        self.logger.info("Shutting down CCTV Supervisor...")
        self.running = False

        if self.stream_manager:
            self.stream_manager.stop_all()

        if self.csv_logger:
            self.csv_logger.close()

        if self.display:
            self.display.close()

        # Send shutdown notification
        # if self.notifier:
        #     self.notifier.send_status(
        #         "⛔ *CCTV Supervisor Offline*\n\n"
        #         f"Sistem monitoring berhenti.\n"
        #         f"🕐 {datetime.now().strftime('%H:%M:%S %d/%m/%Y')}"
        #     )

        self.logger.info("CCTV Supervisor stopped ✓")


# ───────────────────────────────────────────────────────────────────
# Entry Point
# ───────────────────────────────────────────────────────────────────

def main():
    """Main entry point."""
    setup_logging()
    args = parse_args()

    cameras = Config.load_cameras()
    if not cameras:
        logging.error("No cameras configured in cameras.json. Exiting.")
        sys.exit(1)
        
    enable_display = args.display.lower() == "true"
    activity_toggles = parse_activity_toggles(args.activity)

    app = CCTVSupervisor(
        cameras=cameras,
        batch_size=args.batch_size,
        duration_loop=args.duration_loop,
        enable_display=enable_display,
        activity_toggles=activity_toggles
    )

    # Signal handler for graceful shutdown (Ctrl+C)
    def signal_handler(signum, frame):
        logging.info("Received interrupt signal")
        app.shutdown()
        sys.exit(0)

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    try:
        app.setup()
        app.start()
    except KeyboardInterrupt:
        pass
    except Exception as e:
        logging.error(f"Fatal error: {e}", exc_info=True)
    finally:
        app.shutdown()


if __name__ == "__main__":
    main()
