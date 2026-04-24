import sys
import os
from unittest.mock import MagicMock, patch

# Add current directory to path so we can import our modules
sys.path.append(os.getcwd())

from notifier import TelegramNotifier
from violation_tracker import Violation
from datetime import datetime
import numpy as np

def test_notifier():
    bot_token = "test_token"
    chat_ids = ["111", "222", "333"]
    notifier = TelegramNotifier(bot_token, chat_ids)

    # Mock violation object
    violation = MagicMock(spec=Violation)
    violation.activity_type = "sleeping"
    violation.channel = 16
    violation.duration = 120.0
    violation.start_time = datetime.now()
    violation.end_time = datetime.now()
    violation.screenshot = None  # Test message first

    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"ok": True}

        print(f"Sending message to {chat_ids}...")
        success = notifier.send_violation(violation)
        
        print(f"Success: {success}")
        print(f"Total calls: {mock_post.call_count}")
        
        assert success == True
        assert mock_post.call_count == len(chat_ids)
        
        # Check if each chat_id was called
        called_ids = [call.kwargs['json']['chat_id'] for call in mock_post.call_args_list]
        print(f"Called IDs: {called_ids}")
        assert set(called_ids) == set(chat_ids)

    # Test with photo
    violation.screenshot = np.zeros((100, 100, 3), dtype=np.uint8)
    with patch('requests.post') as mock_post:
        mock_post.return_value.status_code = 200
        mock_post.return_value.json.return_value = {"ok": True}

        print(f"\nSending photo to {chat_ids}...")
        success = notifier.send_violation(violation)
        
        print(f"Success: {success}")
        print(f"Total calls: {mock_post.call_count}")
        
        assert success == True
        assert mock_post.call_count == len(chat_ids)
        
        # Check if each chat_id was called in 'data'
        called_ids = [call.kwargs['data']['chat_id'] for call in mock_post.call_args_list]
        print(f"Called IDs: {called_ids}")
        assert set(called_ids) == set(chat_ids)

    print("\nAll tests passed! [OK]")

if __name__ == "__main__":
    test_notifier()
