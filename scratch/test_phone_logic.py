import numpy as np
import sys
import os

# Mocking PersonDetection to avoid imports if needed, but let's try importing
sys.path.append(os.getcwd())
from detector import PersonDetection
from activity_analyzer import PhoneDetector

def test_logic():
    # Mock person keypoints (normalized or pixel space doesn't matter for logic check)
    # We just need enough to trigger some heuristics or not
    kps = np.zeros((17, 3)) 
    # Nose at (100, 100)
    kps[0] = [100, 150, 1.0] # Nose (down from eyes)
    kps[1] = [90, 100, 1.0]  # Left Eye
    kps[2] = [110, 100, 1.0] # Right Eye
    kps[5] = [80, 200, 1.0]  # L Shoulder
    kps[6] = [120, 200, 1.0] # R Shoulder
    kps[9] = [95, 300, 1.0]  # L Wrist (near body)
    kps[10] = [105, 300, 1.0] # R Wrist (near body)
    kps[11] = [90, 500, 1.0]  # L Hip
    kps[12] = [110, 500, 1.0] # R Hip
    
    person = PersonDetection(
        keypoints=kps,
        bbox=np.array([50, 50, 150, 600]),
        confidence=0.9
    )
    
    # Test Case 1: Require Object = True, Phone Present
    detector = PhoneDetector(proximity_px=100.0, require_phone_object=True, pose_threshold=0.8)
    phone_bboxes = [np.array([90, 290, 110, 310])] # Near wrists
    score = detector.detect(person, phone_bboxes)
    print(f"Test 1 (Require Obj=True, Phone Present): {score}")
    assert score is not None
    
    # Test Case 2: Require Object = True, Phone Absent
    score = detector.detect(person, [])
    print(f"Test 2 (Require Obj=True, Phone Absent): {score}")
    assert score is None
    
    # Test Case 3: Require Object = False, Phone Absent, Pose High
    # In the current logic, our mock person has head down and hands together
    detector_no_req = PhoneDetector(proximity_px=100.0, require_phone_object=False, pose_threshold=0.8)
    score = detector_no_req.detect(person, [])
    print(f"Test 3 (Require Obj=False, Phone Absent, Pose High): {score}")
    # With pose_threshold=0.8, it should fail if pose score is ~0.6
    # Let's see what the score is. 
    # head_down gives +0.3 (actually +0.2 and +0.1 in logic)
    # hands together gives +0.3
    # total ~0.6. 0.6 < 0.8 -> None.
    
    # Test Case 4: Require Object = False, Phone Absent, Pose High, Lower Threshold
    detector_low_thresh = PhoneDetector(proximity_px=100.0, require_phone_object=False, pose_threshold=0.5)
    score = detector_low_thresh.detect(person, [])
    print(f"Test 4 (Require Obj=False, Phone Absent, Pose High, Thresh 0.5): {score}")
    assert score is not None

if __name__ == "__main__":
    test_logic()
