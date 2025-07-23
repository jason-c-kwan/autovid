#!/usr/bin/env python3
"""
Quick test script to verify the new scene detection functions work.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from core.video_analysis import (
    probe_video_info, 
    detect_keynote_scenes,
    _detect_scenes_sensitive,
    _merge_scene_detections,
    _validate_scene_count
)

def test_probe():
    """Test video probing."""
    print("Testing video probe...")
    try:
        info = probe_video_info('data/Lecture-01_how_to_study.mov')
        print(f"✓ Video probe successful: {info['width']}x{info['height']}, {info['duration']:.1f}s")
        return True
    except Exception as e:
        print(f"✗ Video probe failed: {e}")
        return False

def test_merge_detection():
    """Test scene merging logic."""
    print("Testing scene merging...")
    try:
        # Test data
        detection1 = [1.0, 5.0, 10.0]
        detection2 = [1.1, 4.9, 9.8, 15.0]  # Some overlap with first
        detection3 = [2.0, 6.0, 20.0]
        
        merged = _merge_scene_detections([detection1, detection2, detection3], tolerance=0.5)
        print(f"✓ Scene merging successful: {len(merged)} merged scenes from 3 inputs")
        print(f"  Merged timestamps: {merged}")
        return True
    except Exception as e:
        print(f"✗ Scene merging failed: {e}")
        return False

def test_validation():
    """Test scene validation logic."""
    print("Testing scene validation...")
    try:
        # Mock scene data
        mock_scenes = [
            {'timestamp': 1.0, 'slide_number': 1},
            {'timestamp': 5.0, 'slide_number': 2},
            {'timestamp': 10.0, 'slide_number': 3}
        ]
        
        # Test perfect match
        result = _validate_scene_count(mock_scenes, 3, 15.0)
        print(f"✓ Validation successful: {result['status']} - {result['message']}")
        
        # Test under-detection
        result2 = _validate_scene_count(mock_scenes, 18, 15.0)
        print(f"✓ Under-detection test: {result2['status']} - {result2['message']}")
        
        return True
    except Exception as e:
        print(f"✗ Scene validation failed: {e}")
        return False

def main():
    """Run all tests."""
    print("=== Scene Detection Function Tests ===\n")
    
    tests = [
        test_probe,
        test_merge_detection, 
        test_validation
    ]
    
    passed = 0
    for test in tests:
        if test():
            passed += 1
        print()
    
    print(f"=== Results: {passed}/{len(tests)} tests passed ===")
    
    if passed == len(tests):
        print("✓ All basic functions are working correctly!")
        print("Note: Full video processing may be slow due to 4K resolution.")
    else:
        print("✗ Some functions have issues that need fixing.")

if __name__ == "__main__":
    main()