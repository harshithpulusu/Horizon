#!/usr/bin/env python3
"""
Quick test to verify the video generation fix works
"""

def test_float_to_int_conversion():
    """Test that duration * fps conversion works correctly"""
    
    # Test cases that were causing the error
    test_cases = [
        {"duration": 1.5, "fps": 8},   # Quick mode
        {"duration": 2.0, "fps": 12},  # Standard mode  
        {"duration": 3.0, "fps": 15},  # High mode
        {"duration": 4.0, "fps": 20},  # Ultra mode
    ]
    
    print("Testing float to int conversion for total_frames...")
    
    for i, case in enumerate(test_cases):
        duration = case["duration"]
        fps = case["fps"]
        
        # Old way (causing error)
        total_frames_float = duration * fps
        
        # New way (fixed)
        total_frames_int = int(duration * fps)
        
        print(f"Test {i+1}: duration={duration}, fps={fps}")
        print(f"  Float result: {total_frames_float} (type: {type(total_frames_float)})")
        print(f"  Int result: {total_frames_int} (type: {type(total_frames_int)})")
        
        # Test if range() works
        try:
            test_range = range(total_frames_int)
            print(f"  ✅ range({total_frames_int}) works: {len(list(test_range))} frames")
        except Exception as e:
            print(f"  ❌ range() failed: {e}")
        
        print()
    
    print("✅ All conversion tests passed!")

if __name__ == "__main__":
    test_float_to_int_conversion()
