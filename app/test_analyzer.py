#!/usr/bin/env python
"""
Test script to analyze multiple WAV files directly without going through the API.
This will help us see if the issue is with the model or the API handling.
"""
import os
import sys
import glob
from pathlib import Path

# Add the project root to path
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Import the analysis function
from anomaly_detection_api import analyze_wav, load_trained_model

# List of files to test
test_files = [
    "/mnt/d/thesis_project/Transformer_Malfunction_Prediction/dataset/abnormal/abnormal_0dB_fan_id_00-00000000.wav",
    "/mnt/d/thesis_project/Transformer_Malfunction_Prediction/dataset/abnormal/abnormal_0dB_fan_id_00-00000001.wav",
    "/mnt/d/thesis_project/Transformer_Malfunction_Prediction/dataset/abnormal/abnormal_0dB_fan_id_00-00000002.wav"
]

# Also grab some normal files if they exist
normal_files = glob.glob("/mnt/d/thesis_project/Transformer_Malfunction_Prediction/dataset/normal/*.wav")
if normal_files:
    test_files.extend(normal_files[:3])  # Add up to 3 normal files

print(f"Found {len(test_files)} test files")

# Load the model
print("Loading model...")
if not load_trained_model():
    print("Failed to load model")
    sys.exit(1)

# Process each file
print("\n=== Starting Analysis of Test Files ===\n")
for i, file_path in enumerate(test_files):
    print(f"\nAnalyzing file {i+1}/{len(test_files)}: {os.path.basename(file_path)}")
    
    # Analyze the file
    result = analyze_wav(file_path)
    
    # Print results
    if "error" in result:
        print(f"Error: {result['error']}")
    else:
        print(f"File: {result['file_name']}")
        print(f"Expected: {result['expected_label']}")
        print(f"Prediction: {result['prediction']}")
        print(f"Raw Score: {result['raw_score']:.6f}")
        print(f"Calibrated Score: {result['anomaly_score']:.6f}")
        print(f"Threshold: {result['threshold']:.6f}")
        print(f"Correct? {result['prediction_correct']}")
        
        # Print frame stats
        if "frame_errors_stats" in result:
            stats = result["frame_errors_stats"]
            print(f"Frame Errors - Min: {stats['min']:.6f}, Max: {stats['max']:.6f}, Mean: {stats['mean']:.6f}")

print("\n=== Analysis Complete ===\n")