#!/usr/bin/env python
"""
Test script for the LTXV-Modal API.
This script runs a test video generation with a specific prompt.
"""

import os
import requests
import base64
import json
import argparse
import time

# The URL needs to be updated with your actual deployment URL
BASE_URL = "https://yourname--ltxv-video"

def test_generate_video(output_file="test_output.mp4", display=False):
    """Run a test video generation with a specific prompt."""
    # Test parameters
    prompt = "A frog holding a sign reading EPOCH, on a lilypad, appears to be real life footage"
    
    # API parameters
    params = {
        "prompt": prompt,
        "height": 736, # Lower resolution for faster test
        "width": 1280,
        "num_frames": 73, # Fewer frames for faster test
        "frame_rate": 25,
        "guidance_scale": 3.0,
        "num_inference_steps": 30, # Fewer steps for faster test
        "stg_scale": 1.0,
        "stg_rescale": 0.7
    }
    
    # First, get model info to check if using volume
    print("Checking model information...")
    info_url = f"{BASE_URL}-ltxvideomodel-info.modal.run"
    
    try:
        info_response = requests.get(info_url, timeout=300)
        if info_response.status_code == 200:
            model_info = info_response.json()
            print("\nModel Information:")
            print(f"- Name: {model_info.get('model', 'unknown')}")
            print(f"- Parameters: {model_info.get('parameters', 'unknown')}")
            print(f"- Source: {model_info.get('source', 'unknown')}")
            if 'source' in model_info and model_info['source'] == 'volume':
                print("✅ Model is loading from volume!")
            else:
                print("⚠️ Model is not loading from volume.")
            print(f"- Volume path: {model_info.get('volume_path', 'unknown')}")
        else:
            print(f"❌ Error getting model info: {info_response.status_code}")
    except Exception as e:
        print(f"❌ Error connecting to info endpoint: {e}")
    
    # Make the generation request
    url = f"{BASE_URL}-ltxvideomodel-generate-video-api.modal.run"
    
    print(f"\nGenerating video with prompt: '{prompt}'")
    start_time = time.time()
    
    try:
        response = requests.get(url, params=params, timeout=300)
        
        if response.status_code != 200:
            print(f"❌ Error: {response.status_code}")
            print(response.text)
            return False
        
        result = response.json()
        generation_time = time.time() - start_time
        
        # Save the video
        if "video" in result:
            video_data = base64.b64decode(result["video"])
            with open(output_file, "wb") as f:
                f.write(video_data)
            print(f"✅ Video saved to {output_file}")
            
            # Get the reported generation time from the API
            api_gen_time = result.get("generation_time", "unknown")
            print(f"\nPerformance:")
            print(f"- API reported generation time: {api_gen_time} seconds")
            print(f"- Total round-trip time: {generation_time:.2f} seconds")
            
            # Display the video if requested
            if display:
                try:
                    import platform
                    import subprocess
                    
                    if platform.system() == 'Windows':
                        os.startfile(output_file)
                    elif platform.system() == 'Darwin':  # macOS
                        subprocess.run(['open', output_file], check=True)
                    else:  # Linux
                        subprocess.run(['xdg-open', output_file], check=True)
                except Exception as e:
                    print(f"Could not open video automatically: {e}")
                    
            return True
        else:
            print("❌ Error: No video in response")
            print(json.dumps(result, indent=2))
            return False
            
    except Exception as e:
        print(f"❌ Error connecting to API: {e}")
        return False

def update_url(url):
    """Update the base URL for the API."""
    # Save the URL to a config file for persistence
    with open(os.path.join(os.path.dirname(__file__), '.api_url'), 'w') as f:
        f.write(url)
    print(f"API URL set to: {url}")
    return True

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the LTXV-Modal API")
    parser.add_argument("--url", type=str, help="Set the base URL for the API")
    parser.add_argument("--display", action="store_true", help="Display the generated video")
    parser.add_argument("--output", type=str, default="test_output.mp4", help="Output file path")
    args = parser.parse_args()
    
    # Try to load saved URL if it exists
    config_path = os.path.join(os.path.dirname(__file__), '.api_url')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_url = f.read().strip()
            if saved_url:
                BASE_URL = saved_url
                
    # Update URL if provided
    if args.url:
        update_url(args.url)
        BASE_URL = args.url
        
    # Check if we have a valid URL
    if BASE_URL == "https://yourname--ltxv-video":
        print("⚠️ Warning: You need to set your actual Modal deployment URL!")
        print("Run: python test-generate.py --url YOUR_DEPLOYMENT_URL")
        print("Your URL will be saved for future use.")
        exit(1)
        
    # Run the test
    test_generate_video(output_file=args.output, display=args.display)
