#!/usr/bin/env python
"""
Test script for the LTXV-Modal API's image-to-video capability.
"""

import os
import requests
import base64
import json
import argparse
import time

# The URL needs to be updated with your actual deployment URL
BASE_URL = "https://yourname--ltxv-video"

def test_img2vid(image_path, output_file="img2vid_output.mp4", display=False):
    """Run a test image-to-video generation with a specific image."""
    if not os.path.exists(image_path):
        print(f"Error: Image file {image_path} not found.")
        return False
    
    # Test parameters
    prompt = "The image comes to life with subtle movements, cinematic quality"
    
    # Read and encode the image
    with open(image_path, "rb") as f:
        image_data = f.read()
    encoded_image = base64.b64encode(image_data).decode("utf-8")
    
    # Build URL and request data
    url = f"{BASE_URL}-ltxvideomodel-condition-video-api.modal.run"
    data = {
        "prompt": prompt,
        "media_data_list": [encoded_image],
        "media_types": ["image"],
        "start_frames": [0],
        "strengths": [1.0],
        "height": 480,  # Lower resolution for faster test
        "width": 640,
        "num_frames": 25,  # Fewer frames for faster test
        "frame_rate": 25,
        "guidance_scale": 3.0,
        "num_inference_steps": 20,  # Fewer steps for faster test
        "stg_scale": 1.0,
        "stg_rescale": 0.7
    }
    
    print(f"Generating video from image: {image_path}")
    print(f"Using prompt: '{prompt}'")
    start_time = time.time()
    
    try:
        response = requests.post(url, json=data, timeout=300)
        
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

if __name__ == "__main__":
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Test the LTXV-Modal image-to-video API")
    parser.add_argument("--url", type=str, help="Set the base URL for the API")
    parser.add_argument("--image", type=str, required=True, help="Path to the input image")
    parser.add_argument("--display", action="store_true", help="Display the generated video")
    parser.add_argument("--output", type=str, default="img2vid_output.mp4", help="Output file path")
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
        with open(os.path.join(os.path.dirname(__file__), '.api_url'), 'w') as f:
            f.write(args.url)
        print(f"API URL set to: {args.url}")
        BASE_URL = args.url
        
    # Check if we have a valid URL
    if BASE_URL == "https://yourname--ltxv-video":
        print("⚠️ Warning: You need to set your actual Modal deployment URL!")
        print("Run: python test-img2vid.py --url YOUR_DEPLOYMENT_URL --image path/to/image.jpg")
        print("Your URL will be saved for future use.")
        exit(1)
        
    # Run the test
    test_img2vid(args.image, output_file=args.output, display=args.display)
