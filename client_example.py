#!/usr/bin/env python
"""
Example client script for the LTXV-Modal API.
This script demonstrates how to interact with the deployed Modal API for video generation.

Make sure to deploy the API first:
    modal deploy ltxv_app.py
"""

import base64
import io
import json
import os
import requests
import argparse
import tempfile
from typing import List, Optional, Dict, Any, Union
from PIL import Image
import subprocess
import sys
import platform

# Replace with your actual deployment URL after deploying
# The URL will look like: https://yourname--ltxv-video-ltxvideomodel-generate-video-api.modal.run
BASE_URL = "https://yourname--ltxv-video"

def open_file(file_path):
    """Open a file with the default application."""
    if platform.system() == 'Windows':
        os.startfile(file_path)
    elif platform.system() == 'Darwin':  # macOS
        subprocess.run(['open', file_path], check=True)
    else:  # Linux
        subprocess.run(['xdg-open', file_path], check=True)

def generate_video(
    prompt: str,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    height: int = 736,
    width: int = 1280,
    num_frames: int = 73,
    frame_rate: int = 25,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 40,
    seed: Optional[int] = None,
    stg_scale: float = 1.0,
    stg_rescale: float = 0.7,
    save_path: str = "output.mp4",
    open_video: bool = False
) -> Dict[str, Any]:
    """
    Generate a video using the Modal API.
    
    Args:
        prompt: Text prompt for video generation
        negative_prompt: Negative prompt for unwanted features
        height: Video height in pixels
        width: Video width in pixels
        num_frames: Number of frames to generate
        frame_rate: Frames per second
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of inference steps
        seed: Random seed for reproducibility
        stg_scale: Spatiotemporal guidance scale
        stg_rescale: Spatiotemporal guidance rescale value
        save_path: Path to save the generated video
        open_video: Whether to open the video after generation
        
    Returns:
        Dict containing API response data
    """
    # Build URL with query parameters
    url = f"{BASE_URL}-ltxvideomodel-generate-video-api.modal.run"
    params = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "stg_scale": stg_scale,
        "stg_rescale": stg_rescale
    }
    
    if seed is not None:
        params["seed"] = seed
    
    print(f"Generating video for prompt: '{prompt}'")
    print(f"Using parameters: {json.dumps({k: v for k, v in params.items() if k != 'prompt'}, indent=2)}")
    
    try:
        response = requests.get(url, params=params, timeout=600)  # Longer timeout for video generation
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return {"error": response.text}
        
        result = response.json()
        
        # Save the video
        if "video" in result:
            video_data = base64.b64decode(result["video"])
            with open(save_path, "wb") as f:
                f.write(video_data)
            print(f"Video saved to {save_path}")
            
            # Open the video if requested
            if open_video:
                open_file(save_path)
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return {"error": str(e)}

def generate_conditioned_video(
    prompt: str,
    media_path: str,
    start_frame: int = 0,
    media_type: Optional[str] = None,
    strength: float = 1.0,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    height: int = 736,
    width: int = 1280,
    num_frames: int = 73,
    frame_rate: int = 25,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 40,
    seed: Optional[int] = None,
    stg_scale: float = 1.0,
    stg_rescale: float = 0.7,
    save_path: str = "output.mp4",
    open_video: bool = False
) -> Dict[str, Any]:
    """
    Generate a video conditioned on an image or another video using the Modal API.
    
    Args:
        prompt: Text prompt for video generation
        media_path: Path to the conditioning image or video
        start_frame: Frame index where the conditioning should be applied
        media_type: Either "image" or "video", if None will be inferred from file extension
        strength: Conditioning strength (0.0 to 1.0)
        negative_prompt: Negative prompt for unwanted features
        height: Video height in pixels
        width: Video width in pixels
        num_frames: Number of frames to generate
        frame_rate: Frames per second
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of inference steps
        seed: Random seed for reproducibility
        stg_scale: Spatiotemporal guidance scale
        stg_rescale: Spatiotemporal guidance rescale value
        save_path: Path to save the generated video
        open_video: Whether to open the video after generation
        
    Returns:
        Dict containing API response data
    """
    # Detect media type if not provided
    if media_type is None:
        ext = os.path.splitext(media_path)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
            media_type = "image"
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.gif']:
            media_type = "video"
        else:
            raise ValueError(f"Cannot determine media type from extension: {ext}")
    
    # Read media file
    with open(media_path, "rb") as f:
        media_data = f.read()
    
    # Encode media data
    encoded_media = base64.b64encode(media_data).decode("utf-8")
    
    # Build URL
    url = f"{BASE_URL}-ltxvideomodel-condition-video-api.modal.run"
    
    # Prepare request data
    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "media_data_list": [encoded_media],
        "media_types": [media_type],
        "start_frames": [start_frame],
        "strengths": [strength],
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "stg_scale": stg_scale,
        "stg_rescale": stg_rescale
    }
    
    if seed is not None:
        data["seed"] = seed
    
    print(f"Generating video conditioned on {media_type} at frame {start_frame}")
    print(f"Prompt: '{prompt}'")
    print(f"Using parameters: {json.dumps({k: v for k, v in data.items() if k not in ['prompt', 'media_data_list']}, indent=2)}")
    
    try:
        response = requests.post(url, json=data, timeout=600)  # Longer timeout for video generation
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return {"error": response.text}
        
        result = response.json()
        
        # Save the video
        if "video" in result:
            video_data = base64.b64decode(result["video"])
            with open(save_path, "wb") as f:
                f.write(video_data)
            print(f"Video saved to {save_path}")
            
            # Open the video if requested
            if open_video:
                open_file(save_path)
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return {"error": str(e)}

def generate_multi_conditioned_video(
    prompt: str,
    media_paths: List[str],
    start_frames: List[int],
    media_types: Optional[List[str]] = None,
    strengths: Optional[List[float]] = None,
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    height: int = 736,
    width: int = 1280,
    num_frames: int = 73,
    frame_rate: int = 25,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 40,
    seed: Optional[int] = None,
    stg_scale: float = 1.0,
    stg_rescale: float = 0.7,
    save_path: str = "output.mp4",
    open_video: bool = False
) -> Dict[str, Any]:
    """
    Generate a video conditioned on multiple images or videos using the Modal API.
    
    Args:
        prompt: Text prompt for video generation
        media_paths: List of paths to conditioning images or videos
        start_frames: List of frame indices where each conditioning should be applied
        media_types: List of "image" or "video" for each media, if None will be inferred
        strengths: List of conditioning strengths (0.0 to 1.0) for each media
        negative_prompt: Negative prompt for unwanted features
        height: Video height in pixels
        width: Video width in pixels
        num_frames: Number of frames to generate
        frame_rate: Frames per second
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of inference steps
        seed: Random seed for reproducibility
        stg_scale: Spatiotemporal guidance scale
        stg_rescale: Spatiotemporal guidance rescale value
        save_path: Path to save the generated video
        open_video: Whether to open the video after generation
        
    Returns:
        Dict containing API response data
    """
    # Validate inputs
    if len(media_paths) != len(start_frames):
        raise ValueError("media_paths and start_frames must have the same length")
    
    # Auto-detect media types if not provided
    if media_types is None:
        media_types = []
        for path in media_paths:
            ext = os.path.splitext(path)[1].lower()
            if ext in ['.jpg', '.jpeg', '.png', '.webp', '.bmp']:
                media_types.append("image")
            elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.gif']:
                media_types.append("video")
            else:
                raise ValueError(f"Cannot determine media type from extension: {ext}")
    
    # Use default strengths if not provided
    if strengths is None:
        strengths = [1.0] * len(media_paths)
    
    # Read and encode all media
    encoded_media_list = []
    for path in media_paths:
        with open(path, "rb") as f:
            media_data = f.read()
        encoded_media_list.append(base64.b64encode(media_data).decode("utf-8"))
    
    # Build URL
    url = f"{BASE_URL}-ltxvideomodel-condition-video-api.modal.run"
    
    # Prepare request data
    data = {
        "prompt": prompt,
        "negative_prompt": negative_prompt,
        "media_data_list": encoded_media_list,
        "media_types": media_types,
        "start_frames": start_frames,
        "strengths": strengths,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "stg_scale": stg_scale,
        "stg_rescale": stg_rescale
    }
    
    if seed is not None:
        data["seed"] = seed
    
    print(f"Generating video with {len(media_paths)} conditioning items")
    print(f"Prompt: '{prompt}'")
    print(f"Using parameters: {json.dumps({k: v for k, v in data.items() if k not in ['prompt', 'media_data_list']}, indent=2)}")
    
    try:
        response = requests.post(url, json=data, timeout=600)  # Longer timeout for video generation
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return {"error": response.text}
        
        result = response.json()
        
        # Save the video
        if "video" in result:
            video_data = base64.b64decode(result["video"])
            with open(save_path, "wb") as f:
                f.write(video_data)
            print(f"Video saved to {save_path}")
            
            # Open the video if requested
            if open_video:
                open_file(save_path)
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return {"error": str(e)}

def batch_generate(
    prompts: List[str],
    negative_prompt: str = "worst quality, inconsistent motion, blurry, jittery, distorted",
    height: int = 736,
    width: int = 1280,
    num_frames: int = 73,
    frame_rate: int = 25,
    guidance_scale: float = 3.0,
    num_inference_steps: int = 40,
    base_seed: Optional[int] = None,
    stg_scale: float = 1.0,
    stg_rescale: float = 0.7,
    output_dir: str = "outputs",
    open_videos: bool = False
) -> Dict[str, Any]:
    """
    Generate multiple videos in a batch using the Modal API.
    
    Args:
        prompts: List of text prompts
        negative_prompt: Negative prompt for unwanted features
        height: Video height in pixels
        width: Video width in pixels
        num_frames: Number of frames to generate
        frame_rate: Frames per second
        guidance_scale: Classifier-free guidance scale
        num_inference_steps: Number of inference steps
        base_seed: Base random seed (will be incremented for each video)
        stg_scale: Spatiotemporal guidance scale
        stg_rescale: Spatiotemporal guidance rescale value
        output_dir: Directory to save generated videos
        open_videos: Whether to open the videos after generation
        
    Returns:
        Dict containing API response data
    """
    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Prepare request data
    url = f"{BASE_URL}-ltxvideomodel-batch-api.modal.run"
    data = {
        "prompts": prompts,
        "negative_prompt": negative_prompt,
        "height": height,
        "width": width,
        "num_frames": num_frames,
        "frame_rate": frame_rate,
        "guidance_scale": guidance_scale,
        "num_inference_steps": num_inference_steps,
        "stg_scale": stg_scale,
        "stg_rescale": stg_rescale
    }
    
    if base_seed is not None:
        data["base_seed"] = base_seed
    
    print(f"Batch generating {len(prompts)} videos...")
    
    try:
        response = requests.post(url, json=data, timeout=600 * len(prompts))  # Scale timeout with number of videos
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return {"error": response.text}
        
        result = response.json()
        
        # Save the videos
        if "results" in result:
            for i, video_result in enumerate(result["results"]):
                if "video" in video_result:
                    # Create a filename from the prompt
                    prompt = video_result["prompt"]
                    filename = f"{i+1:03d}_{prompt[:30].replace(' ', '_')}.mp4"
                    filepath = os.path.join(output_dir, filename)
                    
                    # Save the video
                    video_data = base64.b64decode(video_result["video"])
                    with open(filepath, "wb") as f:
                        f.write(video_data)
                    print(f"Video {i+1} saved to {filepath}")
                    
                    # Open the video if requested
                    if open_videos:
                        open_file(filepath)
        
        return result
    except requests.exceptions.RequestException as e:
        print(f"Request error: {e}")
        return {"error": str(e)}

def get_model_info() -> Dict[str, Any]:
    """
    Get information about the model.
    
    Returns:
        Dict containing model information
    """
    url = f"{BASE_URL}-ltxvideomodel-info.modal.run"
    
    print(f"Getting model information from: {url}")
    try:
        response = requests.get(url, timeout=30)
        
        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            return {"error": response.text}
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Connection error: {e}")
        return {"error": f"Connection error: {e}"}

def main():
    parser = argparse.ArgumentParser(description="Client for LTXV-Modal API")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Generate video command
    gen_parser = subparsers.add_parser("generate", help="Generate a video from text prompt")
    gen_parser.add_argument("prompt", type=str, help="Text prompt for video generation")
    gen_parser.add_argument("--negative-prompt", type=str, default="worst quality, inconsistent motion, blurry, jittery, distorted", help="Negative prompt")
    gen_parser.add_argument("--height", type=int, default=736, help="Video height")
    gen_parser.add_argument("--width", type=int, default=1280, help="Video width")
    gen_parser.add_argument("--frames", type=int, default=73, help="Number of frames")
    gen_parser.add_argument("--fps", type=int, default=25, help="Frames per second")
    gen_parser.add_argument("--guidance-scale", type=float, default=3.0, help="Guidance scale")
    gen_parser.add_argument("--steps", type=int, default=40, help="Inference steps")
    gen_parser.add_argument("--seed", type=int, help="Random seed")
    gen_parser.add_argument("--stg-scale", type=float, default=1.0, help="Spatiotemporal guidance scale")
    gen_parser.add_argument("--stg-rescale", type=float, default=0.7, help="Spatiotemporal guidance rescale")
    gen_parser.add_argument("--output", type=str, default="output.mp4", help="Output file path")
    gen_parser.add_argument("--open", action="store_true", help="Open the video after generation")
    
    # Image-to-video command
    img2vid_parser = subparsers.add_parser("image-to-video", help="Generate a video from an image")
    img2vid_parser.add_argument("prompt", type=str, help="Text prompt for video generation")
    img2vid_parser.add_argument("image", type=str, help="Path to input image")
    img2vid_parser.add_argument("--start-frame", type=int, default=0, help="Frame where the image should be placed")
    img2vid_parser.add_argument("--strength", type=float, default=1.0, help="Conditioning strength")
    img2vid_parser.add_argument("--negative-prompt", type=str, default="worst quality, inconsistent motion, blurry, jittery, distorted", help="Negative prompt")
    img2vid_parser.add_argument("--height", type=int, default=736, help="Video height")
    img2vid_parser.add_argument("--width", type=int, default=1280, help="Video width")
    img2vid_parser.add_argument("--frames", type=int, default=73, help="Number of frames")
    img2vid_parser.add_argument("--fps", type=int, default=25, help="Frames per second")
    img2vid_parser.add_argument("--guidance-scale", type=float, default=3.0, help="Guidance scale")
    img2vid_parser.add_argument("--steps", type=int, default=40, help="Inference steps")
    img2vid_parser.add_argument("--seed", type=int, help="Random seed")
    img2vid_parser.add_argument("--stg-scale", type=float, default=1.0, help="Spatiotemporal guidance scale")
    img2vid_parser.add_argument("--stg-rescale", type=float, default=0.7, help="Spatiotemporal guidance rescale")
    img2vid_parser.add_argument("--output", type=str, default="output.mp4", help="Output file path")
    img2vid_parser.add_argument("--open", action="store_true", help="Open the video after generation")
    
    # Batch generate command
    batch_parser = subparsers.add_parser("batch", help="Generate multiple videos")
    batch_parser.add_argument("--prompts-file", type=str, help="File containing prompts (one per line)")
    batch_parser.add_argument("--prompts", nargs="+", help="List of prompts")
    batch_parser.add_argument("--negative-prompt", type=str, default="worst quality, inconsistent motion, blurry, jittery, distorted", help="Negative prompt")
    batch_parser.add_argument("--height", type=int, default=736, help="Video height")
    batch_parser.add_argument("--width", type=int, default=1280, help="Video width")
    batch_parser.add_argument("--frames", type=int, default=73, help="Number of frames")
    batch_parser.add_argument("--fps", type=int, default=25, help="Frames per second")
    batch_parser.add_argument("--guidance-scale", type=float, default=3.0, help="Guidance scale")
    batch_parser.add_argument("--steps", type=int, default=40, help="Inference steps")
    batch_parser.add_argument("--base-seed", type=int, help="Base random seed")
    batch_parser.add_argument("--stg-scale", type=float, default=1.0, help="Spatiotemporal guidance scale")
    batch_parser.add_argument("--stg-rescale", type=float, default=0.7, help="Spatiotemporal guidance rescale")
    batch_parser.add_argument("--output-dir", type=str, default="outputs", help="Output directory")
    batch_parser.add_argument("--open", action="store_true", help="Open the videos after generation")
    
    # Multi-conditioned video generation
    multi_parser = subparsers.add_parser("multi-condition", help="Generate video with multiple conditionings")
    multi_parser.add_argument("prompt", type=str, help="Text prompt for video generation")
    multi_parser.add_argument("--media", nargs="+", required=True, help="Paths to input media files")
    multi_parser.add_argument("--frames", nargs="+", type=int, required=True, help="Frame indices for each media")
    multi_parser.add_argument("--types", nargs="+", help="Types for each media (image or video)")
    multi_parser.add_argument("--strengths", nargs="+", type=float, help="Strengths for each media")
    multi_parser.add_argument("--negative-prompt", type=str, default="worst quality, inconsistent motion, blurry, jittery, distorted", help="Negative prompt")
    multi_parser.add_argument("--height", type=int, default=736, help="Video height")
    multi_parser.add_argument("--width", type=int, default=1280, help="Video width")
    multi_parser.add_argument("--total-frames", type=int, default=73, help="Number of frames")
    multi_parser.add_argument("--fps", type=int, default=25, help="Frames per second")
    multi_parser.add_argument("--guidance-scale", type=float, default=3.0, help="Guidance scale")
    multi_parser.add_argument("--steps", type=int, default=40, help="Inference steps")
    multi_parser.add_argument("--seed", type=int, help="Random seed")
    multi_parser.add_argument("--stg-scale", type=float, default=1.0, help="Spatiotemporal guidance scale")
    multi_parser.add_argument("--stg-rescale", type=float, default=0.7, help="Spatiotemporal guidance rescale")
    multi_parser.add_argument("--output", type=str, default="output.mp4", help="Output file path")
    multi_parser.add_argument("--open", action="store_true", help="Open the video after generation")
    
    # Info command
    subparsers.add_parser("info", help="Get model information")
    
    # Update URL command
    url_parser = subparsers.add_parser("set-url", help="Set the API base URL")
    url_parser.add_argument("url", type=str, help="Base URL for the API")
    
    args = parser.parse_args()
    
    # Check if BASE_URL is set to the default value
    global BASE_URL
    if BASE_URL == "https://yourname--ltxv-video" and args.command != "set-url":
        print("Warning: You need to set your actual Modal deployment URL first!")
        print("Run: python client_example.py set-url YOUR_DEPLOYMENT_URL")
        print("Or edit the BASE_URL variable in this script.")
        return
    
    if args.command == "generate":
        result = generate_video(
            prompt=args.prompt,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            frame_rate=args.fps,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            seed=args.seed,
            stg_scale=args.stg_scale,
            stg_rescale=args.stg_rescale,
            save_path=args.output,
            open_video=args.open
        )
        print(f"Generation time: {result.get('generation_time', 'unknown')} seconds")
        
    elif args.command == "image-to-video":
        result = generate_conditioned_video(
            prompt=args.prompt,
            media_path=args.image,
            start_frame=args.start_frame,
            media_type="image",
            strength=args.strength,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            frame_rate=args.fps,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            seed=args.seed,
            stg_scale=args.stg_scale,
            stg_rescale=args.stg_rescale,
            save_path=args.output,
            open_video=args.open
        )
        print(f"Generation time: {result.get('generation_time', 'unknown')} seconds")
        
    elif args.command == "batch":
        prompts = []
        if args.prompts_file:
            with open(args.prompts_file, 'r') as f:
                prompts = [line.strip() for line in f if line.strip() and not line.startswith('#')]
        elif args.prompts:
            prompts = args.prompts
        else:
            print("Error: Either --prompts-file or --prompts must be provided")
            return
            
        result = batch_generate(
            prompts=prompts,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.frames,
            frame_rate=args.fps,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            base_seed=args.base_seed,
            stg_scale=args.stg_scale,
            stg_rescale=args.stg_rescale,
            output_dir=args.output_dir,
            open_videos=args.open
        )
        print(f"Total generation time: {result.get('total_generation_time', 'unknown')} seconds")
        
    elif args.command == "multi-condition":
        # Validate inputs
        if len(args.media) != len(args.frames):
            print("Error: Number of media files must match number of frame indices")
            return
        
        # Parse types
        media_types = args.types
        if media_types and len(media_types) != len(args.media):
            print("Error: If providing media types, must provide one for each media file")
            return
        
        # Parse strengths
        strengths = args.strengths
        if strengths and len(strengths) != len(args.media):
            print("Error: If providing strengths, must provide one for each media file")
            return
        
        result = generate_multi_conditioned_video(
            prompt=args.prompt,
            media_paths=args.media,
            start_frames=args.frames,
            media_types=media_types,
            strengths=strengths,
            negative_prompt=args.negative_prompt,
            height=args.height,
            width=args.width,
            num_frames=args.total_frames,
            frame_rate=args.fps,
            guidance_scale=args.guidance_scale,
            num_inference_steps=args.steps,
            seed=args.seed,
            stg_scale=args.stg_scale,
            stg_rescale=args.stg_rescale,
            save_path=args.output,
            open_video=args.open
        )
        print(f"Generation time: {result.get('generation_time', 'unknown')} seconds")
        
    elif args.command == "info":
        info = get_model_info()
        print("Model Information:")
        print(json.dumps(info, indent=2))
        
    elif args.command == "set-url":
        # Save the URL to a config file for persistence
        with open(os.path.join(os.path.dirname(__file__), '.api_url'), 'w') as f:
            f.write(args.url)
        print(f"API URL set to: {args.url}")
        BASE_URL = args.url
        
    else:
        parser.print_help()


if __name__ == "__main__":
    # Try to load saved URL if it exists
    config_path = os.path.join(os.path.dirname(__file__), '.api_url')
    if os.path.exists(config_path):
        with open(config_path, 'r') as f:
            saved_url = f.read().strip()
            if saved_url:
                BASE_URL = saved_url
    
    main()
