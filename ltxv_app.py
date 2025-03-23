import base64
import io
import os
import modal
from typing import Optional, List, Union, Dict, Any
import torch
import time
import json
import numpy as np
import imageio

# Create the Modal app
app = modal.App("ltxv-video")

# Define a volume to store model weights
model_volume = modal.Volume.from_name("ltxv-model-weights", create_if_missing=True)
MODEL_MOUNT_PATH = "/vol/models/ltxv-video"

# Define the image with required dependencies
image = (
    modal.Image.debian_slim()
    .run_commands(
        # Install core dependencies
        "pip install torch==2.5.1+cu121 torchvision torchaudio diffusers>=0.28.2 transformers>=4.47.2 sentencepiece>=0.1.96 'fastapi[standard]' accelerate huggingface-hub~=0.25.2 --extra-index-url https://download.pytorch.org/whl/cu121",
        # Install additional dependencies
        "pip install imageio[ffmpeg] einops timm matplotlib numpy",
        # Install LTX-Video directly from GitHub zip (no git required)
        "pip install https://github.com/Lightricks/LTX-Video/archive/refs/heads/main.zip"
    )
)

@app.cls(gpu="A100-40GB", image=image, volumes={MODEL_MOUNT_PATH: model_volume})
class LTXVideoModel:
    """LTX-Video model for text-to-video generation served via Modal."""
    
    def _model_exists_in_volume(self):
        """Check if the model exists in the volume."""
        model_dir = MODEL_MOUNT_PATH
        # Check for the VAE config file which is needed for initialization
        vae_config_file = os.path.join(model_dir, "vae", "config.json")
        return os.path.exists(vae_config_file)
    
    @modal.enter()
    def load_model(self):
        """Load the model during container initialization."""
        import time
        from huggingface_hub import snapshot_download

        start_time = time.time()
        
        # Model source and destination paths
        repo_id = "Lightricks/LTX-Video"
        local_model_path = MODEL_MOUNT_PATH

        print("Loading LTX-Video model...")
        
        try:
            if self._model_exists_in_volume():
                print(f"Loading model from volume at: {local_model_path}")
                self._initialize_pipeline(local_model_path)
                print(f"Model loaded from volume in {time.time() - start_time:.2f} seconds")
            else:
                print(f"Model not found in volume. Downloading from HuggingFace: {repo_id}")
                # Download the entire model repository from HuggingFace
                os.makedirs(local_model_path, exist_ok=True)
                snapshot_download(
                    repo_id=repo_id,
                    local_dir=local_model_path,
                    local_dir_use_symlinks=False,
                    repo_type='model'
                )
                
                print(f"Model repository downloaded to: {local_model_path}")
                self._initialize_pipeline(local_model_path)
                
                # Commit changes to volume
                model_volume.commit()
                print(f"Model saved to volume successfully in {time.time() - start_time:.2f} seconds")
            
            print(f"Model fully initialized and ready in {time.time() - start_time:.2f} seconds")
        except Exception as e:
            print(f"Error loading model: {e}")
            # Re-raise to ensure Modal knows there was a problem
            raise
    
    def _initialize_pipeline(self, model_path):
        """Initialize the LTX-Video pipeline with components."""
        from transformers import T5EncoderModel, T5Tokenizer
        from ltx_video.models.autoencoders.causal_video_autoencoder import CausalVideoAutoencoder
        from ltx_video.models.transformers.symmetric_patchifier import SymmetricPatchifier
        from ltx_video.models.transformers.transformer3d import Transformer3DModel
        from ltx_video.pipelines.pipeline_ltx_video import LTXVideoPipeline
        from ltx_video.schedulers.rf import RectifiedFlowScheduler
        
        device = "cuda"
        
        # Initialize pipeline components
        vae = CausalVideoAutoencoder.from_pretrained(model_path)
        transformer = Transformer3DModel.from_pretrained(model_path)
        scheduler = RectifiedFlowScheduler.from_pretrained(model_path)
        
        text_encoder_model_path = "PixArt-alpha/PixArt-XL-2-1024-MS"
        text_encoder = T5EncoderModel.from_pretrained(
            text_encoder_model_path, subfolder="text_encoder"
        )
        tokenizer = T5Tokenizer.from_pretrained(
            text_encoder_model_path, subfolder="tokenizer"
        )
        patchifier = SymmetricPatchifier(patch_size=1)
        
        # Move models to device and convert to bfloat16
        transformer = transformer.to(device).to(torch.bfloat16)
        vae = vae.to(device).to(torch.bfloat16)
        text_encoder = text_encoder.to(device).to(torch.bfloat16)
        
        # Create pipeline with None values for prompt enhancer components
        self.pipeline = LTXVideoPipeline(
            transformer=transformer,
            patchifier=patchifier,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            scheduler=scheduler,
            vae=vae,
            prompt_enhancer_image_caption_model=None,
            prompt_enhancer_image_caption_processor=None,
            prompt_enhancer_llm_model=None,
            prompt_enhancer_llm_tokenizer=None
        )
        
        # Enable CPU offloading for VRAM optimization
        self.pipeline = self.pipeline.to(device)
    
    def _load_image_to_tensor(self, 
                              image_data: bytes, 
                              height: int = 736, 
                              width: int = 1280) -> torch.Tensor:
        """Load and process an image into a tensor for conditioning."""
        from PIL import Image
        import numpy as np
        
        image = Image.open(io.BytesIO(image_data)).convert("RGB")
        
        # Resize and crop to maintain aspect ratio
        input_width, input_height = image.size
        aspect_ratio_target = width / height
        aspect_ratio_frame = input_width / input_height
        
        if aspect_ratio_frame > aspect_ratio_target:
            new_width = int(input_height * aspect_ratio_target)
            new_height = input_height
            x_start = (input_width - new_width) // 2
            y_start = 0
        else:
            new_width = input_width
            new_height = int(input_width / aspect_ratio_target)
            x_start = 0
            y_start = (input_height - new_height) // 2

        image = image.crop((x_start, y_start, x_start + new_width, y_start + new_height))
        image = image.resize((width, height))
        
        # Convert to tensor and normalize
        frame_tensor = torch.tensor(np.array(image)).permute(2, 0, 1).float()
        frame_tensor = (frame_tensor / 127.5) - 1.0
        
        # Create 5D tensor: (batch_size=1, channels=3, num_frames=1, height, width)
        return frame_tensor.unsqueeze(0).unsqueeze(2)
    
    def _load_video_to_tensor(self, 
                              video_data: bytes, 
                              height: int = 736, 
                              width: int = 1280,
                              max_frames: int = 73) -> torch.Tensor:
        """Load and process a video into a tensor for conditioning."""
        import tempfile
        import numpy as np
        from PIL import Image
        
        # Save video data to a temporary file
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as temp_file:
            temp_file.write(video_data)
            temp_path = temp_file.name
        
        try:
            # Read frames
            reader = imageio.get_reader(temp_path)
            num_frames = min(reader.count_frames(), max_frames)
            
            # Process frames
            frames = []
            for i in range(num_frames):
                frame = Image.fromarray(reader.get_data(i))
                
                # Resize and crop to maintain aspect ratio (same logic as image)
                input_width, input_height = frame.size
                aspect_ratio_target = width / height
                aspect_ratio_frame = input_width / input_height
                
                if aspect_ratio_frame > aspect_ratio_target:
                    new_width = int(input_height * aspect_ratio_target)
                    new_height = input_height
                    x_start = (input_width - new_width) // 2
                    y_start = 0
                else:
                    new_width = input_width
                    new_height = int(input_width / aspect_ratio_target)
                    x_start = 0
                    y_start = (input_height - new_height) // 2

                frame = frame.crop((x_start, y_start, x_start + new_width, y_start + new_height))
                frame = frame.resize((width, height))
                
                # Convert to tensor
                frame_tensor = torch.tensor(np.array(frame)).permute(2, 0, 1).float()
                frame_tensor = (frame_tensor / 127.5) - 1.0
                frame_tensor = frame_tensor.unsqueeze(0).unsqueeze(0)  # Add batch and frame dimensions
                frames.append(frame_tensor)
            
            reader.close()
            
            # Concatenate frames along the frame dimension
            return torch.cat(frames, dim=1)
        finally:
            # Clean up temp file
            os.unlink(temp_path)
    
    def _prepare_conditioning_items(self, 
                                   media_data_list: List[bytes],
                                   media_types: List[str],
                                   start_frames: List[int],
                                   strengths: List[float],
                                   height: int,
                                   width: int):
        """Prepare conditioning items from media data."""
        from ltx_video.pipelines.pipeline_ltx_video import ConditioningItem
        
        conditioning_items = []
        
        for i, (media_data, media_type, start_frame, strength) in enumerate(
            zip(media_data_list, media_types, start_frames, strengths)
        ):
            if media_type == "image":
                tensor = self._load_image_to_tensor(media_data, height, width)
                conditioning_items.append(ConditioningItem(tensor, start_frame, strength))
            elif media_type == "video":
                tensor = self._load_video_to_tensor(media_data, height, width)
                conditioning_items.append(ConditioningItem(tensor, start_frame, strength))
            else:
                raise ValueError(f"Unsupported media type: {media_type}")
        
        return conditioning_items
    
    def _calculate_padding(self, height, width):
        """Calculate padding to make dimensions divisible by 32."""
        height_padded = ((height - 1) // 32 + 1) * 32
        width_padded = ((width - 1) // 32 + 1) * 32
        
        # Calculate padding for each side
        pad_height = height_padded - height
        pad_width = width_padded - width
        
        pad_top = pad_height // 2
        pad_bottom = pad_height - pad_top
        pad_left = pad_width // 2
        pad_right = pad_width - pad_left
        
        # Padding format is (left, right, top, bottom)
        return (pad_left, pad_right, pad_top, pad_bottom), (height_padded, width_padded)
    
    def _adjust_num_frames(self, num_frames):
        """Adjust number of frames to be (N * 8 + 1)."""
        return ((num_frames - 1) // 8 + 1) * 8 + 1
    
    def _tensor_to_video(self, tensor, fps=25):
        """Convert a tensor to an MP4 video."""
        # Tensor is in shape [B, C, F, H, W]
        video_np = tensor[0].permute(1, 2, 3, 0).cpu().float().numpy()
        # Unnormalize to [0, 255]
        video_np = (video_np * 255).clip(0, 255).astype(np.uint8)
        
        # Create MP4 in memory
        with io.BytesIO() as output:
            with imageio.get_writer(output, format='mp4', fps=fps) as writer:
                for frame in video_np:
                    writer.append_data(frame)
            video_bytes = output.getvalue()
        
        return video_bytes
    
    def _encode_video(self, video_bytes):
        """Convert video bytes to base64 string."""
        return base64.b64encode(video_bytes).decode("utf-8")
    
    @modal.method()
    def generate_video(
        self, 
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
        return_base64: bool = True
    ):
        """Generate a video and return it as base64 or raw bytes."""
        import time
        from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
        
        # Track time for generation
        start_time = time.time()
        
        # Adjust dimensions to be divisible by 32 and frames to be (N * 8 + 1)
        padding, (height_padded, width_padded) = self._calculate_padding(height, width)
        num_frames_padded = self._adjust_num_frames(num_frames)
        
        # Set up seed for reproducibility if provided
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
            # Also set global seeds
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        
        # Prepare STG (Spatiotemporal Guidance)
        skip_block_list = [19]  # Default value, can be made configurable
        skip_layer_strategy = SkipLayerStrategy.AttentionValues
        
        # Generate the video
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=frame_rate,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pt",
            is_video=True,
            vae_per_channel_normalize=True,
            skip_layer_strategy=skip_layer_strategy,
            skip_block_list=skip_block_list,
            stg_scale=stg_scale,
            do_rescaling=stg_rescale != 1,
            rescaling_scale=stg_rescale,
            enhance_prompt=False,
        )
        
        # Crop the padded images to the desired resolution and number of frames
        (pad_left, pad_right, pad_top, pad_bottom) = padding
        pad_bottom = -pad_bottom if pad_bottom > 0 else None
        pad_right = -pad_right if pad_right > 0 else None
        images = output.images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]
        
        # Convert tensor to video
        video_bytes = self._tensor_to_video(images, frame_rate)
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Return as base64 if requested
        if return_base64:
            return {
                "video": self._encode_video(video_bytes),
                "parameters": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                    "frame_rate": frame_rate,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                    "seed": seed,
                    "stg_scale": stg_scale,
                    "stg_rescale": stg_rescale
                },
                "generation_time": round(generation_time, 2)
            }
        else:
            # Return the raw video bytes
            return video_bytes
    
    @modal.method()
    def generate_video_with_conditioning(
        self,
        prompt: str,
        media_data_list: List[bytes],
        media_types: List[str],
        start_frames: List[int],
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
        return_base64: bool = True
    ):
        """Generate a video with conditioning and return it as base64 or raw bytes."""
        import time
        from ltx_video.utils.skip_layer_strategy import SkipLayerStrategy
        
        # Track time for generation
        start_time = time.time()
        
        # Validate inputs
        if len(media_data_list) != len(media_types) or len(media_data_list) != len(start_frames):
            raise ValueError("media_data_list, media_types, and start_frames must have the same length")
        
        # Use default strengths if not provided
        if strengths is None:
            strengths = [1.0] * len(media_data_list)
        elif len(strengths) != len(media_data_list):
            raise ValueError("strengths must have the same length as media_data_list")
        
        # Adjust dimensions to be divisible by 32 and frames to be (N * 8 + 1)
        padding, (height_padded, width_padded) = self._calculate_padding(height, width)
        num_frames_padded = self._adjust_num_frames(num_frames)
        
        # Prepare conditioning items
        conditioning_items = self._prepare_conditioning_items(
            media_data_list, media_types, start_frames, strengths, height, width
        )
        
        # Set up seed for reproducibility if provided
        generator = None
        if seed is not None:
            generator = torch.Generator("cuda").manual_seed(seed)
            # Also set global seeds
            torch.manual_seed(seed)
            torch.cuda.manual_seed(seed)
            np.random.seed(seed)
        
        # Prepare STG (Spatiotemporal Guidance)
        skip_block_list = [19]  # Default value, can be made configurable
        skip_layer_strategy = SkipLayerStrategy.AttentionValues
        
        # Generate the video
        output = self.pipeline(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height_padded,
            width=width_padded,
            num_frames=num_frames_padded,
            frame_rate=frame_rate,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            generator=generator,
            output_type="pt",
            is_video=True,
            vae_per_channel_normalize=True,
            skip_layer_strategy=skip_layer_strategy,
            skip_block_list=skip_block_list,
            stg_scale=stg_scale,
            do_rescaling=stg_rescale != 1,
            rescaling_scale=stg_rescale,
            conditioning_items=conditioning_items,
            enhance_prompt=False,
        )
        
        # Crop the padded images to the desired resolution and number of frames
        (pad_left, pad_right, pad_top, pad_bottom) = padding
        pad_bottom = -pad_bottom if pad_bottom > 0 else None
        pad_right = -pad_right if pad_right > 0 else None
        images = output.images[:, :, :num_frames, pad_top:pad_bottom, pad_left:pad_right]
        
        # Convert tensor to video
        video_bytes = self._tensor_to_video(images, frame_rate)
        
        # Calculate generation time
        generation_time = time.time() - start_time
        
        # Return as base64 if requested
        if return_base64:
            # Create a more user-friendly representation of conditioning
            conditioning_info = []
            for i, (type_, start, strength) in enumerate(zip(media_types, start_frames, strengths)):
                conditioning_info.append({
                    "index": i,
                    "type": type_,
                    "start_frame": start,
                    "strength": strength
                })
            
            return {
                "video": self._encode_video(video_bytes),
                "parameters": {
                    "prompt": prompt,
                    "negative_prompt": negative_prompt,
                    "height": height,
                    "width": width,
                    "num_frames": num_frames,
                    "frame_rate": frame_rate,
                    "guidance_scale": guidance_scale,
                    "num_inference_steps": num_inference_steps,
                    "seed": seed,
                    "stg_scale": stg_scale,
                    "stg_rescale": stg_rescale,
                    "conditioning": conditioning_info
                },
                "generation_time": round(generation_time, 2)
            }
        else:
            # Return the raw video bytes
            return video_bytes
    
    @modal.method()
    def batch_generate(
        self,
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
        stg_rescale: float = 0.7
    ):
        """Generate multiple videos from a list of prompts."""
        import time
        
        results = []
        total_start_time = time.time()
        
        for i, prompt in enumerate(prompts):
            # Set seed for this prompt (increment base_seed if provided)
            seed = None
            if base_seed is not None:
                seed = base_seed + i
            
            # Generate video for this prompt
            start_time = time.time()
            video_bytes = self.generate_video.remote(
                prompt=prompt,
                negative_prompt=negative_prompt,
                height=height,
                width=width,
                num_frames=num_frames,
                frame_rate=frame_rate,
                guidance_scale=guidance_scale,
                num_inference_steps=num_inference_steps,
                seed=seed,
                stg_scale=stg_scale,
                stg_rescale=stg_rescale,
                return_base64=False
            )
            generation_time = time.time() - start_time
            
            # Add to results
            results.append({
                "prompt": prompt,
                "video": self._encode_video(video_bytes),
                "seed": seed,
                "generation_time": round(generation_time, 2)
            })
        
        # Calculate total generation time
        total_time = time.time() - total_start_time
        
        return {
            "results": results,
            "parameters": {
                "negative_prompt": negative_prompt,
                "height": height,
                "width": width,
                "num_frames": num_frames,
                "frame_rate": frame_rate,
                "guidance_scale": guidance_scale,
                "num_inference_steps": num_inference_steps,
                "base_seed": base_seed,
                "stg_scale": stg_scale,
                "stg_rescale": stg_rescale
            },
            "total_generation_time": round(total_time, 2),
            "videos_generated": len(prompts)
        }
    
    @modal.method()
    def get_model_info(self):
        """Return information about the model."""
        # Check if model is loaded from volume
        model_source = "volume" if self._model_exists_in_volume() else "huggingface"
        
        return {
            "model": "Lightricks/LTX-Video",
            "version": "v0.9.5",
            "parameters": "2B",
            "format": "safetensors",
            "source": model_source,
            "capabilities": [
                "text-to-video",
                "image-to-video"
            ],
            "recommended_settings": {
                "height": 736,
                "width": 1280,
                "num_frames": 73,
                "frame_rate": 25,
                "guidance_scale": 3.0,
                "num_inference_steps": 40
            },
            "volume_path": MODEL_MOUNT_PATH
        }
    
    @modal.method()
    def force_model_reload(self):
        """Force reload the model from HuggingFace and save to volume."""
        from huggingface_hub import snapshot_download
        import time
        
        start_time = time.time()
        repo_id = "Lightricks/LTX-Video"
        local_model_path = MODEL_MOUNT_PATH
        
        print(f"Force downloading model from HuggingFace: {repo_id}")
        try:
            # Download the entire model repository from HuggingFace with force_download
            os.makedirs(local_model_path, exist_ok=True)
            snapshot_download(
                repo_id=repo_id,
                local_dir=local_model_path,
                local_dir_use_symlinks=False,
                repo_type='model',
                force_download=True
            )
            
            # Re-initialize the pipeline
            self._initialize_pipeline(local_model_path)
            
            # Commit changes to volume
            model_volume.commit()
            
            total_time = time.time() - start_time
            print(f"Model repository reloaded and saved to volume in {total_time:.2f} seconds")
            
            return {
                "success": True,
                "message": f"Model successfully reloaded in {total_time:.2f} seconds",
                "model_path": local_model_path
            }
        except Exception as e:
            error_message = f"Error reloading model: {str(e)}"
            print(error_message)
            return {
                "success": False,
                "message": error_message
            }
    
    # FastAPI endpoints
    @modal.fastapi_endpoint()
    def generate_video_api(
        self,
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
        stg_rescale: float = 0.7
    ):
        """API endpoint for text-to-video generation."""
        result = self.generate_video.remote(
            prompt=prompt,
            negative_prompt=negative_prompt,
            height=height,
            width=width,
            num_frames=num_frames,
            frame_rate=frame_rate,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            seed=seed,
            stg_scale=stg_scale,
            stg_rescale=stg_rescale,
            return_base64=True
        )
        return result
    
    @modal.fastapi_endpoint(method="POST")
    def condition_video_api(self, data: dict):
        """API endpoint for conditioned video generation."""
        # Extract parameters from request body
        prompt = data.get("prompt")
        if not prompt:
            return {"error": "No prompt provided"}
        
        # Get conditioning media
        media_data_list = data.get("media_data_list", [])
        media_types = data.get("media_types", [])
        start_frames = data.get("start_frames", [])
        strengths = data.get("strengths")
        
        if not media_data_list or not media_types or not start_frames:
            return {"error": "Missing conditioning data"}
        
        # Decode base64 media data
        try:
            decoded_media = [base64.b64decode(m) for m in media_data_list]
        except Exception as e:
            return {"error": f"Invalid media data: {str(e)}"}
        
        # Get other parameters with defaults
        params = {
            "negative_prompt": data.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"),
            "height": data.get("height", 736),
            "width": data.get("width", 1280),
            "num_frames": data.get("num_frames", 73),
            "frame_rate": data.get("frame_rate", 25),
            "guidance_scale": data.get("guidance_scale", 3.0),
            "num_inference_steps": data.get("num_inference_steps", 40),
            "seed": data.get("seed"),
            "stg_scale": data.get("stg_scale", 1.0),
            "stg_rescale": data.get("stg_rescale", 0.7)
        }
        
        # Call the conditioned generation method
        return self.generate_video_with_conditioning.remote(
            prompt=prompt,
            media_data_list=decoded_media,
            media_types=media_types,
            start_frames=start_frames,
            strengths=strengths,
            **params
        )
    
    @modal.fastapi_endpoint(method="POST")
    def batch_api(self, data: dict):
        """API endpoint for batch generating multiple videos."""
        # Extract parameters from request body
        prompts = data.get("prompts", [])
        if not prompts:
            return {"error": "No prompts provided"}
        
        # Get other parameters with defaults
        params = {
            "negative_prompt": data.get("negative_prompt", "worst quality, inconsistent motion, blurry, jittery, distorted"),
            "height": data.get("height", 736),
            "width": data.get("width", 1280),
            "num_frames": data.get("num_frames", 73),
            "frame_rate": data.get("frame_rate", 25),
            "guidance_scale": data.get("guidance_scale", 3.0),
            "num_inference_steps": data.get("num_inference_steps", 40),
            "base_seed": data.get("base_seed"),
            "stg_scale": data.get("stg_scale", 1.0),
            "stg_rescale": data.get("stg_rescale", 0.7)
        }
        
        # Call the batch generation method
        return self.batch_generate.remote(
            prompts=prompts,
            **params
        )
    
    @modal.fastapi_endpoint()
    def info(self):
        """API endpoint for getting model information."""
        return self.get_model_info.remote()
    
    @modal.fastapi_endpoint(method="POST")
    def reload_model(self):
        """API endpoint to force reload the model from HuggingFace."""
        return self.force_model_reload.remote()


# Example of how to call the model functions directly from Python
@app.function(image=image)
def main():
    model = LTXVideoModel()
    
    # First, get model information to check the source (volume or huggingface)
    info = model.get_model_info.remote()
    print(f"Model information:")
    print(f"  - Name: {info['model']}")
    print(f"  - Parameters: {info['parameters']}")
    print(f"  - Source: {info['source']}")
    print(f"  - Volume path: {info['volume_path']}")
    print()
    
    # Example: Generate a simple video
    prompt = "A scenic mountain landscape with clouds rolling over the peaks at sunset"
    print(f"Generating video for prompt: '{prompt}'")
    
    result = model.generate_video.remote(
        prompt=prompt,
        height=480,  # Lower resolution for testing
        width=640,
        num_frames=25,  # Fewer frames for testing
        num_inference_steps=20  # Fewer steps for testing
    )
    
    print(f"Generated video with parameters: {result['parameters']}")
    print(f"Generation time: {result['generation_time']} seconds")
    
    print("\nNote: The first run will download the model from HuggingFace")
    print("and save it to the volume. Subsequent runs will be faster")
    print("as the model will be loaded directly from the volume.")
    print("\nTo force a model update, use the reload_model endpoint:")
    print("  POST /reload_model")


# Utility function to create or initialize the volume
@app.function(image=modal.Image.debian_slim())
def create_volume():
    """Create or ensure the model volume exists.
    
    This function can be run separately to set up the volume
    before running the main application.
    """
    # Ensure volume exists
    vol = modal.Volume.from_name("ltxv-model-weights", create_if_missing=True)
    
    # Check if it exists
    vol_info = vol.get()
    
    # Get volume size
    size_mb = vol_info.get("size_mb", 0)
    
    print(f"Volume 'ltxv-model-weights' is ready.")
    print(f"Current size: {size_mb:.2f} MB")
    print(f"Mount path in containers: {MODEL_MOUNT_PATH}")
    
    if size_mb < 1:
        print("Volume is empty. When you run the app, the model will be")
        print("downloaded from HuggingFace and saved to the volume.")
    else:
        print("Volume contains data. The app will attempt to load the model from the volume.")
    
    return {
        "name": "ltxv-model-weights",
        "size_mb": size_mb,
        "mount_path": MODEL_MOUNT_PATH
    }


if __name__ == "__main__":
    # For local development and testing
    # When running modal serve ltxv_app.py, the web endpoints will be available
    # For production, use modal deploy ltxv_app.py
    
    # Uncomment to create/check the volume first:
    # create_volume.local()
    
    main.local()
