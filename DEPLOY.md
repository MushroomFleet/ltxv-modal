# Deploying the LTXV-Modal API

This guide provides detailed instructions for deploying the LTXV-Video generation API to Modal, addressing common issues, and verifying the installation.

## Prerequisites

1. Install Modal and authenticate
   ```bash
   pip install modal
   modal token new
   ```

2. Ensure you have a Modal account with GPU access

## Deployment Steps

1. Deploy the application
   ```bash
   cd ltxv-modal
   modal deploy ltxv_app.py
   ```

2. Modal will build the image and deploy the application, providing URLs for the endpoints:
   ```
   ✓ Created objects.
   ├── 🔨 Created mount [path]
   ├── 🔨 Created function LTXVideoModel.*.
   ├── 🔨 Created function main.
   ├── 🔨 Created web endpoint for LTXVideoModel.generate_video_api
   ├── 🔨 Created web endpoint for LTXVideoModel.condition_video_api
   ├── 🔨 Created web endpoint for LTXVideoModel.info
   └── 🔨 Created web endpoint for LTXVideoModel.batch_api
   ✓ App deployed! 🎉
   ```

3. Update the client with the correct endpoint URLs:
   ```bash
   cd ltxv-modal
   python client_example.py set-url your-deployment-url
   ```

## Testing the Deployment

1. Test the info endpoint:
   ```bash
   python client_example.py info
   ```

2. Generate a simple video:
   ```bash
   python client_example.py generate "A simple test video" --width 640 --height 480 --frames 25 --steps 20
   ```

3. For image-to-video generation:
   ```bash
   python client_example.py image-to-video "An image turning into a video" path/to/your/image.jpg
   ```

4. For batch generation:
   ```bash
   python client_example.py batch --prompts "Prompt 1" "Prompt 2" "Prompt 3"
   ```

## Important Implementation Details

### Model Loading

The implementation includes these important features:

1. **Efficient Model Loading**: 
   - Uses huggingface_hub to download model weights directly
   - Manages torch precision with bfloat16 for optimal performance
   - Implements proper CPU offloading for memory management

2. **Required Libraries**:
   - PyTorch with CUDA support
   - transformers and diffusers for model framework
   - imageio[ffmpeg] for video processing
   - einops and timm for model components
   - accelerate for optimized loading

3. **Error Handling**:
   - Robust try/except blocks for model loading
   - Proper cleanup for temporary files
   - Validation of input parameters

### Video Processing

1. **Tensor to Video Conversion**:
   - Proper normalization and format conversion
   - In-memory video encoding using imageio
   - Base64 encoding for API transmission

2. **Conditioning Handling**:
   - Support for both image and video conditioning
   - Multi-conditioning with different start frames
   - Strength adjustment for each conditioning item

### Memory Optimization

1. **CPU Offloading**:
   ```python
   self.pipeline = self.pipeline.to(device)
   ```

2. **Precision Control**:
   ```python
   model = model.to(torch.bfloat16)
   ```

3. **Proper Tensor Management**:
   - Release unused tensors promptly
   - Manage batch sizes carefully

## Model Requirements

The LTX-Video model requires significant resources:

- GPU: A100-40GB or equivalent (at least 16GB VRAM)
- Disk Space: ~10GB for model weights
- Memory: ~16GB RAM recommended

## Monitoring and Logs

To view the logs for your deployment:

```bash
modal app logs ltxv-video
```

This shows all server-side logs, including model loading times, inference times, and any errors.

## Customization Options

You can customize various aspects of the API:

1. **Model Version**: Edit the repo_id and model_filename in ltxv_app.py if a newer version is released
2. **Default Parameters**: Modify the default values for height, width, frames, etc.
3. **GPU Selection**: Change the GPU type in the @app.cls decorator if needed (e.g., "A100" to "T4")
