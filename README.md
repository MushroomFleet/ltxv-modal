# 🚀✨ LTXV-VIDEO MODAL API ✨🚀

> *A serverless API for LTX-Video text-to-video and image-to-video generation with OPTIMIZED COLD STARTS!* 🎬🎨

Generate high-quality videos with the power of the LTX-Video model in a **SERVERLESS** ☁️ environment using Modal! No more GPU hassles! No more dependency conflicts! Just pure, beautiful video generation! 🤯

![LTX-Video](https://img.shields.io/badge/LTX--Video-2B%20Parameters-blueviolet) ![GPU](https://img.shields.io/badge/GPU-A100--40GB-brightgreen) ![Modal](https://img.shields.io/badge/Platform-Modal-blue) ![FastAPI](https://img.shields.io/badge/API-FastAPI-green)

## 🤔 What is This? 🤔

This project lets you deploy the **LTX-Video model** (a powerful 2B parameter text-to-video model) as a serverless API using Modal. It handles all the infrastructure so you can focus on generating amazing videos! ✨🎥✨

The implementation uses **Modal Volumes** to store model weights and **Memory Snapshots** for BLAZING FAST ⚡ cold starts, making everything FASTER 🏎️💨 and MORE RELIABLE 🔒 than ever before!

## ✨ Features That'll Blow Your Mind ✨

- 🔥 **SERVERLESS DEPLOYMENT**: No servers to manage! Modal handles EVERYTHING!
- ⚡ **OPTIMIZED COLD STARTS**: Memory snapshots = LIGHTNING FAST STARTUP!
- 🦸‍♂️ **CONCURRENT LOADING**: Model components load in parallel for maximum speed!
- 🧠 **POWERFUL MODEL**: 2B parameter LTX-Video model for high-quality video generation!
- 🚄 **LIGHTNING FAST**: Uses A100-40GB GPUs for optimized inference!
- 📦 **VOLUME STORAGE**: Store model weights for faster startup times!
- 🔄 **AUTO-SCALING**: Handle peak loads with dynamic container scaling!
- 🔌 **MULTIPLE ENDPOINTS**: Text-to-video, image-to-video, batch processing - WE GOT IT ALL!
- 🎛️ **CUSTOMIZATION**: Control height, width, frames, FPS - TWEAK ALL THE THINGS!
- 🔋 **OPTIMIZED**: CPU-to-GPU memory snapshots, bfloat16 precision, ThreadPoolExecutor magic!

## 🛠️ Prerequisites Before The Magic Happens 🛠️

- 💻 Python 3.10+ (the fresher the better!)
- 🔑 Modal account (sign up at [modal.com](https://modal.com) - it's FREE to start!)
- 🌐 Internet connection (to download the model on first run)
- 🤩 An imagination ready to be UNLEASHED in video form!

## 🚀 Installation: Let's Summon The Beast 🚀

1. 📦 **Install Modal CLI**:
   ```bash
   pip install modal
   ```

2. 🔐 **Authenticate with Modal**:
   ```bash
   modal token new
   ```

3. 📥 **Clone this repository**:
   ```bash
   git clone https://github.com/MushroomFleet/ltxv-modal
   cd ltxv-modal
   ```

## 🧙‍♂️ Deployment: The Grand Conjuring 🧙‍♂️

1. 🚀 **Deploy the Application**:
   ```bash
   modal deploy ltxv_app.py
   ```

2. 🔍 **Note Your Deployment URLs** (or check your `.api_url` file):
   Modal will show something like:
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

3. 🧪 **Test the Deployment**:
   ```bash
   python test-generate.py
   ```

> 🔍 **FIRST RUN WARNING**: The first time you run, the model will download from HuggingFace and save to a volume (~5-10 minutes). Subsequent runs will be MUCH faster thanks to our memory snapshot optimizations! PATIENCE, YOUNG SORCERER! ⏳

## 🔮 Usage: Unleash The Video-Creating Kraken 🔮

### 🎥 Generate a Video from Text

```bash
# Simple test with default settings
python test-generate.py

# High-quality test with custom settings
python test-generate-hd.py

# Using the Python client
python client_example.py generate "A majestic eagle soaring over a mountain range at sunset" --width 1280 --height 736
```

### 🎨 Generate a Video from an Image

```bash
# Using the test script
python test-img2vid.py --image your_image.jpg

# Using the Python client
python client_example.py image-to-video "An image turning into a video" your_image.jpg
```

### 📚 Batch Generate Multiple Videos

```bash
# Using the client
python client_example.py batch --prompts "A cat wizard" "A dog astronaut" "A rabbit pirate"
```

### 💁‍♀️ Get Model Information

```bash
python client_example.py info
```

## 🚀 Performance Optimizations: We Made It FASTER! 🚀

We've turbocharged this API with cutting-edge optimizations:

### 🌩️ Memory Snapshots

Our implementation uses Modal's memory snapshots for incredibly fast cold starts:

```python
@modal.enter(snap=True)
def load_model_to_cpu(self):
    # Load model to CPU for snapshotting
    
@modal.enter(snap=False)
def move_model_to_gpu(self):
    # Move model to GPU after snapshot restoration
```

This two-stage approach loads models to CPU first for snapshotting, then quickly transfers to GPU for inference!

### 🧵 Concurrent Model Loading

We use ThreadPoolExecutor to load model components in parallel:

```python
with ThreadPoolExecutor(max_workers=5) as executor:
    future_to_component = {
        executor.submit(load_vae): "vae",
        executor.submit(load_transformer): "transformer",
        # ... other components
    }
```

This slashes initialization time from minutes to seconds! 🤯

### 📊 Dynamic Container Scaling

Intelligent container scaling based on time of day and workload:

```python
@app.cls(
    scaledown_window=300,  # Keep containers warm for 5 minutes
    min_containers=1,      # Always have one ready to go!
    buffer_containers=1    # Have an extra on standby during active periods
)
```

Peak performance when you need it, cost efficiency when you don't!

## 🌟 API Endpoints: For the Tech-Savvy Wizards 🌟

### 1. 🎥 Generate Video from Text
```
GET /generate_video_api?prompt=your_prompt_here
```
Parameters:
- `prompt`: Your imagination in words ✨
- `negative_prompt`: What you don't want to see
- `height`: Video height (default: 736)
- `width`: Video width (default: 1280)
- `num_frames`: Number of frames (default: 73)
- `frame_rate`: Frames per second (default: 25)
- `guidance_scale`: Creativity control (default: 3.0)
- `num_inference_steps`: Quality control (default: 40)
- `seed`: Reproducibility magic (optional)
- `stg_scale`: Spatiotemporal guidance scale (default: 1.0)
- `stg_rescale`: Spatiotemporal guidance rescale (default: 0.7)

### 2. 🎨 Generate Video from Image/Video
```
POST /condition_video_api
```
JSON body with media data in base64 format.

### 3. 📚 Batch Generate Videos
```
POST /batch_api
```
JSON body with multiple prompts and parameters.

### 4. ℹ️ Get Model Information
```
GET /info
```

### 5. 📊 Get Profile Information
```
GET /profile_info
```

### 6. 🔄 Set Scaling Profile
```
POST /set_profile?profile=solo|team
```

## 📊 Performance: It's FAST (Well, for Video Generation) 📊

With our optimized implementation, you get:

- 🚀 **BLAZING COLD STARTS**: Memory snapshots cut startup time by up to 70%!
- 🧠 **EFFICIENT LOADING**: Concurrent model loading slashes initialization time!
- 🌐 **REDUCED BANDWIDTH**: Download once, use forever with Modal Volumes!
- 🔒 **BETTER RELIABILITY**: Less dependency on external APIs!
- 📈 **SMART SCALING**: Dynamically scales containers based on demand!

```
First run: ~5-10 minutes (downloads & saves model)
Subsequent cold starts: ~10-15 seconds (with memory snapshots)
Warm container response: Nearly instant!
```

## 🩺 Troubleshooting: When The Magic Goes Sideways 🩺

### 💥 Memory Issues

If you're getting OOM (Out Of Memory) errors:
- 📏 Reduce video dimensions (try 512x512)
- 🎞️ Generate fewer frames (try 33 or 25)
- 🔢 Use fewer inference steps (try 20)
- 📉 Lower batch size

### 🐌 Slow First Run

- 😴 This is normal! The model is being downloaded and saved to the volume.
- ⏱️ Subsequent runs will be MUCH faster thanks to memory snapshots!

### 🔌 Can't Connect to API

- 🔍 Double-check your deployment URL
- 🧹 Make sure you have quota/credits on Modal
- 📋 Check logs with `modal app logs ltxv-video`

### 🌐 FastAPI Web Endpoint Issues

- 🔍 Make sure you're using `fastapi[standard]` in your image definition
- 🧩 Check that all functions with `@modal.fastapi_endpoint` use the same image

## 📚 Documentation: The Sacred Scrolls 📚

- [LTX-Video GitHub](https://github.com/Lightricks/LTX-Video): Official repository
- [LTX-Video Paper](https://arxiv.org/abs/2501.00103): Research paper
- [Modal Documentation](https://modal.com/docs): Modal platform docs
- [Memory Snapshots](https://modal.com/docs/guide/memory-snapshot): Learn about Modal's memory snapshots
- [FastAPI Documentation](https://fastapi.tiangolo.com/): FastAPI docs for endpoint development

## 🙏 Acknowledgments 🙏

- 🌟 [Lightricks](https://github.com/Lightricks) for the amazing LTX-Video model
- 🚀 [Modal](https://modal.com) for the incredible serverless platform
- 🤗 [Hugging Face](https://huggingface.co) for hosting the model
- 🎬 The video generation community for inspiration!
- 🧙‍♂️ The Modal engineering team for the memory snapshot feature!

---

> 💫 Made with ✨MAGICAL ENERGY✨, 🧪 TECHNICAL WIZARDRY 🧪, and too much caffeine ☕☕☕

```
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣀⣤⣴⣶⣶⣶⣶⣶⣶⣦⣤⣄⡀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣠⣶⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣦⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣠⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣷⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢀⣾⣿⣿⣿⣿⣿⣿⠿⠿⠛⠛⠛⠛⠛⠿⠿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⡿⠟⠋⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⠙⠿⣿⣿⣿⣿⣿⣿⡆⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢠⣿⣿⣿⣿⠏⠀⠀⠀⠀⢀⣠⣴⣶⣶⣶⣶⣤⣀⠀⠀⠀⠀⠈⢻⣿⣿⣿⣿⣿⡄⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⠀⠀⠀⢀⣴⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣦⡀⠀⠀⠀⢿⣿⣿⣿⣿⣧⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⣿⣿⣿⣿⣿⡀⠀⣰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣆⠀⠀⢸⣿⣿⣿⣿⣿⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⣿⣇⢰⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡄⠀⣼⣿⣿⣿⣿⠏⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡗⢠⣿⣿⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⣿⡿⠿⠟⠛⠛⠛⠛⠿⢿⣿⣿⣿⣿⣿⣿⠇⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠘⣿⣿⣿⣿⣿⣿⣟⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠉⢹⣿⣿⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢹⣿⣿⡿⡿⡿⡏⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢸⣿⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⢻⣿⣧⡀⠃⠑⠠⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠸⣿⠃⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⠿⣿⣶⠄⠂⠈⠁⠐⠠⣄⠀⠀⠀⠀⠀⠀⠀⠀⠀⣸⡿⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠨⢿⣧⣐⡀⠀⠀⠀⠀⠑⢄⡐⠀⠀⠀⠀⢀⣴⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣦⡀⢢⠀⠀⠀⠀⠀⠀⠀⣰⣿⠟⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣦⡑⢄⠀⠀⢀⣴⣿⡿⠁⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠙⢿⣿⣮⣡⣶⣯⡿⠋⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠈⠛⠿⠛⠉⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀⠀
