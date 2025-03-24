# 🚀✨ LTXV-VIDEO MODAL API ✨🚀

> *A serverless API for LTX-Video text-to-video and image-to-video generation* 🎬🎨

! UNDER CONSTRUCTION !

Generate high-quality videos with the power of the LTX-Video model in a **SERVERLESS** ☁️ environment using Modal! No more GPU hassles! No more dependency conflicts! Just pure, beautiful video generation! 🤯

![LTX-Video](https://img.shields.io/badge/LTX--Video-2B%20Parameters-blueviolet) ![GPU](https://img.shields.io/badge/GPU-A100--40GB-brightgreen) ![Modal](https://img.shields.io/badge/Platform-Modal-blue)

## 🤔 What is This? 🤔

This project lets you deploy the **LTX-Video model** (a powerful 2B parameter text-to-video model) as a serverless API using Modal. It handles all the infrastructure so you can focus on generating amazing videos! ✨🎥✨

The implementation uses **Modal Volumes** to store model weights, making everything FASTER 🏎️💨 and MORE RELIABLE 🔒 than ever before!

## ✨ Features That'll Blow Your Mind ✨

- 🔥 **SERVERLESS DEPLOYMENT**: No servers to manage! Modal handles EVERYTHING!
- 🧠 **POWERFUL MODEL**: 2B parameter LTX-Video model for high-quality video generation!
- 🚄 **LIGHTNING FAST**: Uses A100-40GB GPUs for optimized inference!
- 📦 **VOLUME STORAGE**: Store model weights for faster startup times!
- 🔄 **AUTO-SCALING**: Handles as many requests as you throw at it!
- 🔌 **MULTIPLE ENDPOINTS**: Text-to-video, image-to-video, batch processing - WE GOT IT ALL!
- 🎛️ **CUSTOMIZATION**: Control height, width, frames, FPS - TWEAK ALL THE THINGS!
- 🔋 **OPTIMIZED**: CPU offloading, bfloat16 precision, memory optimization!

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

2. 📝 **Note Your Deployment URL**:
   ```
   https://yourname--ltxv-video
   ```

3. 🧪 **Test the Deployment**:
   ```bash
   python test-generate.py --url https://yourname--ltxv-video
   ```

4. 🎉 **MARVEL at your Creation**:
   ```bash
   python test-generate.py --display
   ```

> 🔍 **FIRST RUN WARNING**: The first time you run, the model will download from HuggingFace and save to a volume (~5-10 minutes). Subsequent runs will be MUCH faster. PATIENCE, YOUNG SORCERER! ⏳

## 🔮 Usage: Unleash The Video-Creating Kraken 🔮

### 🎥 Generate a Video from Text

```bash
# Using the test script (EASIEST WAY)
python test-generate.py

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

### 🌟 Advanced Multi-Conditioning

```bash
# Using multiple conditioning images/videos at different points in the timeline
python client_example.py multi-condition "A journey through seasons" --media spring.jpg summer.jpg fall.jpg winter.jpg --frames 0 18 36 54
```

### 🔍 Check Model Information

```bash
python client_example.py info
```

## 🌟 API Endpoints: For the Tech-Savvy Wizards 🌟

### 1. 🎥 Generate Video from Text
```
GET /LTXVideoModel/generate_video_api?prompt=your_prompt_here
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
POST /LTXVideoModel/condition_video_api
```
JSON body with media data in base64 format.

### 3. 📚 Batch Generate Videos
```
POST /LTXVideoModel/batch_api
```
JSON body with multiple prompts and parameters.

### 4. ℹ️ Get Model Information
```
GET /LTXVideoModel/info
```

## 📊 Performance: It's FAST (Well, for Video Generation) 📊

With the Modal Volumes implementation, you get:

- 🚀 **FASTER STARTUP**: No more downloading the model every time!
- 🌐 **REDUCED BANDWIDTH**: Download once, use forever!
- 🔒 **BETTER RELIABILITY**: Less dependency on external APIs!
- 🧠 **SMART LOADING**: Automatically uses volume if available!

```
First run: ~5-10 minutes (downloads & saves model)
Subsequent runs: ~1-2 minutes (loads from volume)
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
- ⏱️ Subsequent runs will be MUCH faster!

### 🔌 Can't Connect to API

- 🔍 Double-check your deployment URL
- 🧹 Make sure you have quota/credits on Modal
- 📋 Check logs with `modal app logs ltxv-video`

## 📚 Documentation: The Sacred Scrolls 📚

- [LTX-Video GitHub](https://github.com/Lightricks/LTX-Video): Official repository
- [LTX-Video Paper](https://arxiv.org/abs/2501.00103): Research paper
- [Modal Documentation](https://modal.com/docs): Modal platform docs

## 🙏 Acknowledgments 🙏

- 🌟 [Lightricks](https://github.com/Lightricks) for the amazing LTX-Video model
- 🚀 [Modal](https://modal.com) for the incredible serverless platform
- 🤗 [Hugging Face](https://huggingface.co) for hosting the model
- 🎬 The video generation community for inspiration!

---

> 💫 Made with ✨MAGICAL ENERGY✨ and too much caffeine ☕☕☕
