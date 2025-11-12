# VGGT: Visual Geometry Grounded Transformer

A 3D reconstruction system that generates point clouds and camera poses from video or image sequences.

## Prerequisites

- Linux (tested on Ubuntu/WSL2)
- CUDA-compatible GPU (recommended)
- **Option A**: [Docker](https://docs.docker.com/get-docker/) with [NVIDIA Container Toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)
- **Option B**: [Pixi](https://pixi.sh) package manager

## Quick Start with Docker (Recommended)

The easiest way to run VGGT is using Docker. The Docker image comes with:
- All dependencies pre-installed
- **VGGT model pre-downloaded** (~4.68GB) - no wait time!
- Proper CUDA setup
- Ready to use immediately

### Build the Docker Image

**Using the build script (recommended):**

```bash
./docker-build.sh
```

This builds the image as `ghcr.io/infinitespacesorg/vggt:latest` and tags it locally as `vggt:latest`.

**Build with specific version:**

```bash
./docker-build.sh v1.0
```

**Manual build:**

```bash
docker build -t ghcr.io/infinitespacesorg/vggt:latest -t vggt:latest .
```

**Note**: The first build takes 10-15 minutes as it downloads and caches the model.

### Pull Pre-built Image (Coming Soon)

Once published, you can pull the pre-built image:

```bash
docker pull ghcr.io/infinitespacesorg/vggt:latest
```

### Run with Docker

**Option 1: Using the helper script**

```bash
./docker-run.sh
```

**Option 2: Using Docker Compose**

```bash
docker-compose up
```

**Option 3: Manual Docker command**

```bash
# With GPU support (using local tag)
docker run -it --rm --gpus all -p 7860:7860 vggt:latest

# Or using ghcr.io image
docker run -it --rm --gpus all -p 7860:7860 ghcr.io/infinitespacesorg/vggt:latest

# CPU only
docker run -it --rm -p 7860:7860 vggt:latest
```

**Run from ghcr.io without building:**

```bash
./docker-run.sh ghcr.io/infinitespacesorg/vggt:latest
```

### Access the Demo

Once running, open your browser to:
- **http://localhost:7860**

To stop the container:
- Press `Ctrl+C` in the terminal

### Docker Benefits

✅ **No model download wait** - Model is pre-downloaded in the image  
✅ **Consistent environment** - Same setup everywhere  
✅ **Easy deployment** - Single command to run  
✅ **Isolated** - No conflicts with system packages  
✅ **GPU support** - Automatic CUDA setup  

### Publishing to GitHub Container Registry

To push your built image to ghcr.io:

```bash
# Login to GitHub Container Registry
echo $GITHUB_TOKEN | docker login ghcr.io -u USERNAME --password-stdin

# Build and push
./docker-build.sh latest
docker push ghcr.io/infinitespacesorg/vggt:latest

# Or push with version tag
./docker-build.sh v1.0
docker push ghcr.io/infinitespacesorg/vggt:v1.0
```

**Note**: You need write access to the `infinitespacesorg` organization on GitHub.

---

## Installation with Pixi (Alternative)

### 1. Install Pixi

```bash
curl -fsSL https://pixi.sh/install.sh | bash
```

After installation, restart your shell or run:

```bash
source ~/.bashrc
```

Verify installation:

```bash
pixi --version
```

### 2. Clone the Repository

```bash
cd /path/to/your/projects
git clone <repository-url>
cd vggt
```

### 3. Install Dependencies with Pixi

Pixi will automatically install all dependencies including PyTorch 2.3.1, Python 3.12, and all required packages:

```bash
pixi install -e demo
```

This creates an isolated environment with:
- Python 3.12.12
- PyTorch 2.3.1
- torchvision 0.18.*
- All demo dependencies (gradio, opencv, scipy, matplotlib, trimesh, etc.)

## Running the Demo

### Method 1: Using Pixi Task (Recommended)

```bash
pixi run -e demo demo
```

### Method 2: Direct Python Command

```bash
pixi run -e demo python demo_gradio.py
```

The demo will:
1. Download the VGGT-1B model (~4.68 GB) on first run
2. Initialize the model (may take a few minutes)
3. Launch a Gradio web interface

### Accessing the Interface

By default, the demo creates a **public shareable URL** via Gradio's tunneling service. You'll see output like:

```
Running on local URL:  http://127.0.0.1:7860
Running on public URL: https://xxxxx.gradio.live
```

- **Local access**: http://127.0.0.1:7860
- **Remote access**: Use the public URL (valid for 72 hours)

### Making the Demo Local-Only

To disable public sharing and keep the demo local-only:

1. Edit `demo_gradio.py`
2. Find line 685: `demo.queue(max_size=20).launch(show_error=True, share=True)`
3. Change to: `demo.queue(max_size=20).launch(show_error=True, share=False)`

## Using the Demo

1. **Upload Input**:
   - Upload a video (frames extracted at 1 FPS), OR
   - Upload multiple images

2. **Preview**: Your images appear in the gallery

3. **Reconstruct**: Click the "Reconstruct" button to start 3D processing

4. **Visualize**: The 3D point cloud and camera poses appear in the viewer

5. **Adjust Visualization** (Optional):
   - Confidence Threshold: Filter points by confidence
   - Show Points from Frame: Select specific frames
   - Show Camera: Toggle camera visualization
   - Filter Sky/Black/White Background: Remove unwanted points
   - Prediction Mode: Choose between "Depthmap and Camera Branch" or "Pointmap Branch"

6. **Download**: Save the GLB file for use in other 3D applications

## COLMAP Export

VGGT can export reconstructions to COLMAP format for use with other 3D reconstruction tools like Gaussian Splatting, NeRF, etc.

### Basic Usage

**From images:**
```bash
pixi run -e demo python vggt_process.py --image_dir path/to/images --output_dir colmap_output
```

**From video:**
```bash
pixi run -e demo python vggt_process.py --video_file path/to/video.mp4 --output_dir colmap_output
```

### Command Line Arguments

**Input (one required):**
- `--image_dir` - Directory containing input images (PNG, JPG, JPEG)
- `--video_file` - Path to input video file (will extract frames automatically)

**Optional:**
- `--output_dir` - Output directory for COLMAP files (default: `colmap_output`)
- `--fps` - Frames per second to extract from video (default: `1.0`)
  - Only used with `--video_file`
  - Example: `--fps 2.0` extracts 2 frames per second
- `--conf_threshold` - Confidence threshold percentage 0-100 for including points (default: `50.0`)
  - Higher values = fewer but more confident points
  - Lower values = more points but may include noise
- `--stride` - Point sampling stride (default: `1`)
  - Higher values = fewer points, faster processing
  - Example: `--stride 2` samples every 2nd pixel
- `--prediction_mode` - Which prediction branch to use (default: `Depthmap and Camera Branch`)
  - `"Depthmap and Camera Branch"` - Uses depth maps and camera estimates
  - `"Pointmap Branch"` - Uses direct 3D point predictions

**Filtering Options:**
- `--mask_sky` - Filter out sky regions (downloads sky segmentation model)
- `--mask_black_bg` - Filter out dark/black background points
- `--mask_white_bg` - Filter out bright/white background points

**Output Format:**
- `--binary` - Output binary COLMAP files (.bin) instead of text (.txt)

### Examples

**Process video file:**
```bash
pixi run -e demo python vggt_process.py \
    --video_file path/to/video.mp4 \
    --output_dir colmap_output
```

**Process video with custom frame rate:**
```bash
pixi run -e demo python vggt_process.py \
    --video_file path/to/video.mp4 \
    --fps 2.0 \
    --image_dir my_extracted_frames \
    --output_dir colmap_output
```

**Basic reconstruction from images:**
```bash
pixi run -e demo python vggt_process.py \
    --image_dir examples/kitchen/images \
    --output_dir kitchen_colmap
```

**High-quality reconstruction with filtering:**
```bash
pixi run -e demo python vggt_process.py \
    --image_dir examples/room/images \
    --output_dir room_colmap \
    --conf_threshold 70.0 \
    --mask_sky \
    --mask_black_bg
```

**Fast reconstruction with fewer points:**
```bash
pixi run -e demo python vggt_process.py \
    --image_dir examples/llff_fern/images \
    --output_dir fern_colmap \
    --stride 2 \
    --conf_threshold 60.0
```

**Binary output for faster loading:**
```bash
pixi run -e demo python vggt_process.py \
    --image_dir path/to/images \
    --output_dir colmap_output \
    --binary
```

**Using Pointmap Branch:**
```bash
pixi run -e demo python vggt_process.py \
    --image_dir path/to/images \
    --output_dir colmap_output \
    --prediction_mode "Pointmap Branch"
```

**Process video with all options:**
```bash
pixi run -e demo python vggt_process.py \
    --video_file examples/videos/room.mp4 \
    --fps 1.0 \
    --image_dir room_frames \
    --output_dir room_colmap \
    --conf_threshold 60.0 \
    --mask_sky \
    --stride 1 \
    --binary
```

### Output Files

The script generates COLMAP-compatible files:

**Text format (default):**
- `cameras.txt` - Camera intrinsics
- `images.txt` - Camera poses and 2D keypoints
- `points3D.txt` - 3D points with colors and tracks

**Binary format (with --binary):**
- `cameras.bin`
- `images.bin`
- `points3D.bin`

These files can be used directly with:
- COLMAP's GUI and processing tools
- Gaussian Splatting training
- NeRF training
- Other 3D reconstruction pipelines

## Stopping the Demo

If the demo is running in the background:

```bash
# Method 1: Kill by name
pkill -f demo_gradio.py

# Method 2: Find PID and kill
ps aux | grep demo_gradio.py
kill <PID>

# Method 3: Force kill if needed
pkill -9 -f demo_gradio.py
```

If running in terminal:
- Press `Ctrl+C` to stop

## Verifying Installation

Test that all dependencies are working:

```bash
pixi run -e demo python -c "import torch, torchvision, cv2, gradio, numpy, scipy, matplotlib, trimesh; print('All imports successful!'); print(f'PyTorch: {torch.__version__}')"
```

Expected output:
```
All imports successful!
PyTorch: 2.3.1.post100
```

## Troubleshooting

### Docker Troubleshooting

**GPU not detected in Docker:**

Ensure NVIDIA Container Toolkit is installed:
```bash
# Install nvidia-container-toolkit
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
    sudo tee /etc/apt/sources.list.d/nvidia-docker.list
sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo systemctl restart docker
```

Verify GPU is available:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

**Cannot connect to Docker daemon:**
```bash
sudo systemctl start docker
sudo usermod -aG docker $USER
# Log out and back in for group changes to take effect
```

**Port 7860 already in use:**
```bash
# Use a different port
docker run -it --rm --gpus all -p 8080:7860 vggt:latest
# Then access at http://localhost:8080
```

**Rebuild without cache:**
```bash
docker build --no-cache -t vggt:latest .
```

**View logs from running container:**
```bash
docker logs vggt-demo
```

**Access running container shell:**
```bash
docker exec -it vggt-demo /bin/bash
```

### Model Download Issues (Pixi Installation)

The model is downloaded from Hugging Face on first run. If download fails:
- Check internet connection
- Ensure you have ~5GB free disk space in `~/.cache/torch/hub/checkpoints/`
- Try running again (download resumes automatically)

### CUDA Not Available

The demo will use CPU if CUDA is not available. This is normal on systems without NVIDIA GPUs or in WSL2 environments. The model will still work but may be slower.

To check CUDA status:

```bash
pixi run -e demo python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Port Already in Use

If port 7860 is already in use, Gradio will automatically try the next available port (7861, 7862, etc.). Check the console output for the actual URL.

### Memory Issues

The VGGT-1B model requires significant memory. If you encounter OOM errors:
- Close other applications
- Use fewer/smaller input images
- Reduce the number of frames extracted from videos

## Project Structure

```
vggt/
├── demo_gradio.py          # Main Gradio web demo
├── demo_viser.py           # Alternative Viser demo
├── vggt_to_colmap.py       # Export to COLMAP format
├── visual_util.py          # Visualization utilities
├── vggt/                   # Core VGGT package
│   ├── models/             # Model definitions
│   └── utils/              # Utility functions
├── examples/               # Example images and videos
│   ├── kitchen/
│   ├── room/
│   ├── llff_fern/
│   └── ...
├── Dockerfile              # Docker container definition
├── docker-compose.yml      # Docker Compose configuration
├── docker-build.sh         # Build Docker image script
├── docker-run.sh           # Run Docker container script
├── pyproject.toml          # Package configuration (Pixi + pip)
├── requirements.txt        # Legacy pip requirements
├── README.md               # This file
└── DOCKER.md               # Detailed Docker documentation
```

## Why Pixi?

Pixi provides several advantages over pip for this project:

1. **Correct PyTorch Version**: Installs PyTorch 2.3.1 (not available via pip for Python 3.13)
2. **Better Dependency Management**: Handles complex C++ dependencies (CUDA, MKL, OpenCV)
3. **Reproducible Environments**: Creates lockfiles for exact dependency versions
4. **Conda + PyPI Integration**: Uses conda-forge for system libraries, PyPI for Python packages
5. **Isolated Environments**: No conflicts with system Python or other projects

## Development

The package is installed in editable mode, so changes to the code take effect immediately:

```bash
pixi run -e demo python -c "from vggt.models.vggt import VGGT; print(VGGT)"
```

## Environment Information

The Pixi environment includes:

**Conda packages (from conda-forge and pytorch channels):**
- python 3.12.x
- pytorch 2.3.*
- torchvision 0.18.*
- pillow <12.0

**PyPI packages:**
- gradio >=5.17.1
- viser 0.2.23
- opencv-python
- scipy, matplotlib, numpy
- trimesh
- And all other dependencies from `pyproject.toml`

## License

See LICENSE file for details.

## Citation

If you use VGGT in your research, please cite the original paper.

## Links

- 🐙 [GitHub Repository](https://github.com/facebookresearch/vggt)
- 📄 Project Page (link)
- 🤗 [Model on Hugging Face](https://huggingface.co/facebook/VGGT-1B)

---

**Note**: This project uses Pixi for package management. While `requirements.txt` exists for legacy compatibility, it is recommended to use Pixi for installation to ensure all dependencies are correctly resolved.
