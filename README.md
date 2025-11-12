<<<<<<< HEAD
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
=======
<div align="center">
<h1>VGGT: Visual Geometry Grounded Transformer</h1>

<a href="https://jytime.github.io/data/VGGT_CVPR25.pdf" target="_blank" rel="noopener noreferrer">
  <img src="https://img.shields.io/badge/Paper-VGGT" alt="Paper PDF">
</a>
<a href="https://arxiv.org/abs/2503.11651"><img src="https://img.shields.io/badge/arXiv-2503.11651-b31b1b" alt="arXiv"></a>
<a href="https://vgg-t.github.io/"><img src="https://img.shields.io/badge/Project_Page-green" alt="Project Page"></a>
<a href="https://huggingface.co/spaces/facebook/vggt"><img src='https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Demo-blue'></a>


**[Visual Geometry Group, University of Oxford](https://www.robots.ox.ac.uk/~vgg/)**; **[Meta AI](https://ai.facebook.com/research/)**


[Jianyuan Wang](https://jytime.github.io/), [Minghao Chen](https://silent-chen.github.io/), [Nikita Karaev](https://nikitakaraevv.github.io/), [Andrea Vedaldi](https://www.robots.ox.ac.uk/~vedaldi/), [Christian Rupprecht](https://chrirupp.github.io/), [David Novotny](https://d-novotny.github.io/)
</div>

```bibtex
@inproceedings{wang2025vggt,
  title={VGGT: Visual Geometry Grounded Transformer},
  author={Wang, Jianyuan and Chen, Minghao and Karaev, Nikita and Vedaldi, Andrea and Rupprecht, Christian and Novotny, David},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  year={2025}
}
```

## Updates

- [July 29, 2025] We've updated the license for VGGT to permit **commercial use** (excluding military applications). All code in this repository is now under a commercial-use-friendly license. However, only the newly released checkpoint [**VGGT-1B-Commercial**](https://huggingface.co/facebook/VGGT-1B-Commercial) is licensed for commercial usage — the original checkpoint remains non-commercial. Full license details are available [here](https://github.com/facebookresearch/vggt/blob/main/LICENSE.txt). Access to the checkpoint requires completing an application form, which is processed by a system similar to LLaMA's approval workflow, automatically. The new checkpoint delivers similar performance to the original model. Please submit an issue if you notice a significant performance discrepancy.



- [July 6, 2025] Training code is now available in the `training` folder, including an example to finetune VGGT on a custom dataset. 


- [June 13, 2025] Honored to receive the Best Paper Award at CVPR 2025! Apologies if I’m slow to respond to queries or GitHub issues these days. If you’re interested, our oral presentation is available [here](https://docs.google.com/presentation/d/1JVuPnuZx6RgAy-U5Ezobg73XpBi7FrOh/edit?usp=sharing&ouid=107115712143490405606&rtpof=true&sd=true). Another long presentation can be found [here](https://docs.google.com/presentation/d/1aSv0e5PmH1mnwn2MowlJIajFUYZkjqgw/edit?usp=sharing&ouid=107115712143490405606&rtpof=true&sd=true) (Note: it’s shared in .pptx format with animations — quite large, but feel free to use it as a template if helpful.)


- [June 2, 2025] Added a script to run VGGT and save predictions in COLMAP format, with bundle adjustment support optional. The saved COLMAP files can be directly used with [gsplat](https://github.com/nerfstudio-project/gsplat) or other NeRF/Gaussian splatting libraries.


- [May 3, 2025] Evaluation code for reproducing our camera pose estimation results on Co3D is now available in the [evaluation](https://github.com/facebookresearch/vggt/tree/evaluation) branch. 


## Overview

Visual Geometry Grounded Transformer (VGGT, CVPR 2025) is a feed-forward neural network that directly infers all key 3D attributes of a scene, including extrinsic and intrinsic camera parameters, point maps, depth maps, and 3D point tracks, **from one, a few, or hundreds of its views, within seconds**.


## Quick Start

First, clone this repository to your local machine, and install the dependencies (torch, torchvision, numpy, Pillow, and huggingface_hub). 

```bash
git clone git@github.com:facebookresearch/vggt.git 
cd vggt
pip install -r requirements.txt
```

Alternatively, you can install VGGT as a package (<a href="docs/package.md">click here</a> for details).


Now, try the model with just a few lines of code:

```python
import torch
from vggt.models.vggt import VGGT
from vggt.utils.load_fn import load_and_preprocess_images

device = "cuda" if torch.cuda.is_available() else "cpu"
# bfloat16 is supported on Ampere GPUs (Compute Capability 8.0+) 
dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16

# Initialize the model and load the pretrained weights.
# This will automatically download the model weights the first time it's run, which may take a while.
model = VGGT.from_pretrained("facebook/VGGT-1B").to(device)

# Load and preprocess example images (replace with your own image paths)
image_names = ["path/to/imageA.png", "path/to/imageB.png", "path/to/imageC.png"]  
images = load_and_preprocess_images(image_names).to(device)

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        # Predict attributes including cameras, depth maps, and point maps.
        predictions = model(images)
```

The model weights will be automatically downloaded from Hugging Face. If you encounter issues such as slow loading, you can manually download them [here](https://huggingface.co/facebook/VGGT-1B/blob/main/model.pt) and load, or:

```python
model = VGGT()
_URL = "https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt"
model.load_state_dict(torch.hub.load_state_dict_from_url(_URL))
```

## Detailed Usage

<details>
<summary>Click to expand</summary>

You can also optionally choose which attributes (branches) to predict, as shown below. This achieves the same result as the example above. This example uses a batch size of 1 (processing a single scene), but it naturally works for multiple scenes.

```python
from vggt.utils.pose_enc import pose_encoding_to_extri_intri
from vggt.utils.geometry import unproject_depth_map_to_point_map

with torch.no_grad():
    with torch.cuda.amp.autocast(dtype=dtype):
        images = images[None]  # add batch dimension
        aggregated_tokens_list, ps_idx = model.aggregator(images)
                
    # Predict Cameras
    pose_enc = model.camera_head(aggregated_tokens_list)[-1]
    # Extrinsic and intrinsic matrices, following OpenCV convention (camera from world)
    extrinsic, intrinsic = pose_encoding_to_extri_intri(pose_enc, images.shape[-2:])

    # Predict Depth Maps
    depth_map, depth_conf = model.depth_head(aggregated_tokens_list, images, ps_idx)

    # Predict Point Maps
    point_map, point_conf = model.point_head(aggregated_tokens_list, images, ps_idx)
        
    # Construct 3D Points from Depth Maps and Cameras
    # which usually leads to more accurate 3D points than point map branch
    point_map_by_unprojection = unproject_depth_map_to_point_map(depth_map.squeeze(0), 
                                                                extrinsic.squeeze(0), 
                                                                intrinsic.squeeze(0))

    # Predict Tracks
    # choose your own points to track, with shape (N, 2) for one scene
    query_points = torch.FloatTensor([[100.0, 200.0], 
                                        [60.72, 259.94]]).to(device)
    track_list, vis_score, conf_score = model.track_head(aggregated_tokens_list, images, ps_idx, query_points=query_points[None])
```


Furthermore, if certain pixels in the input frames are unwanted (e.g., reflective surfaces, sky, or water), you can simply mask them by setting the corresponding pixel values to 0 or 1. Precise segmentation masks aren't necessary - simple bounding box masks work effectively (check this [issue](https://github.com/facebookresearch/vggt/issues/47) for an example).

</details>


## Interactive Demo

We provide multiple ways to visualize your 3D reconstructions. Before using these visualization tools, install the required dependencies:

```bash
pip install -r requirements_demo.txt
```

### Interactive 3D Visualization

**Please note:** VGGT typically reconstructs a scene in less than 1 second. However, visualizing 3D points may take tens of seconds due to third-party rendering, independent of VGGT's processing time. The visualization is slow especially when the number of images is large.


#### Gradio Web Interface

Our Gradio-based interface allows you to upload images/videos, run reconstruction, and interactively explore the 3D scene in your browser. You can launch this in your local machine or try it on [Hugging Face](https://huggingface.co/spaces/facebook/vggt).


```bash
python demo_gradio.py
```

<details>
<summary>Click to preview the Gradio interactive interface</summary>

![Gradio Web Interface Preview](https://jytime.github.io/data/vggt_hf_demo_screen.png)
</details>


#### Viser 3D Viewer

Run the following command to run reconstruction and visualize the point clouds in viser. Note this script requires a path to a folder containing images. It assumes only image files under the folder. You can set `--use_point_map` to use the point cloud from the point map branch, instead of the depth-based point cloud.

```bash
python demo_viser.py --image_folder path/to/your/images/folder
```

## Exporting to COLMAP Format

We also support exporting VGGT's predictions directly to COLMAP format, by:

```bash 
# Feedforward prediction only
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/ 

# With bundle adjustment
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/ --use_ba

# Run with bundle adjustment using reduced parameters for faster processing
# Reduces max_query_pts from 4096 (default) to 2048 and query_frame_num from 8 (default) to 5
# Trade-off: Faster execution but potentially less robust reconstruction in complex scenes (you may consider setting query_frame_num equal to your total number of images) 
# See demo_colmap.py for additional bundle adjustment configuration options
python demo_colmap.py --scene_dir=/YOUR/SCENE_DIR/ --use_ba --max_query_pts=2048 --query_frame_num=5
```

Please ensure that the images are stored in `/YOUR/SCENE_DIR/images/`. This folder should contain only the images. Check the examples folder for the desired data structure. 

The reconstruction result (camera parameters and 3D points) will be automatically saved under `/YOUR/SCENE_DIR/sparse/` in the COLMAP format, such as:

``` 
SCENE_DIR/
├── images/
└── sparse/
    ├── cameras.bin
    ├── images.bin
    └── points3D.bin
```

## Integration with Gaussian Splatting


The exported COLMAP files can be directly used with [gsplat](https://github.com/nerfstudio-project/gsplat) for Gaussian Splatting training. Install `gsplat` following their official instructions (we recommend `gsplat==1.3.0`):

An example command to train the model is:
```
cd gsplat
python examples/simple_trainer.py  default --data_factor 1 --data_dir /YOUR/SCENE_DIR/ --result_dir /YOUR/RESULT_DIR/
```



## Zero-shot Single-view Reconstruction

Our model shows surprisingly good performance on single-view reconstruction, although it was never trained for this task. The model does not need to duplicate the single-view image to a pair, instead, it can directly infer the 3D structure from the tokens of the single view image. Feel free to try it with our demos above, which naturally works for single-view reconstruction.


We did not quantitatively test monocular depth estimation performance ourselves, but [@kabouzeid](https://github.com/kabouzeid) generously provided a comparison of VGGT to recent methods [here](https://github.com/facebookresearch/vggt/issues/36). VGGT shows competitive or better results compared to state-of-the-art monocular approaches such as DepthAnything v2 or MoGe, despite never being explicitly trained for single-view tasks. 



## Runtime and GPU Memory

We benchmark the runtime and GPU memory usage of VGGT's aggregator on a single NVIDIA H100 GPU across various input sizes. 

| **Input Frames** | 1 | 2 | 4 | 8 | 10 | 20 | 50 | 100 | 200 |
|:----------------:|:-:|:-:|:-:|:-:|:--:|:--:|:--:|:---:|:---:|
| **Time (s)**     | 0.04 | 0.05 | 0.07 | 0.11 | 0.14 | 0.31 | 1.04 | 3.12 | 8.75 |
| **Memory (GB)**  | 1.88 | 2.07 | 2.45 | 3.23 | 3.63 | 5.58 | 11.41 | 21.15 | 40.63 |

Note that these results were obtained using Flash Attention 3, which is faster than the default Flash Attention 2 implementation while maintaining almost the same memory usage. Feel free to compile Flash Attention 3 from source to get better performance.


## Research Progression

Our work builds upon a series of previous research projects. If you're interested in understanding how our research evolved, check out our previous works:


<table border="0" cellspacing="0" cellpadding="0">
  <tr>
    <td align="left">
      <a href="https://github.com/jytime/Deep-SfM-Revisited">Deep SfM Revisited</a>
    </td>
    <td style="white-space: pre;">──┐</td>
    <td></td>
  </tr>
  <tr>
    <td align="left">
      <a href="https://github.com/facebookresearch/PoseDiffusion">PoseDiffusion</a>
    </td>
    <td style="white-space: pre;">─────►</td>
    <td>
      <a href="https://github.com/facebookresearch/vggsfm">VGGSfM</a> ──►
      <a href="https://github.com/facebookresearch/vggt">VGGT</a>
    </td>
  </tr>
  <tr>
    <td align="left">
      <a href="https://github.com/facebookresearch/co-tracker">CoTracker</a>
    </td>
    <td style="white-space: pre;">──┘</td>
    <td></td>
  </tr>
</table>


## Acknowledgements

Thanks to these great repositories: [PoseDiffusion](https://github.com/facebookresearch/PoseDiffusion), [VGGSfM](https://github.com/facebookresearch/vggsfm), [CoTracker](https://github.com/facebookresearch/co-tracker), [DINOv2](https://github.com/facebookresearch/dinov2), [Dust3r](https://github.com/naver/dust3r), [Moge](https://github.com/microsoft/moge), [PyTorch3D](https://github.com/facebookresearch/pytorch3d), [Sky Segmentation](https://github.com/xiongzhu666/Sky-Segmentation-and-Post-processing), [Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2), [Metric3D](https://github.com/YvanYin/Metric3D) and many other inspiring works in the community.

## Checklist

- [x] Release the training code
- [ ] Release VGGT-500M and VGGT-200M


## License
See the [LICENSE](./LICENSE.txt) file for details about the license under which this code is made available.

Please note that only this [model checkpoint](https://huggingface.co/facebook/VGGT-1B-Commercial) allows commercial usage. This new checkpoint achieves the same performance level (might be slightly better) as the original one, e.g., AUC@30: 90.37 vs. 89.98 on the Co3D dataset.
>>>>>>> 44b3afbd1869d8bde4894dd8ea1e293112dd5eba
