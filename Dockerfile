# Use NVIDIA CUDA base image for GPU support
FROM nvidia/cuda:12.1.0-runtime-ubuntu22.04

# Set environment variables
ENV DEBIAN_FRONTEND=noninteractive
ENV PIXI_HOME=/root/.pixi
ENV PATH="${PIXI_HOME}/bin:${PATH}"
ENV PYTHONUNBUFFERED=1

# Install system dependencies
RUN apt-get update && apt-get install -y \
    curl \
    ca-certificates \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install Pixi
RUN curl -fsSL https://pixi.sh/install.sh | bash

# Set working directory
WORKDIR /app

# Copy project files
COPY pyproject.toml ./
COPY requirements.txt ./
COPY vggt/ ./vggt/
COPY visual_util.py ./
COPY demo_gradio.py ./
COPY vggt_process.py ./

# Install dependencies using Pixi
RUN pixi install -e demo

# Pre-download the VGGT model (this saves ~4.68GB download time on each run)
# This creates a simple script to trigger the model download
RUN echo "import torch\n\
print('Pre-downloading VGGT model...')\n\
_URL = 'https://huggingface.co/facebook/VGGT-1B/resolve/main/model.pt'\n\
state_dict = torch.hub.load_state_dict_from_url(_URL, progress=True)\n\
print(f'Model downloaded successfully to cache')\n\
print(f'Model size: {sum(p.numel() for p in state_dict.values()) / 1e9:.2f}B parameters')" > /tmp/download_model.py \
    && pixi run -e demo python /tmp/download_model.py \
    && rm /tmp/download_model.py

# Expose Gradio default port
EXPOSE 7860

# Set the entrypoint
ENTRYPOINT ["pixi", "run", "-e", "demo"]

# Default command runs the demo with local-only access
# Override with: docker run ... demo_gradio.py --share to enable public URL
CMD ["python", "demo_gradio.py"]

