# VGGT Docker Guide

## Quick Reference

### Build
```bash
./docker-build.sh
```

### Run
```bash
./docker-run.sh
```

### Access
Open browser to: **http://localhost:7860**

---

## Detailed Commands

### Building the Image

**Standard build:**
```bash
docker build -t vggt:latest .
```

**Build with no cache (clean rebuild):**
```bash
docker build --no-cache -t vggt:latest .
```

**Build with custom tag:**
```bash
docker build -t vggt:v1.0 .
```

### Running the Container

**Interactive mode (see console output):**
```bash
docker run -it --rm --gpus all -p 7860:7860 vggt:latest
```

**Detached mode (background):**
```bash
docker run -d --name vggt-demo --gpus all -p 7860:7860 vggt:latest
```

**CPU-only (no GPU):**
```bash
docker run -it --rm -p 7860:7860 vggt:latest
```

**Custom port:**
```bash
docker run -it --rm --gpus all -p 8080:7860 vggt:latest
# Access at http://localhost:8080
```

**With volume mount for outputs:**
```bash
docker run -it --rm --gpus all \
    -p 7860:7860 \
    -v $(pwd)/outputs:/app/outputs \
    vggt:latest
```

### Docker Compose

**Start:**
```bash
docker-compose up
```

**Start in background:**
```bash
docker-compose up -d
```

**Stop:**
```bash
docker-compose down
```

**View logs:**
```bash
docker-compose logs -f
```

**Rebuild and start:**
```bash
docker-compose up --build
```

### Managing Containers

**List running containers:**
```bash
docker ps
```

**Stop a container:**
```bash
docker stop vggt-demo
```

**Remove a container:**
```bash
docker rm vggt-demo
```

**View logs:**
```bash
docker logs vggt-demo
docker logs -f vggt-demo  # Follow mode
```

**Execute command in running container:**
```bash
docker exec -it vggt-demo /bin/bash
```

**Check resource usage:**
```bash
docker stats vggt-demo
```

### Managing Images

**List images:**
```bash
docker images | grep vggt
```

**Remove image:**
```bash
docker rmi vggt:latest
```

**Tag image:**
```bash
docker tag vggt:latest vggt:v1.0
```

**Export image:**
```bash
docker save vggt:latest | gzip > vggt-latest.tar.gz
```

**Import image:**
```bash
docker load < vggt-latest.tar.gz
```

---

## Image Details

### What's Included

- **Base Image**: NVIDIA CUDA 12.1.0 on Ubuntu 22.04
- **Python**: 3.12.x (via Pixi)
- **PyTorch**: 2.3.1 with CUDA support
- **VGGT Model**: Pre-downloaded (4.68 GB)
- **All Dependencies**: Installed and ready

### Image Size

- Approximately **12-15 GB** (includes model cache)
- Model alone is ~4.68 GB
- Environment and dependencies ~7-10 GB

### Layers

1. CUDA runtime base
2. System dependencies
3. Pixi installation
4. Project files
5. Pixi environment
6. Model download (cached)

---

## Advanced Usage

### Custom Entry Point

**Run Python interpreter:**
```bash
docker run -it --rm vggt:latest python
```

**Run bash shell:**
```bash
docker run -it --rm vggt:latest /bin/bash
```

**Run custom script:**
```bash
docker run -it --rm --gpus all \
    -v $(pwd)/my_script.py:/app/my_script.py \
    vggt:latest python my_script.py
```

### Environment Variables

**Set environment variables:**
```bash
docker run -it --rm --gpus all \
    -e CUDA_VISIBLE_DEVICES=0 \
    -e GRADIO_SERVER_NAME=0.0.0.0 \
    -p 7860:7860 \
    vggt:latest
```

### Multi-GPU Setup

**Use specific GPU:**
```bash
docker run -it --rm --gpus '"device=0"' -p 7860:7860 vggt:latest
```

**Use multiple GPUs:**
```bash
docker run -it --rm --gpus '"device=0,1"' -p 7860:7860 vggt:latest
```

### Persistent Storage

**Mount data directory:**
```bash
docker run -it --rm --gpus all \
    -p 7860:7860 \
    -v /path/to/data:/app/data \
    -v /path/to/outputs:/app/outputs \
    vggt:latest
```

---

## Production Deployment

### Docker Compose for Production

Create `docker-compose.prod.yml`:

```yaml
version: '3.8'

services:
  vggt:
    image: vggt:latest
    container_name: vggt-prod
    restart: always
    ports:
      - "7860:7860"
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
        limits:
          memory: 16G
    volumes:
      - ./outputs:/app/outputs
      - ./logs:/app/logs
    logging:
      driver: "json-file"
      options:
        max-size: "10m"
        max-file: "3"
```

Run with:
```bash
docker-compose -f docker-compose.prod.yml up -d
```

### Health Check

Add to Dockerfile or docker-compose.yml:

```yaml
healthcheck:
  test: ["CMD", "curl", "-f", "http://localhost:7860/"]
  interval: 30s
  timeout: 10s
  retries: 3
  start_period: 60s
```

### Reverse Proxy (Nginx)

Example nginx configuration:

```nginx
server {
    listen 80;
    server_name vggt.example.com;

    location / {
        proxy_pass http://localhost:7860;
        proxy_http_version 1.1;
        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

---

## Troubleshooting

### Container Exits Immediately

View logs:
```bash
docker logs vggt-demo
```

Run interactively to debug:
```bash
docker run -it vggt:latest /bin/bash
```

### Out of Memory

Limit memory or reduce model usage:
```bash
docker run -it --rm --gpus all \
    --memory="16g" \
    --memory-swap="16g" \
    -p 7860:7860 \
    vggt:latest
```

### GPU Not Working

Check NVIDIA runtime:
```bash
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

If fails, reinstall nvidia-container-toolkit.

### Permission Issues

Run as specific user:
```bash
docker run -it --rm --gpus all \
    --user $(id -u):$(id -g) \
    -p 7860:7860 \
    vggt:latest
```

---

## Best Practices

1. **Use docker-compose** for consistent deployments
2. **Mount volumes** for persistent outputs
3. **Set resource limits** to prevent OOM
4. **Use specific tags** (not `latest`) in production
5. **Monitor logs** regularly
6. **Backup model cache** if rebuilding frequently
7. **Use health checks** for production deployments

---

## Questions?

See main [README.md](README.md) for more information.

