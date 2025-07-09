# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**MatAnyone** is a stable video matting framework with consistent memory propagation. It's a deep learning project focused on human video matting - extracting foreground subjects (particularly humans) from video backgrounds with high accuracy on both semantic regions and fine-grained boundaries.

### Key Features
- Target assignment support (can handle multiple subjects)
- Stable performance in both core regions and fine-grained boundary details
- Interactive segmentation capabilities
- Support for both image and video matting
- Cross-platform support (CUDA GPUs and Apple Silicon MPS)

## Architecture Overview

### Core Components
- **`/matanyone`**: Main Python package containing the neural network implementation
  - `/config`: Hydra configuration files for model settings
  - `/inference`: Inference pipeline and utilities
  - `/model`: Neural network architecture (memory-based video matting)
  - `/utils`: Helper functions and utilities
- **`/pretrained_models`**: Pre-trained model weights storage

### Technology Stack
- **Deep Learning**: PyTorch with CUDA/MPS support
- **Configuration**: Hydra for configuration management
- **Video I/O**: imageio with ffmpeg backend
- **Segmentation**: SAM (Segment Anything Model) integration

## Essential Commands

### Installation
```bash
# Create conda environment
conda create -n matanyone python=3.8 -y
conda activate matanyone

# Install package
pip install -e .

```

### Running Inference
```bash
# Single target matting
python inference_matanyone.py -i inputs/video/test-sample1.mp4 -m inputs/mask/test-sample1.png

# Multiple targets (using different masks)
python inference_matanyone.py -i inputs/video/test-sample0 -m inputs/mask/test-sample0_1.png --suffix target1
python inference_matanyone.py -i inputs/video/test-sample0 -m inputs/mask/test-sample0_2.png --suffix target2

# With resolution limit
python inference_matanyone.py -i video.mp4 -m mask.png --max_size 1080

# Save per-frame images
python inference_matanyone.py -i video.mp4 -m mask.png --save_image
```

## Input/Output Format

### Inputs
- **Video**: MP4, MOV, AVI formats or folder containing frame sequences
- **Mask**: PNG image with first-frame segmentation (can be generated using SAM2)

### Outputs
- **Foreground video**: RGB video with matted subject
- **Alpha video**: Grayscale alpha matte video
- Optional: Per-frame PNG images when using `--save_image`

## Model Configuration

The model uses Hydra for configuration management. Key configuration files:
- `matanyone/config/eval_matanyone_config.yaml`: Main evaluation config
- Model weights are automatically downloaded on first run or can be manually placed in `pretrained_models/matanyone.pth`

## Development Notes

### Platform Support
- CUDA GPUs: Primary target platform
- Apple Silicon (MPS): Supported with automatic device detection
- CPU: Fallback option (slower performance)

### Memory Considerations
- Video resolution impacts memory usage significantly
- Use `--max_size` parameter to limit resolution for large videos
- The model uses memory propagation, so longer videos require more memory

### Key Dependencies
- PyTorch (with appropriate CUDA/MPS support)
- Hydra for configuration
- imageio[ffmpeg] for video I/O
- SAM (Segment Anything Model) for segmentation