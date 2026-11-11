# i.maestro.opencl

A GRASS GIS addon that applies the MAESTRO multimodal masked-autoencoder foundation model encoder using OpenCL.

## Overview

**i.maestro.opencl** is a GRASS GIS module that implements the MAESTRO (Multimodal masked-Autoencoder for EO TRansformers) encoder model in C + OpenCL. It reads original PyTorch checkpoints and produces per-modality embedding raster maps at the model's patch grid resolution.

MAESTRO is a foundation model for Earth Observation developed by IGNF (French National Geographic Institute). This addon provides a faithful translation of the encoder forward pass, enabling efficient inference on GPUs and CPUs without requiring Python or PyTorch.

## Features

- **Multi-modal support**: Processes aerial, SPOT, DEM, Sentinel-2, and Sentinel-1 imagery
- **Hardware acceleration**: Runs on GPUs and CPUs via OpenCL
- **On-demand weights**: Automatically downloads checkpoints from Hugging Face Hub
- **Tiled processing**: Handles large regions by processing in tiles
- **Flexible output**: Select specific embedding channels or output all 768 dimensions

## Model Variants

| Variant | Repository | Size | Status |
|---------|------------|------|--------|
| flair-hub | IGNF/MAESTRO_FLAIR-HUB_base | 1.4 GB | Published |
| s2-naip | IGNF/MAESTRO_S2-NAIP-urban_base | 1.1 GB | Published |
| forest | IGNF/MAESTRO_FLAIR-HUB_forest | TBD | Unpublished |
| agriculture | IGNF/MAESTRO_FLAIR-HUB_agriculture | TBD | Unpublished |
| urban | IGNF/MAESTRO_S2-NAIP-urban | TBD | Unpublished |

## Installation

### Prerequisites

- GRASS GIS 8.x (tested with 8.6)
- OpenCL implementation (NVIDIA, AMD, Intel, or PoCL for CPU)
- libcurl (for weight downloads)
- Standard C compilation toolchain

### Build

```bash
cd i.maestro.opencl
make MODULE_TOPDIR=/path/to/grass/source
export LD_LIBRARY_PATH=/path/to/grass/lib:$LD_LIBRARY_PATH
make install
```

Or use the GRASS addon system:
```bash
cd ~/grass_database/location/mapset
g.extension i.maestro.opencl url=https://github.com/YannChemin/i.maestro.opencl.git
```

### Build in detached mode (recommended for long builds)

```bash
cd /home/yann/dev/i.maestro.opencl
nohup make MODULE_TOPDIR=$HOME/dev/grass > build.log 2>&1 &
echo $! > build.pid
# Monitor progress
tail -f build.log
# Or check status
kill -0 $(cat build.pid) && echo "Still building..." || echo "Build complete or failed"
```

## Usage

### Quick start

```bash
# Set up GRASS environment
gis_set
# Set region to match your data
g.region raster=ortho_2022 -p

# Run MAESTRO encoder
i.maestro.opencl aerial=ortho_2022 output=maestro_emb
```

### Using multiple modalities

```bash
i.maestro.opencl \
  aerial=ortho_2022 \
  s2=B02,B03,B04,B08,B11,B12 \
  s1=VV,VH \
  output=maestro_emb \
  model=s2-naip \
  channels=1-32
```

### Weight management

By default, checkpoints are downloaded to `~/models/maestro/` or `$MAESTRO_MODELS_DIR/maestro/`.

```bash
# Use custom weights directory
i.maestro.opencl aerial=ortho output=emb weights_dir=/data/models/maestro

# Disable automatic download
i.maestro.opencl aerial=ortho output=emb -n

# Use explicit checkpoint
i.maestro.opencl aerial=ortho output=emb \
  checkpoint=/data/checkpoints/MAESTRO_FLAIR-HUB_base/checkpoints/pretrain-epoch=99.ckpt
```

### Device selection

```bash
# Use GPU
i.maestro.opencl aerial=ortho output=emb device=gpu

# Use CPU with half precision
i.maestro.opencl aerial=ortho output=emb device=cpu precision=fp16

# Select specific platform
i.maestro.opencl aerial=ortho output=emb platform=NVIDIA
```

## Architecture

The addon consists of the following components:

| File | Purpose |
|------|---------|
| `main.c` | GRASS module interface, option parsing, tile loop |
| `maestro_model.c/h` | Model definition, weight loading, forward pass |
| `maestro_kernels.cl` | OpenCL kernels (GEMM, LayerNorm, Softmax, etc.) |
| `ocl_backend.c/h` | OpenCL device management, buffer allocation |
| `pth_loader.c/h` | PyTorch checkpoint parser (pickle, tensors) |
| `download.c/h` | Weight download from Hugging Face Hub |
| `progress.c/h` | Progress tracking and reporting |
| `raster_io.c/h` | Raster input/output with GRASS API |

### Key Algorithms

1. **Patch Embedding**: Conv2d with kernel=patch, stride=patch, followed by GroupNorm
2. **Positional Encoding**: 2D sincos encoding with block-mean over patch grid
3. **Date Encoding**: Time difference + cyclic day-of-year and hour encodings
4. **Transformer**: Pre-norm architecture with multi-head attention and MLP
5. **Fusion**: Group-based modality fusion with inter-encoder

## Performance

The addon is optimized for various hardware configurations:

| Device Type | Profile | Performance |
|-------------|---------|-------------|
| GPU (NVIDIA/AMD) | dgpu | Fastest |
| Integrated GPU | igpu | Fast |
| CPU (PoCL) | cpu | Moderate |
| External GPU | egpu | Slower (PCIe transfer overhead) |

### Benchmarks (estimated)

| Variant | Tile Size | GPU Time | CPU Time |
|---------|-----------|----------|----------|
| flair-hub | 102.4m | ~0.5s/tile | ~5s/tile |
| s2-naip | 120m | ~0.4s/tile | ~4s/tile |

## Testing

### Unit tests

```bash
# Run tests
python3 -m pytest tests/

# Run with specific checkpoint
I_MAESTRO_OPENCL_CHECKPOINT=/path/to/checkpoint.ckpt python3 -m pytest tests/
```

### Numerical validation

The `tests/maestro_reference.py` script provides a PyTorch reference implementation for numerical validation:

```bash
# Generate reference outputs
python3 tests/maestro_reference.py --variant flair --outdir /tmp/maestro_reference

# Compare C implementation vs reference
# (This is done automatically by the test suite)
```

## Development

### Directory Structure

```
i.maestro.opencl/
├── CMakeLists.txt          # Not yet implemented
├── LICENSE                 # Unlicense
├── Makefile                # GRASS module build
├── README_for_LLM.md      # Development notes
├── README.md               # User documentation
├── TODO.md                 # Roadmap
├── download.c/h            # Weight download
├── i.maestro.opencl.html   # HTML manual
├── i.maestro.opencl.md     # Manual
├── maestro_kernels.cl      # OpenCL kernels
├── maestro_model.c/h       # Model implementation
├── main.c                  # Main module
├── ocl_backend.c/h         # OpenCL backend
├── progress.c/h            # Progress tracking
├── pth_loader.c/h          # Checkpoint loader
├── raster_io.c/h           # Raster I/O
└── tests/                  # Test suite
    ├── conftest.py
    ├── maestro_reference.py
    └── reference_compare.py
```

### Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Update documentation
6. Submit a pull request

### Environment Variables

| Variable | Description |
|----------|-------------|
| `MAESTRO_MODELS_DIR` | Base directory for model weights |
| `I_MAESTRO_OPENCL_DOWNLOAD_URL` | Override download URL |
| `I_MAESTRO_OPENCL_PROFILE` | Enable kernel profiling |
| `I_MAESTRO_OPENCL_CHECKPOINT` | Path to checkpoint for tests |

## Known Issues and Limitations

1. **Batch processing**: Currently only batch=1 is fully implemented
2. **Date slots**: Multi-date slot parsing is not yet implemented
3. **Band selection**: Band selection within date slots needs refinement
4. **Fine-tuned variants**: Forest, agriculture, and urban variants are not yet published
5. **Memory**: Large tiles with many modalities may exceed GPU memory

## Roadmap

See [TODO.md](TODO.md) for planned features and improvements.

## License

This project is licensed under the Unlicense - see [LICENSE](LICENSE) for details.

## References

- [MAESTRO on Hugging Face](https://huggingface.co/IGNF)
- [GRASS GIS](https://grass.osgeo.org/)
- [OpenCL](https://www.khronos.org/opencl/)

## Author

Yann Chemin <dr.yann.chemin@gmail.com>

## Acknowledgments

- IGNF for developing and releasing the MAESTRO model
- The GRASS GIS development team
- Contributors to the i.sam.opencl addon which served as a template
