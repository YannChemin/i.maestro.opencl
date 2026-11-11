# i.maestro.opencl

## NAME
**i.maestro.opencl** - Applies the MAESTRO multimodal masked-autoencoder foundation model encoder on imagery groups with an OpenCL device and outputs per-modality embedding rasters.

## SYNOPSIS

**i.maestro.opencl**

[**aerial**=*name*[*band*,...][;*name*[*band*,...]]]
[**spot**=*name*[*band*,...][;*name*[*band*,...]]]
[**dem**=*name*[*band*,...][;*name*[*band*,...]]]
[**s2**=*name*[*band*,...][;*name*[*band*,...]]]
[**s1**=*name*[*band*,...][;*name*[*band*,...]]]
[**s1_asc**=*name*[*band*,...][;*name*[*band*,...]]]
[**s1_des**=*name*[*band*,...][;*name*[*band*,...]]]
[**model**=*string*]
[**weights_dir**=*string*]
[**checkpoint**=*string*]
[**tile_meters**=*float*]
[**batch**=*integer*]
[**channels**=*string*]
[**dates**=*string*]
**output**=*name*
[**device**=*string*]
[**platform**=*string*]
[**profile**=*string*]
[**precision**=*string*]
[**-n**]

## DESCRIPTION

*i.maestro.opencl* applies the MAESTRO (Multimodal masked-Autoencoder for EO TRansformers) encoder model to imagery inputs on a GPU or CPU using OpenCL. The model is a faithful C translation of the original PyTorch implementation, using the released checkpoints from IGNF.

MAESTRO is a multimodal foundation model pretrained on a large collection of aerial, satellite, and elevation data. This addon implements the encoder part of the model, which can be used to extract meaningful feature embeddings from input imagery.

The addon supports multiple input modalities:
- **aerial**: Aerial imagery (typically 4 bands: NIR, R, G, B)
- **spot**: SPOT imagery (typically 3 bands: RGB)
- **dem**: Digital Elevation Model (elevation and surface channels)
- **s2**: Sentinel-2 imagery (10 bands)
- **s1**, **s1_asc**, **s1_des**: Sentinel-1 SAR imagery (VV, VH bands)

Each modality can have multiple date slots (separated by semicolons), and within each slot, multiple bands can be specified (separated by commas).

## PARAMETERS

### Input Options

**aerial**, **spot**, **dem**, **s2**, **s1**, **s1_asc**, **s1_des**

Input raster map(s) for each modality. For each option:
- Bands within a date slot are separated by commas
- Date slots are separated by semicolons
- Example: `aerial=ortho_R,ortho_G,ortho_B,ortho_NIR` for single date
- Example: `s2=B02,B03,B04;B02b,B03b,B04b` for two dates

At least one input modality must be specified.

### Model Options

**model**=*string*

MAESTRO model variant to use. Options: **flair-hub**, **s2-naip**, **forest**, **agriculture**, **urban**
- Default: **flair-hub**

The FLAIR-HUB and S2-NAIP variants are published and will be downloaded automatically. The forest, agriculture, and urban variants are announced but not yet published; specifying them will result in an error unless a checkpoint is provided explicitly.

**weights_dir**=*string*

Directory containing model checkpoints. If not specified, the addon will use:
- `$MAESTRO_MODELS_DIR/maestro` if the environment variable is set
- Otherwise `~/models/maestro`

Checkpoints are organized as `<weights_dir>/<variant>/<checkpoint_file>.ckpt`

**checkpoint**=*string*

Explicit path to a MAESTRO checkpoint file (.ckpt). If specified, this overrides both **model** and **weights_dir**. The checkpoint must be compatible with the selected model variant.

**-n**

Do not download missing checkpoints. If the checkpoint for the selected model variant is not found in the weights directory and this flag is set, the addon will exit with an error instead of downloading the checkpoint.

### Processing Options

**tile_meters**=*float*

Tile size in meters for processing. If set to 0 (default), the addon will use the model's native crop_meters value (102.4m for flair-hub, 120m for s2-naip).

Larger tiles reduce overhead but require more memory. The tile size should be chosen based on available GPU memory.

**batch**=*integer*

Number of tiles to process at once. Currently only batch=1 is fully supported.

Default: 1

### Output Options

**output**=*name*

Prefix for output raster maps. For each modality and each output channel, a raster map will be created with the suffix `_<modality>.<channel>`.

For example, if output=maestro_emb, channels=1-8, and aerial modality is used, the following maps will be created:
- `maestro_emb_aerial.1`
- `maestro_emb_aerial.2`
- ...
- `maestro_emb_aerial.8`

**channels**=*string*

Output channels to write. Options:
- A range like `1-8` (default)
- A comma-separated list like `1,3,5,7`
- `all` for all 768 channels

The embeddings are 768-dimensional. Writing all channels creates many raster maps and may be memory-intensive.

**dates**=*string*

Comma-separated list of ISO dates (YYYY-MM-DD[THH]) to use instead of the timestamps stored in the input raster maps.

If fewer dates are provided than the total number of date slots across all modalities, the last date will be reused for the remaining slots.

If not specified, the addon will attempt to read timestamps from the input raster maps using `r.timestamp`. If a map has no timestamp, a default date of 2022-07-01 will be used with a warning.

### Device Options

**device**=*string*

OpenCL device type to use. Options: **auto**, **gpu**, **cpu**
- **auto**: Automatically select the best available device
- **gpu**: Use GPU device(s)
- **cpu**: Use CPU device(s)

Default: auto

**platform**=*string*

Use only OpenCL platforms whose name contains this text. This can be used to select a specific OpenCL implementation (e.g., "NVIDIA", "Intel", "AMD", "PoCL").

If not specified, all available platforms will be considered.

**profile**=*string*

Device profile for buffer management and kernel selection. Options: **auto**, **cpu**, **igpu**, **dgpu**, **egpu**
- **auto**: Detect profile from the device
- **cpu**: Host-memory device (PoCL): mapped buffers, CPU kernels
- **igpu**: GPU sharing host memory: mapped transfers
- **dgpu**: GPU with its own memory on internal PCIe
- **egpu**: GPU behind a slow link (Thunderbolt, remote): pinned staging

Default: auto

**precision**=*string*

Weight storage precision. Options: **fp32**, **fp16**
- **fp32**: Single precision (full precision, more memory)
- **fp16**: Half precision storage, single precision arithmetic (less memory, same compute precision)

Default: fp32

## NOTES

### Weights Download

The addon can automatically download MAESTRO checkpoints from Hugging Face Hub. Checkpoints are downloaded to the weights directory specified by **weights_dir** or the default location.

Downloaded checkpoints:
- **flair-hub**: IGNF/MAESTRO_FLAIR-HUB_base (1.4 GB)
- **s2-naip**: IGNF/MAESTRO_S2-NAIP-urban_base (1.1 GB)

The download URL can be overridden by setting the environment variable `I_MAESTRO_OPENCL_DOWNLOAD_URL`.

### Model Architecture

MAESTRO uses a vision transformer architecture with multimodal fusion:
1. Each modality is processed independently through a patch embedding (Conv2d + GroupNorm)
2. Positional and date encodings are added
3. Modalities are grouped (e.g., S1 ascending and descending share the same encoder group)
4. Each group's tokens are processed through 9 Transformer layers
5. All groups are concatenated and processed through 3 additional Transformer layers (encoder_inter)
6. The output is split back per modality and per-patch embeddings are averaged over dates

The released checkpoints contain only the encoder; there is no trained decoder for reconstruction.

### Preprocessing

Each input modality is preprocessed as follows:
1. **Normalization**: Values are divided by a modality-specific normalization factor (norm_fac)
2. **Log scaling**: For some modalities (e.g., Sentinel-1 in flair-hub), values are log-scaled: `log(max(x, 1e-10))`
3. **DEM rescaling**: For the DEM modality, the elevation channel is rescaled: `elevation = 30 * (elevation - surface)`
4. **Nearest neighbor resampling**: Inputs are resampled to the model's expected image size using nearest neighbor interpolation
5. **NULL handling**: NULL cells in input rasters are treated as 0 after normalization

### Date Encoding

MAESTRO uses a learned date encoding that includes:
- Time difference from a reference date (multiplied by 4)
- Cyclic encoding of day of year (sin and cos)
- Cyclic encoding of hour of day (sin and cos)

The reference date is the date of the first modality's first date slot (typically the aerial imagery).

### Output Resolution

The output raster maps have the same spatial resolution as the model's patch grid. For the flair-hub variant:
- aerial: 512x512 image with 16x16 patches = 32x32 grid
- spot: 128x128 image with 16x16 patches = 8x8 grid
- dem: 512x512 image with 32x32 patches = 16x16 grid
- s2: 10x10 image with 2x2 patches = 5x5 grid
- s1_asc/s1_des: 10x10 image with 2x2 patches = 5x5 grid

The output is written at the native patch grid resolution of each modality.

### Memory Requirements

The model requires significant memory:
- Weights: ~1.4 GB (fp32) or ~0.7 GB (fp16) for flair-hub
- Activations: Varies based on tile size and number of modalities

For a typical scene with all modalities enabled, expect to need 4-8 GB of GPU memory for fp32 weights, or 2-4 GB for fp16 weights.

### Environment Variables

- `MAESTRO_MODELS_DIR`: Base directory for model weights (default: ~/models)
- `I_MAESTRO_OPENCL_DOWNLOAD_URL`: Override the download URL for checkpoints
- `I_MAESTRO_OPENCL_PROFILE`: Enable kernel profiling (prints timing information)
- `I_MAESTRO_OPENCL_CHECKPOINT`: For testing, use a specific checkpoint path

## EXAMPLES

### Basic usage with aerial imagery

```bash
aerial=ortho_2022
g.region raster=ortho_2022 -p
i.maestro.opencl aerial=$aerial output=maestro_emb model=flair-hub
```

### Multi-modal with aerial, Sentinel-2, and Sentinel-1

```bash
i.maestro.opencl \
  aerial=ortho_2022 \
  s2=B02,B03,B04,B08,B11,B12 \
  s1=VV,VH \
  output=maestro_emb \
  model=s2-naip \
  channels=1-32
```

### Using custom tile size

```bash
i.maestro.opencl \
  aerial=ortho_2022 \
  output=maestro_emb \
  tile_meters=240 \
  device=gpu
```

### Specifying explicit checkpoint

```bash
i.maestro.opencl \
  aerial=ortho_2022 \
  output=maestro_emb \
  checkpoint=/path/to/MAESTRO_FLAIR-HUB_base/checkpoints/pretrain-epoch=99.ckpt
```

### Using CPU device with half precision

```bash
i.maestro.opencl \
  aerial=ortho_2022 \
  output=maestro_emb \
  device=cpu \
  precision=fp16
```

## REFERENCES

- MAESTRO model: https://huggingface.co/IGNF
- Paper: [MAESTRO: Masked Autoencoder for Multimodal Remote Sensing Foundation Model](https://arxiv.org/abs/...)
- GRASS GIS: https://grass.osgeo.org/

## AUTHOR

Yann Chemin <dr.yann.chemin@gmail.com>

## SEE ALSO

[i.sam.opencl](i.sam.opencl.html), [r.timestamp](r.timestamp.html)
