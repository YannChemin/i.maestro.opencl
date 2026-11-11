/* SPDX-License-Identifier: Unlicense */

#ifndef I_MAESTRO_OPENCL_MAESTRO_MODEL_H
#define I_MAESTRO_OPENCL_MAESTRO_MODEL_H

#include "ocl_backend.h"
#include "pth_loader.h"

#define MAESTRO_MAX_MODS 8
#define MAESTRO_MAX_GROUPS 8
#define MAESTRO_EMBED_DIM 768
#define MAESTRO_N_HEADS 12
#define MAESTRO_DIM_HEAD 64

/* One acquisition date of one date slot, as fed to the date encoding:
 * calendar year, day of year (1..366) and hour of day (0..23). */
struct maestro_date {
    int year, doy, hour;
};

/* One input modality: the checkpoint's fixed geometry plus its runtime
 * number of date slots and the device buffers holding its weights,
 * encoding and activations. */
struct maestro_mod {
    const char *name; /* modality name, also the GRASS option name */
    const char *group; /* encoder group it feeds (s1_asc -> s1) */
    int image, patch, bands, num_dates; /* checkpoint shape spec */
    double norm_fac;
    int log_scale, rescale_elev;
    /* Derived: grid = image / patch, K = bands * patch^2, L = grid^2. */
    int grid, K, L;
    int ndates; /* runtime date slots, 1..num_dates */
    cl_mem conv_w, conv_b, gn_w, gn_b, enc;
    cl_mem X, emb; /* (B, D*L, 768) activations, (B, L, 768) output */
};

/* One pre-norm Transformer block (vit_pytorch): attention half then
 * feed-forward half. */
struct maestro_layer {
    cl_mem ln1_w, ln1_b; /* layers.N.0.norm */
    cl_mem qkv_w;        /* layers.N.0.to_qkv.weight, no bias */
    cl_mem out_w, out_b; /* layers.N.0.to_out.0 */
    cl_mem ln2_w, ln2_b; /* layers.N.1.net.0 */
    cl_mem fc1_w, fc1_b; /* layers.N.1.net.1 */
    cl_mem fc2_w, fc2_b; /* layers.N.1.net.4 */
};

/* A Transformer stack over one group's fused sequence, or over the
 * concatenation of all groups' encodings (encoder_inter). */
struct maestro_encoder {
    char group[32];
    int depth;
    int T; /* sequence length (B-independent) */
    struct maestro_layer *layers;
    cl_mem norm_w, norm_b; /* final LayerNorm */
    cl_mem Xg;             /* (B, T, 768) group sequence */
};

/* A released checkpoint: its download coordinates and the geometry of
 * every modality, in the checkpoint's own modality order (which is also
 * the order of the modality encodings in the group fusion). */
struct maestro_variant {
    const char *name;  /* model= option value */
    const char *repo;  /* Hugging Face repo, NULL when unpublished */
    const char *file;  /* path inside the repo */
    long long size;    /* byte size of the checkpoint */
    int published;
    double crop_meters; /* grid cell size = crop_meters / grid_pos_enc */
    int grid_pos_enc;   /* positional encoding grid side */
    int nmods;
    struct maestro_mod mods[MAESTRO_MAX_MODS];
};

struct maestro_model {
    struct ocl_backend *ocl;
    const struct maestro_variant *variant;
    struct pth_file pf;
    int B; /* tiles per encode call */
    struct maestro_mod mods[MAESTRO_MAX_MODS];
    int nmods, ngroups;
    struct maestro_encoder encs[MAESTRO_MAX_GROUPS + 1]; /* [ngroups] = inter */
    int T_inter;
    cl_kernel k_gemm, k_gemm_big, k_gemm_t, k_gemm_n;
    cl_kernel k_gemm_h, k_gemm_big_h, k_gemm_t_h, k_gemm_n_h;
    cl_kernel k_ln, k_softmax, k_im2col, k_add, k_place, k_take, k_mean;
    cl_mem img, im2col, Xln, qkv, scores, attn, ffn, inter;
};

/* Variant registry. */
const struct maestro_variant *maestro_variant_find(const char *name);
void maestro_variant_list(void);

/* Open the checkpoint and upload its weights, allocate the activations
 * for `batch` tiles and create the kernels. ndates[i] is the number of
 * date slots read for mods[i] (1..num_dates). */
void maestro_model_init(struct maestro_model *m, struct ocl_backend *ocl,
                        const struct maestro_variant *v, const char *ckpt,
                        const int *ndates, int batch);

/* Compute and upload the positional + date encoding of every modality.
 * dates[i] holds ndates[i] acquisition dates, ref the reference date
 * (first modality's first slot). */
void maestro_set_dates(struct maestro_model *m, const struct maestro_date **dates,
                       const struct maestro_date *ref);

/* Encode one batch of tiles: imgs[i] is (B, ndates[i], bands, image,
 * image) channels-first float data, in the modality's band order;
 * embs[i] receives (B, L, 768). */
void maestro_encode(struct maestro_model *m, const float *const *imgs,
                    float *const *embs);

void maestro_free(struct maestro_model *m);

#endif /* I_MAESTRO_OPENCL_MAESTRO_MODEL_H */
