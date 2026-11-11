/****************************************************************************
 *
 * MODULE:       i.maestro.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      MAESTRO multimodal encoder forward pass on OpenCL, weights
 *               read directly from the original PyTorch checkpoint.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include <grass/gis.h>
#include <grass/glocale.h>

#include "maestro_model.h"
#include "progress.h"

#define ACT_NONE 0
#define ACT_GELU 1
#define ACT_RELU 2

#define LN_EPS 1e-5f /* nn.LayerNorm() and nn.GroupNorm() default. */

#define EMB MAESTRO_EMBED_DIM
#define QKV (3 * EMB)
#define MLP (4 * EMB)

/* Checkpoint registry. The fine-tuned variants are announced but not
 * published: their geometry mirrors the FLAIR-HUB base. */
static const struct maestro_variant variants[] = {
    {
        .name = "flair-hub",
        .repo = "IGNF/MAESTRO_FLAIR-HUB_base",
        .file = "MAESTRO_FLAIR-HUB_base/checkpoints/pretrain-epoch=99.ckpt",
        .size = 1397372182LL,
        .published = 1,
        .crop_meters = 102.4,
        .grid_pos_enc = 160,
        .nmods = 6,
        .mods =
            {
                {"aerial", "aerial", 512, 16, 4, 1, 255.0, 0, 0},
                {"spot", "spot", 128, 16, 3, 1, 2000.0, 0, 0},
                {"dem", "dem", 512, 32, 2, 1, 1000.0, 0, 1},
                {"s2", "s2", 10, 2, 10, 16, 5000.0, 0, 0},
                {"s1_asc", "s1", 10, 2, 2, 4, 5.0, 1, 0},
                {"s1_des", "s1", 10, 2, 2, 4, 5.0, 1, 0},
            },
    },
    {
        .name = "s2-naip",
        .repo = "IGNF/MAESTRO_S2-NAIP-urban_base",
        .file = "MAESTRO_S2-NAIP-urban_base/checkpoints/pretrain-epoch=14.ckpt",
        .size = 1115115148LL,
        .published = 1,
        .crop_meters = 120.0,
        .grid_pos_enc = 192,
        .nmods = 4,
        .mods =
            {
                {"aerial", "aerial", 384, 16, 4, 1, 255.0, 0, 0},
                {"spot", "spot", 128, 16, 3, 1, 255.0, 0, 0},
                {"s2", "s2", 12, 2, 10, 16, 5000.0, 0, 0},
                {"s1", "s1", 12, 2, 2, 4, 20.0, 0, 0},
            },
    },
    {
        .name = "forest",
        .repo = "IGNF/MAESTRO_FLAIR-HUB_forest",
        .file = "MAESTRO_FLAIR-HUB_forest/checkpoints/pretrain-epoch=99.ckpt",
        .size = 0,
        .published = 0,
        .crop_meters = 102.4,
        .grid_pos_enc = 160,
        .nmods = 6,
        .mods =
            {
                {"aerial", "aerial", 512, 16, 4, 1, 255.0, 0, 0},
                {"spot", "spot", 128, 16, 3, 1, 2000.0, 0, 0},
                {"dem", "dem", 512, 32, 2, 1, 1000.0, 0, 1},
                {"s2", "s2", 10, 2, 10, 16, 5000.0, 0, 0},
                {"s1_asc", "s1", 10, 2, 2, 4, 5.0, 1, 0},
                {"s1_des", "s1", 10, 2, 2, 4, 5.0, 1, 0},
            },
    },
    {
        .name = "agriculture",
        .repo = "IGNF/MAESTRO_FLAIR-HUB_agriculture",
        .file = "MAESTRO_FLAIR-HUB_agriculture/checkpoints/pretrain-epoch=99.ckpt",
        .size = 0,
        .published = 0,
        .crop_meters = 102.4,
        .grid_pos_enc = 160,
        .nmods = 6,
        .mods =
            {
                {"aerial", "aerial", 512, 16, 4, 1, 255.0, 0, 0},
                {"spot", "spot", 128, 16, 3, 1, 2000.0, 0, 0},
                {"dem", "dem", 512, 32, 2, 1, 1000.0, 0, 1},
                {"s2", "s2", 10, 2, 10, 16, 5000.0, 0, 0},
                {"s1_asc", "s1", 10, 2, 2, 4, 5.0, 1, 0},
                {"s1_des", "s1", 10, 2, 2, 4, 5.0, 1, 0},
            },
    },
    {
        .name = "urban",
        .repo = "IGNF/MAESTRO_FLAIR-HUB_urban",
        .file = "MAESTRO_FLAIR-HUB_urban/checkpoints/pretrain-epoch=99.ckpt",
        .size = 0,
        .published = 0,
        .crop_meters = 102.4,
        .grid_pos_enc = 160,
        .nmods = 6,
        .mods =
            {
                {"aerial", "aerial", 512, 16, 4, 1, 255.0, 0, 0},
                {"spot", "spot", 128, 16, 3, 1, 2000.0, 0, 0},
                {"dem", "dem", 512, 32, 2, 1, 1000.0, 0, 1},
                {"s2", "s2", 10, 2, 10, 16, 5000.0, 0, 0},
                {"s1_asc", "s1", 10, 2, 2, 4, 5.0, 1, 0},
                {"s1_des", "s1", 10, 2, 2, 4, 5.0, 1, 0},
            },
    },
};

const struct maestro_variant *maestro_variant_find(const char *name)
{
    size_t i;

    for (i = 0; i < sizeof(variants) / sizeof(variants[0]); i++)
        if (strcmp(variants[i].name, name) == 0)
            return &variants[i];
    return NULL;
}

void maestro_variant_list(void)
{
    size_t i;

    for (i = 0; i < sizeof(variants) / sizeof(variants[0]); i++) {
        const struct maestro_variant *v = &variants[i];

        if (v->published)
            G_message("  %-12s %s (%s, %.1f GB)", v->name, v->repo, v->file,
                      v->size / 1e9);
        else
            G_message("  %-12s %s (not published)", v->name, v->repo);
    }
}

/* Weight loading. */

static size_t loaded_bytes;

static cl_mem load_dev(struct maestro_model *m, const char *name, int ndim,
                       const long *shape)
{
    const struct pth_tensor *t = pth_require(&m->pf, name, ndim, shape);
    float *h = pth_read_f32(&m->pf, t);
    size_t n = pth_numel(t) * sizeof(float);
    cl_mem b = ocl_upload(m->ocl, h, n, name);

    loaded_bytes += n;
    G_free(h);
    return b;
}

/* Linear/convolution weight matrices, used only as GEMM operand B: half
 * with precision=fp16. */
static cl_mem load_w(struct maestro_model *m, const char *name, int ndim,
                     const long *shape)
{
    const struct pth_tensor *t = pth_require(&m->pf, name, ndim, shape);
    float *h = pth_read_f32(&m->pf, t);
    size_t n = pth_numel(t);
    cl_mem b = ocl_upload_weight(m->ocl, h, n, name);

    loaded_bytes += n * (m->ocl->fp16w ? 2 : 4);
    G_free(h);
    return b;
}

/* Kernel launch helpers. */

#define SETARG(k, i, v) ocl_setarg(k, i, sizeof(v), &(v), #k)

/* Accumulated kernel times when profiling is enabled. */
static struct {
    const char *what;
    double ms, wall_ms, flops;
    long calls;
} prof[64];
static int nprof;
/* FLOPs of the next launch, set by gemm() for the profile. */
static double pending_flops;

static double now_ms(void)
{
    struct timespec ts;

    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec * 1e3 + ts.tv_nsec * 1e-6;
}

static void run(struct maestro_model *m, cl_kernel k, int dim, const size_t *g,
                const size_t *l, const char *what)
{
    cl_event ev;
    cl_ulong t0, t1;
    double w0 = 0.0, w1;
    int i;

    /* Wall time as well as event time: some drivers (rusticl on iris,
     * Mesa 25.0) return wrong event timestamps. */
    if (m->ocl->profile) {
        clFinish(m->ocl->queue);
        w0 = now_ms();
    }
    ocl_check(clEnqueueNDRangeKernel(m->ocl->queue, k, dim, NULL, g, l, 0, NULL,
                                     m->ocl->profile ? &ev : NULL),
              what);
    if (!m->ocl->profile) {
        pending_flops = 0.0;
        return;
    }
    clWaitForEvents(1, &ev);
    w1 = now_ms();
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_START, sizeof(t0), &t0,
                            NULL);
    clGetEventProfilingInfo(ev, CL_PROFILING_COMMAND_END, sizeof(t1), &t1,
                            NULL);
    clReleaseEvent(ev);
    for (i = 0; i < nprof && strcmp(prof[i].what, what) != 0; i++)
        ;
    if (i == nprof && nprof < 64)
        prof[nprof++].what = what;
    if (i < 64) {
        prof[i].ms += (t1 - t0) * 1e-6;
        prof[i].wall_ms += w1 - w0;
        prof[i].flops += pending_flops;
        prof[i].calls++;
    }
    pending_flops = 0.0;
}

static void run1d(struct maestro_model *m, cl_kernel k, size_t n,
                  const char *what)
{
    size_t l = 256, g = (n + l - 1) / l * l;

    run(m, k, 1, &g, &l, what);
}

/* One GEMM operand: buffer, element offset, leading dimension and the
 * two batch strides. */
struct opnd {
    cl_mem buf;
    cl_long off;
    cl_int ld;
    cl_long s0, s1;
};

static struct opnd mat(cl_mem buf, long off, int ld)
{
    struct opnd o = {buf, off, ld, 0, 0};

    return o;
}

static struct opnd bmat(cl_mem buf, long off, int ld, long s0, long s1)
{
    struct opnd o = {buf, off, ld, s0, s1};

    return o;
}

static void gemm(struct maestro_model *m, int M, int N, int K, struct opnd A,
                 struct opnd B, int transb, struct opnd C, int nb, int nb1,
                 cl_mem bias, cl_mem R, long roff, float alpha, int act)
{
    cl_int cM = M, cN = N, cK = K, ct = transb, cnb1 = nb1, cact = act;
    cl_long croff = roff;
    cl_float calpha = alpha;
    size_t g[3], l[3] = {16, 16, 1};
    /* The 128 x 128 float4 kernel needs every A/B row start 16-byte
     * aligned; small or misaligned products use the generic kernel. */
    int big = M >= 128 && N >= 64 && K % 4 == 0 && A.ld % 4 == 0 &&
              A.off % 4 == 0 && A.s0 % 4 == 0 && A.s1 % 4 == 0 &&
              B.ld % 4 == 0 && B.off % 4 == 0 && B.s0 % 4 == 0 && B.s1 % 4 == 0;
    int tile = big ? 128 : 64;
    int half = m->ocl->fp16w && ocl_is_half(B.buf);
    int cpu = m->ocl->arch == OCL_ARCH_CPU;
    cl_kernel k;
    const char *what;

    if (cpu) {
        k = transb ? (half ? m->k_gemm_t_h : m->k_gemm_t)
                   : (half ? m->k_gemm_n_h : m->k_gemm_n);
        what = transb ? "gemm_cpu_t" : "gemm_cpu_n";
    }
    else if (big) {
        k = half ? m->k_gemm_big_h : m->k_gemm_big;
        what = "gemm_big";
    }
    else {
        k = half ? m->k_gemm_h : m->k_gemm;
        what = "gemm";
    }

    SETARG(k, 0, cM);
    SETARG(k, 1, cN);
    SETARG(k, 2, cK);
    SETARG(k, 3, A.buf);
    SETARG(k, 4, A.off);
    SETARG(k, 5, A.ld);
    SETARG(k, 6, A.s0);
    SETARG(k, 7, A.s1);
    SETARG(k, 8, B.buf);
    SETARG(k, 9, B.off);
    SETARG(k, 10, B.ld);
    SETARG(k, 11, B.s0);
    SETARG(k, 12, B.s1);
    SETARG(k, 13, ct);
    SETARG(k, 14, C.buf);
    SETARG(k, 15, C.off);
    SETARG(k, 16, C.ld);
    SETARG(k, 17, C.s0);
    SETARG(k, 18, C.s1);
    SETARG(k, 19, cnb1);
    SETARG(k, 20, bias);
    SETARG(k, 21, R);
    SETARG(k, 22, croff);
    SETARG(k, 23, calpha);
    SETARG(k, 24, cact);
    pending_flops = 2.0 * M * N * K * nb;
    if (cpu) {
        /* One work-item per 2 x 4 (transb) or 4 x 8 output block; the
         * work-group size is left to the driver. */
        g[0] = transb ? (size_t)(N + 3) / 4 : (size_t)(N + 7) / 8;
        g[1] = transb ? (size_t)(M + 1) / 2 : (size_t)(M + 3) / 4;
        g[2] = nb;
        run(m, k, 3, g, NULL, what);
        return;
    }
    g[0] = (size_t)((N + tile - 1) / tile) * 16;
    g[1] = (size_t)((M + tile - 1) / tile) * 16;
    g[2] = nb;
    run(m, k, 3, g, l, what);
}

/* Row LayerNorm over 768 channels, Y may alias X. */
static void layernorm(struct maestro_model *m, long rows, cl_mem X, cl_mem Y,
                      cl_mem g, cl_mem b, float eps, int act)
{
    cl_kernel k = m->k_ln;
    cl_int cC = EMB, cact = act;
    cl_float ceps = eps;
    size_t ls = 256, gs = (size_t)rows * ls;

    SETARG(k, 0, X);
    SETARG(k, 1, Y);
    SETARG(k, 2, g);
    SETARG(k, 3, b);
    SETARG(k, 4, cC);
    SETARG(k, 5, ceps);
    SETARG(k, 6, cact);
    run(m, k, 1, &gs, &ls, "layernorm");
}

static void softmax_rows(struct maestro_model *m, cl_mem S, long rows, int L)
{
    cl_kernel k = m->k_softmax;
    cl_int cL = L;
    size_t ls = 256, gs = (size_t)rows * ls;

    SETARG(k, 0, S);
    SETARG(k, 1, cL);
    run(m, k, 1, &gs, &ls, "softmax_rows");
}

static void add_bcast(struct maestro_model *m, cl_mem out, cl_mem a, cl_mem b,
                      int n, int bmod)
{
    cl_kernel k = m->k_add;
    cl_int cn = n, cbmod = bmod;

    SETARG(k, 0, out);
    SETARG(k, 1, a);
    SETARG(k, 2, b);
    SETARG(k, 3, cn);
    SETARG(k, 4, cbmod);
    run1d(m, k, (size_t)n, "add_bcast");
}

static void place_block(struct maestro_model *m, cl_mem in, cl_mem out, int T,
                        int dst_off, int block)
{
    cl_kernel k = m->k_place;
    cl_int cT = T, cdst = dst_off, cblock = block, cC = EMB;
    size_t g = (size_t)m->B * block * cC;

    SETARG(k, 0, in);
    SETARG(k, 1, out);
    SETARG(k, 2, cT);
    SETARG(k, 3, cdst);
    SETARG(k, 4, cblock);
    SETARG(k, 5, cC);
    run(m, k, 1, &g, NULL, "place_block");
}

static void take_block(struct maestro_model *m, cl_mem in, cl_mem out, int T,
                       int src_off, int block)
{
    cl_kernel k = m->k_take;
    cl_int cT = T, csrc = src_off, cblock = block, cC = EMB;
    size_t g = (size_t)m->B * block * cC;

    SETARG(k, 0, in);
    SETARG(k, 1, out);
    SETARG(k, 2, cT);
    SETARG(k, 3, csrc);
    SETARG(k, 4, cblock);
    SETARG(k, 5, cC);
    run(m, k, 1, &g, NULL, "take_block");
}

static void mean_dates(struct maestro_model *m, cl_mem in, cl_mem out, int D,
                       int L)
{
    cl_kernel k = m->k_mean;
    cl_int cD = D, cL = L, cC = EMB;
    size_t g = (size_t)m->B * L * cC;

    SETARG(k, 0, in);
    SETARG(k, 1, out);
    SETARG(k, 2, cD);
    SETARG(k, 3, cL);
    SETARG(k, 4, cC);
    run(m, k, 1, &g, NULL, "mean_dates");
}

/* One pre-norm Transformer block on X (B, T, 768), in place. */
static void encoder_layer(struct maestro_model *m, struct maestro_layer *l,
                          cl_mem X, int T)
{
    const int B = m->B;

    /* attention: X += to_out(softmax(scale q k^T) v) */
    layernorm(m, (long)B * T, X, m->Xln, l->ln1_w, l->ln1_b, LN_EPS, ACT_NONE);
    gemm(m, T, QKV, EMB, mat(m->Xln, 0, EMB), mat(l->qkv_w, 0, EMB), 1,
         mat(m->qkv, 0, QKV), B, 1, NULL, NULL, 0, 1.0f, ACT_NONE);
    /* scores (B, 12, T, T): q (aoff 0) times k (boff 768), 1/sqrt(64). */
    gemm(m, T, T, MAESTRO_DIM_HEAD, bmat(m->qkv, 0, QKV, (long)T * QKV,
                                        MAESTRO_DIM_HEAD),
         bmat(m->qkv, EMB, QKV, (long)T * QKV, MAESTRO_DIM_HEAD), 1,
         bmat(m->scores, 0, T, 12L * T * T, (long)T * T), B * 12, 12, NULL, NULL,
         0, 0.125f, ACT_NONE);
    softmax_rows(m, m->scores, (long)B * 12 * T, T);
    /* attention output (B, T, 768): p (aoff 0) times v (boff 1536). */
    gemm(m, T, MAESTRO_DIM_HEAD, T,
         bmat(m->scores, 0, T, 12L * T * T, (long)T * T),
         bmat(m->qkv, 2 * EMB, QKV, (long)T * QKV, MAESTRO_DIM_HEAD), 0,
         bmat(m->attn, 0, EMB, (long)T * EMB, MAESTRO_DIM_HEAD), B * 12, 12,
         NULL, NULL, 0, 1.0f, ACT_NONE);
    gemm(m, T, EMB, EMB, mat(m->attn, 0, EMB), mat(l->out_w, 0, EMB), 1,
         mat(X, 0, EMB), B, 1, l->out_b, X, 0, 1.0f, ACT_NONE);
    /* feed-forward: X += fc2(gelu(fc1(LN(X)))) */
    layernorm(m, (long)B * T, X, m->Xln, l->ln2_w, l->ln2_b, LN_EPS, ACT_NONE);
    gemm(m, T, MLP, EMB, mat(m->Xln, 0, EMB), mat(l->fc1_w, 0, EMB), 1,
         mat(m->ffn, 0, MLP), B, 1, l->fc1_b, NULL, 0, 1.0f, ACT_GELU);
    gemm(m, T, EMB, MLP, mat(m->ffn, 0, MLP), mat(l->fc2_w, 0, MLP), 1,
         mat(X, 0, EMB), B, 1, l->fc2_b, X, 0, 1.0f, ACT_NONE);
}

/* Run a full encoder stack (group or inter) on X (B, T, 768), in place,
 * then apply its final LayerNorm. */
static void encoder_run(struct maestro_model *m, struct maestro_encoder *e,
                        cl_mem X)
{
    int i;

    for (i = 0; i < e->depth; i++)
        encoder_layer(m, &e->layers[i], X, e->T);
    layernorm(m, (long)m->B * e->T, X, X, e->norm_w, e->norm_b, LN_EPS,
              ACT_NONE);
}

/* Patch embedding of one modality: image (B, D, c, S, S) already in
 * m->img -> conv 16 x 16, GroupNorm(1, 768), + positional/date encoding,
 * into mod->X (B, D*L, 768). */
static void patch_embed(struct maestro_model *m, struct maestro_mod *md)
{
    cl_kernel k = m->k_im2col;
    cl_int cside = md->image, cgrid = md->grid, cpatch = md->patch,
           cch = md->bands;
    size_t g = (size_t)m->B * md->ndates * md->L * md->K;
    const int D = md->ndates;

    SETARG(k, 0, m->img);
    SETARG(k, 1, m->im2col);
    SETARG(k, 2, cside);
    SETARG(k, 3, cgrid);
    SETARG(k, 4, cpatch);
    SETARG(k, 5, cch);
    run(m, k, 1, &g, NULL, "patch_im2col");
    /* (B, D, L, K) x (768, K)^T -> (B, D*L, 768) */
    gemm(m, md->L, EMB, md->K,
         bmat(m->im2col, 0, md->K, (long)D * md->L * md->K, (long)md->L * md->K),
         mat(md->conv_w, 0, md->K), 1,
         bmat(md->X, 0, EMB, (long)D * md->L * EMB, (long)md->L * EMB), m->B * D,
         D, md->conv_b, NULL, 0, 1.0f, ACT_NONE);
    layernorm(m, (long)m->B * D * md->L, md->X, md->X, md->gn_w, md->gn_b,
              LN_EPS, ACT_NONE);
    add_bcast(m, md->X, md->X, md->enc, m->B * D * md->L * EMB,
              D * md->L * EMB);
}

/* Positional + date encoding of one modality, (D, L, 768), computed on
 * the host and uploaded: block means of the 2-D sincos positional
 * encoding on the grid_pos_enc grid, plus the 8-channel date encoding
 * broadcast over all tokens. */
static void compute_enc(struct maestro_model *m, struct maestro_mod *md,
                        const struct maestro_date *dates,
                        const struct maestro_date *ref)
{
    const int g = md->grid, G = m->variant->grid_pos_enc, ratio = G / g;
    const int D = md->ndates, L = md->L;
    float *enc = G_malloc((size_t)D * L * EMB * sizeof(float));
    double *sx = G_malloc((size_t)G * 190 * sizeof(double));
    double *cx = G_malloc((size_t)G * 190 * sizeof(double));
    double *sy = G_malloc((size_t)G * 190 * sizeof(double));
    double *cy = G_malloc((size_t)G * 190 * sizeof(double));
    double omega[190];
    int i, j, d, c;

    for (i = 0; i < 190; i++)
        omega[i] = pow(10000.0, -(double)i / 189.0);
    /* Sums over each positional grid row/column block. */
    for (i = 0; i < G; i++)
        for (j = 0; j < 190; j++) {
            sx[i * 190 + j] = sin((double)i * omega[j]);
            cx[i * 190 + j] = cos((double)i * omega[j]);
        }
    for (d = 0; d < G; d++) {
        for (j = 0; j < 190; j++) {
            double s = 0.0, cc = 0.0;
            int t;

            for (t = 0; t < ratio; t++) {
                s += sx[(d * ratio + t) * 190 + j];
                cc += cx[(d * ratio + t) * 190 + j];
            }
            sy[d * 190 + j] = s;
            cy[d * 190 + j] = cc;
        }
    }
    for (d = 0; d < D; d++) {
        const double doy = dates[d].doy;
        const double diff = (dates[d].year + doy / 365.25) -
                            (ref->year + (double)ref->doy / 365.25);
        const double drad = 2.0 * M_PI * doy / 365.25;
        const double hrad = 2.0 * M_PI * dates[d].hour / 24.0;
        float extra[EMB];

        for (c = 0; c < EMB; c++)
            extra[c] = 0.0f;
        for (c = 0; c < 4; c++)
            extra[760 + c] = (float)diff;
        extra[764] = (float)sin(drad);
        extra[765] = (float)cos(drad);
        extra[766] = (float)sin(hrad);
        extra[767] = (float)cos(hrad);
        for (i = 0; i < L; i++) {
            const int ty = i / g, tx = i % g;

            for (c = 0; c < EMB; c++) {
                double p = 0.0;

                if (c < 190)
                    p = sx[tx * 190 + c] / ratio;
                else if (c < 380)
                    p = cx[tx * 190 + c - 190] / ratio;
                else if (c < 570)
                    p = sy[ty * 190 + c - 380] / ratio;
                else if (c < 760)
                    p = cy[ty * 190 + c - 570] / ratio;
                enc[((size_t)d * L + i) * EMB + c] = (float)p + extra[c];
            }
        }
    }
    ocl_write(m->ocl, md->enc, 0, enc, (size_t)D * L * EMB * sizeof(float));
    G_free(enc);
    G_free(sx);
    G_free(cx);
    G_free(sy);
    G_free(cy);
}

/* Model lifecycle. */

void maestro_model_init(struct maestro_model *m, struct ocl_backend *ocl,
                        const struct maestro_variant *v, const char *ckpt,
                        const int *ndates, int batch)
{
    int i, g, n, max_img = 0, max_col = 0;
    size_t bytes;

    memset(m, 0, sizeof(*m));
    m->ocl = ocl;
    m->variant = v;
    m->B = batch;
    m->nmods = v->nmods;
    memcpy(m->mods, v->mods, sizeof(m->mods[0]) * v->nmods);
    loaded_bytes = 0;

    pth_open(&m->pf, ckpt);
    G_message(_("Checkpoint <%s>: %.0f MB of tensor storage"), ckpt,
              pth_bytes(&m->pf, ""));

    /* Geometry, then weights of every modality. */
    for (i = 0; i < m->nmods; i++) {
        struct maestro_mod *md = &m->mods[i];
        char name[256];
        const long conv_shape[] = {EMB, -1, 1, 1};
        const long v_shape[] = {EMB};

        if (md->image % md->patch)
            G_fatal_error(_("Modality <%s>: image %d is not a multiple of "
                            "patch %d"),
                          md->name, md->image, md->patch);
        if (v->grid_pos_enc % (md->image / md->patch))
            G_fatal_error(_("Modality <%s>: positional grid %d is not a "
                            "multiple of the token grid %d"),
                          md->name, v->grid_pos_enc, md->image / md->patch);
        md->grid = md->image / md->patch;
        md->K = md->bands * md->patch * md->patch;
        md->L = md->grid * md->grid;
        md->ndates = ndates[i];
        if (md->ndates < 1 || md->ndates > md->num_dates)
            G_fatal_error(_("Modality <%s>: %d date slots, the checkpoint "
                            "supports 1 to %d"),
                          md->name, md->ndates, md->num_dates);

        snprintf(name, sizeof(name),
                 "model.patch_embed.%s.patchify_bands.0.conv.weight", md->name);
        md->conv_w = load_w(m, name, 4, conv_shape);
        snprintf(name, sizeof(name),
                 "model.patch_embed.%s.patchify_bands.0.conv.bias", md->name);
        md->conv_b = load_dev(m, name, 1, v_shape);
        snprintf(name, sizeof(name),
                 "model.patch_embed.%s.patchify_bands.0.norm.weight", md->name);
        md->gn_w = load_dev(m, name, 1, v_shape);
        snprintf(name, sizeof(name),
                 "model.patch_embed.%s.patchify_bands.0.norm.bias", md->name);
        md->gn_b = load_dev(m, name, 1, v_shape);
        md->enc = ocl_alloc(ocl,
                            (size_t)md->ndates * md->L * EMB * sizeof(float),
                            "date encoding");

        if (md->ndates * md->L * md->K > max_col)
            max_col = md->ndates * md->L * md->K;
        if (md->ndates * md->bands * md->image * md->image > max_img)
            max_img = md->ndates * md->bands * md->image * md->image;
    }

    /* Groups, in order of first appearance, then encoder_inter. */
    m->ngroups = 0;
    for (i = 0; i < m->nmods; i++) {
        for (g = 0; g < m->ngroups; g++)
            if (strcmp(m->encs[g].group, m->mods[i].group) == 0)
                break;
        if (g == m->ngroups) {
            if (m->ngroups == MAESTRO_MAX_GROUPS)
                G_fatal_error(_("Too many modality groups"));
            snprintf(m->encs[g].group, sizeof(m->encs[g].group), "%s",
                     m->mods[i].group);
            m->encs[g].T = 0;
            m->ngroups++;
        }
        m->encs[g].T += m->mods[i].ndates * m->mods[i].L;
    }
    m->T_inter = 0;
    for (g = 0; g < m->ngroups; g++)
        m->T_inter += m->encs[g].T;

    /* Load the Transformer stacks: groups first, then encoder_inter. */
    for (g = 0; g <= m->ngroups; g++) {
        struct maestro_encoder *e = &m->encs[g];
        const char *path = g == m->ngroups ? "encoder_inter" : "encoder";
        char name[256];
        const long v_shape[] = {EMB};
        const long qkv_shape[] = {QKV, EMB};
        const long o_shape[] = {EMB, EMB};
        const long fc1_shape[] = {MLP, EMB};
        const long fc2_shape[] = {EMB, MLP};

        if (g < m->ngroups) {
            snprintf(name, sizeof(name), "model.encoder.%s.layers.0.0.norm.weight",
                     e->group);
            if (!pth_find(&m->pf, name))
                G_fatal_error(_("No encoder for group <%s> in the checkpoint"),
                              e->group);
        }
        else
            snprintf(name, sizeof(name),
                     "model.encoder_inter.layers.0.0.norm.weight");
        e->depth = 0;
        for (;;) {
            char probe[256];

            if (g < m->ngroups)
                snprintf(probe, sizeof(probe),
                         "model.encoder.%s.layers.%d.0.norm.weight", e->group,
                         e->depth);
            else
                snprintf(probe, sizeof(probe),
                         "model.encoder_inter.layers.%d.0.norm.weight",
                         e->depth);
            if (!pth_find(&m->pf, probe))
                break;
            e->depth++;
        }
        if (e->depth < 1)
            G_fatal_error(_("Empty encoder for group <%s>"), e->group);
        if (g == m->ngroups)
            e->T = m->T_inter;
        e->layers = G_calloc(e->depth, sizeof(struct maestro_layer));

        for (n = 0; n < e->depth; n++) {
            struct maestro_layer *l = &e->layers[n];
            char pre[256];

            if (g < m->ngroups)
                snprintf(pre, sizeof(pre), "model.encoder.%s.layers.%d",
                         e->group, n);
            else
                snprintf(pre, sizeof(pre), "model.encoder_inter.layers.%d", n);
            snprintf(name, sizeof(name), "%s.0.norm.weight", pre);
            l->ln1_w = load_dev(m, name, 1, v_shape);
            snprintf(name, sizeof(name), "%s.0.norm.bias", pre);
            l->ln1_b = load_dev(m, name, 1, v_shape);
            snprintf(name, sizeof(name), "%s.0.to_qkv.weight", pre);
            l->qkv_w = load_w(m, name, 2, qkv_shape);
            snprintf(name, sizeof(name), "%s.0.to_out.0.weight", pre);
            l->out_w = load_w(m, name, 2, o_shape);
            snprintf(name, sizeof(name), "%s.0.to_out.0.bias", pre);
            l->out_b = load_dev(m, name, 1, v_shape);
            snprintf(name, sizeof(name), "%s.1.net.0.weight", pre);
            l->ln2_w = load_dev(m, name, 1, v_shape);
            snprintf(name, sizeof(name), "%s.1.net.0.bias", pre);
            l->ln2_b = load_dev(m, name, 1, v_shape);
            snprintf(name, sizeof(name), "%s.1.net.1.weight", pre);
            l->fc1_w = load_w(m, name, 2, fc1_shape);
            snprintf(name, sizeof(name), "%s.1.net.1.bias", pre);
            l->fc1_b = load_dev(m, name, 1, v_shape);
            snprintf(name, sizeof(name), "%s.1.net.4.weight", pre);
            l->fc2_w = load_w(m, name, 2, fc2_shape);
            snprintf(name, sizeof(name), "%s.1.net.4.bias", pre);
            l->fc2_b = load_dev(m, name, 1, v_shape);
        }
        if (g < m->ngroups)
            snprintf(name, sizeof(name), "model.encoder.%s.norm.weight",
                     e->group);
        else
            snprintf(name, sizeof(name), "model.encoder_inter.norm.weight");
        e->norm_w = load_dev(m, name, 1, v_shape);
        if (g < m->ngroups)
            snprintf(name, sizeof(name), "model.encoder.%s.norm.bias",
                     e->group);
        else
            snprintf(name, sizeof(name), "model.encoder_inter.norm.bias");
        e->norm_b = load_dev(m, name, 1, v_shape);

        G_message(_("Encoder %-8s: %d layers, %d tokens per tile"),
                  g == m->ngroups ? "inter" : e->group, e->depth, e->T);
    }

    /* Activations. */
    m->img = ocl_alloc(ocl, (size_t)m->B * max_img * sizeof(float),
                       "input images");
    m->im2col = ocl_alloc(ocl, (size_t)m->B * max_col * sizeof(float),
                          "im2col");
    bytes = (size_t)m->B * m->T_inter * EMB * sizeof(float);
    m->Xln = ocl_alloc(ocl, bytes, "layer norm output");
    m->attn = ocl_alloc(ocl, bytes, "attention output");
    m->inter = ocl_alloc(ocl, bytes, "encoder_inter input");
    m->qkv = ocl_alloc(ocl, (size_t)m->B * m->T_inter * QKV * sizeof(float),
                       "qkv");
    m->ffn = ocl_alloc(ocl, (size_t)m->B * m->T_inter * MLP * sizeof(float),
                       "feed-forward");
    m->scores =
        ocl_alloc(ocl, (size_t)m->B * 12 * m->T_inter * m->T_inter *
                           sizeof(float),
                  "attention scores");
    for (i = 0; i < m->nmods; i++) {
        struct maestro_mod *md = &m->mods[i];

        md->X = ocl_alloc(ocl,
                          (size_t)m->B * md->ndates * md->L * EMB *
                              sizeof(float),
                          "modality tokens");
        md->emb = ocl_alloc(ocl, (size_t)m->B * md->L * EMB * sizeof(float),
                            "modality embedding");
    }
    for (g = 0; g < m->ngroups; g++)
        m->encs[g].Xg = ocl_alloc(ocl,
                                  (size_t)m->B * m->encs[g].T * EMB *
                                      sizeof(float),
                                  "group tokens");

    /* Kernels. */
    m->k_gemm = ocl_kernel(ocl, "gemm");
    m->k_gemm_big = ocl_kernel(ocl, "gemm_big");
    m->k_gemm_t = ocl_kernel(ocl, "gemm_cpu_t");
    m->k_gemm_n = ocl_kernel(ocl, "gemm_cpu_n");
    if (ocl->fp16w) {
        m->k_gemm_h = ocl_kernel_h(ocl, "gemm");
        m->k_gemm_big_h = ocl_kernel_h(ocl, "gemm_big");
        m->k_gemm_t_h = ocl_kernel_h(ocl, "gemm_cpu_t");
        m->k_gemm_n_h = ocl_kernel_h(ocl, "gemm_cpu_n");
    }
    m->k_ln = ocl_kernel(ocl, "layernorm");
    m->k_softmax = ocl_kernel(ocl, "softmax_rows");
    m->k_im2col = ocl_kernel(ocl, "patch_im2col");
    m->k_add = ocl_kernel(ocl, "add_bcast");
    m->k_place = ocl_kernel(ocl, "place_block");
    m->k_take = ocl_kernel(ocl, "take_block");
    m->k_mean = ocl_kernel(ocl, "mean_dates");

    G_message(_("MAESTRO %s ready: %d modalities, %d groups, %d inter tokens, "
                "%d tiles per batch, %.0f MB of weights"),
              v->name, m->nmods, m->ngroups, m->T_inter, m->B,
              loaded_bytes / 1e6);
}

void maestro_set_dates(struct maestro_model *m, const struct maestro_date **dates,
                       const struct maestro_date *ref)
{
    int i;

    for (i = 0; i < m->nmods; i++)
        compute_enc(m, &m->mods[i], dates[i], ref);
}

void maestro_encode(struct maestro_model *m, const float *const *imgs,
                    float *const *embs)
{
    int i, g, off;
    int *goff = G_malloc(m->ngroups * sizeof(int));
    int *moff = G_malloc(m->ngroups * sizeof(int));
    int *gidx = G_malloc(m->nmods * sizeof(int));

    /* Patch embedding of every modality. */
    for (i = 0; i < m->nmods; i++) {
        struct maestro_mod *md = &m->mods[i];
        size_t n = (size_t)m->B * md->ndates * md->bands * md->image *
                   md->image * sizeof(float);

        ocl_write(m->ocl, m->img, 0, imgs[i], n);
        patch_embed(m, md);
    }

    /* Group membership, group offsets in the inter sequence. */
    for (i = 0; i < m->nmods; i++) {
        for (g = 0; g < m->ngroups; g++)
            if (strcmp(m->mods[i].group, m->encs[g].group) == 0)
                break;
        gidx[i] = g;
    }
    for (off = 0, g = 0; g < m->ngroups; g++) {
        goff[g] = off;
        moff[g] = 0;
        off += m->encs[g].T;
    }

    /* Group fusion, then encoder_inter, then split back. */
    for (i = 0; i < m->nmods; i++) {
        struct maestro_mod *md = &m->mods[i];

        g = gidx[i];
        place_block(m, md->X, m->encs[g].Xg, m->encs[g].T, moff[g],
                    md->ndates * md->L);
        moff[g] += md->ndates * md->L;
    }
    for (g = 0; g < m->ngroups; g++)
        encoder_run(m, &m->encs[g], m->encs[g].Xg);
    for (g = 0; g < m->ngroups; g++)
        place_block(m, m->encs[g].Xg, m->inter, m->T_inter, goff[g],
                    m->encs[g].T);
    encoder_run(m, &m->encs[m->ngroups], m->inter);
    for (g = 0; g < m->ngroups; g++)
        take_block(m, m->inter, m->encs[g].Xg, m->T_inter, goff[g],
                   m->encs[g].T);
    for (g = 0; g < m->ngroups; g++)
        moff[g] = 0;
    for (i = 0; i < m->nmods; i++) {
        struct maestro_mod *md = &m->mods[i];

        g = gidx[i];
        take_block(m, m->encs[g].Xg, md->X, m->encs[g].T, moff[g],
                   md->ndates * md->L);
        mean_dates(m, md->X, md->emb, md->ndates, md->L);
    }

    clFinish(m->ocl->queue);
    for (i = 0; i < m->nmods; i++)
        ocl_read(m->ocl, m->mods[i].emb, 0, embs[i],
                 (size_t)m->B * m->mods[i].L * EMB * sizeof(float));
    G_free(goff);
    G_free(moff);
    G_free(gidx);
}

void maestro_free(struct maestro_model *m)
{
    int p, i, g;

    for (p = 0; p < nprof; p++) {
        if (prof[p].flops > 0.0 && prof[p].wall_ms > 0.0)
            G_message("kernel %-24s %8.1f ms (wall %8.1f ms) %7ld calls "
                      "%8.1f GFLOP/s",
                      prof[p].what, prof[p].ms, prof[p].wall_ms, prof[p].calls,
                      prof[p].flops / (prof[p].wall_ms * 1e6));
        else
            G_message("kernel %-24s %8.1f ms (wall %8.1f ms) %7ld calls",
                      prof[p].what, prof[p].ms, prof[p].wall_ms, prof[p].calls);
    }

    for (i = 0; i < m->nmods; i++) {
        ocl_release(&m->mods[i].conv_w);
        ocl_release(&m->mods[i].conv_b);
        ocl_release(&m->mods[i].gn_w);
        ocl_release(&m->mods[i].gn_b);
        ocl_release(&m->mods[i].enc);
        ocl_release(&m->mods[i].X);
        ocl_release(&m->mods[i].emb);
    }
    for (g = 0; g <= m->ngroups; g++) {
        struct maestro_encoder *e = &m->encs[g];
        int n;

        for (n = 0; n < e->depth; n++) {
            ocl_release(&e->layers[n].ln1_w);
            ocl_release(&e->layers[n].ln1_b);
            ocl_release(&e->layers[n].qkv_w);
            ocl_release(&e->layers[n].out_w);
            ocl_release(&e->layers[n].out_b);
            ocl_release(&e->layers[n].ln2_w);
            ocl_release(&e->layers[n].ln2_b);
            ocl_release(&e->layers[n].fc1_w);
            ocl_release(&e->layers[n].fc1_b);
            ocl_release(&e->layers[n].fc2_w);
            ocl_release(&e->layers[n].fc2_b);
        }
        G_free(e->layers);
        ocl_release(&e->norm_w);
        ocl_release(&e->norm_b);
        ocl_release(&e->Xg);
    }
    ocl_release(&m->im2col);
    ocl_release(&m->Xln);
    ocl_release(&m->qkv);
    ocl_release(&m->scores);
    ocl_release(&m->attn);
    ocl_release(&m->ffn);
    ocl_release(&m->inter);
    cl_kernel kernels[] = {m->k_gemm,     m->k_gemm_big, m->k_gemm_t,
                           m->k_gemm_n,   m->k_gemm_h,   m->k_gemm_big_h,
                           m->k_gemm_t_h, m->k_gemm_n_h, m->k_ln,
                           m->k_softmax,  m->k_im2col,   m->k_add,
                           m->k_place,    m->k_take,     m->k_mean};
    for (i = 0; i < (int)(sizeof(kernels) / sizeof(kernels[0])); i++)
        if (kernels[i])
            clReleaseKernel(kernels[i]);
    pth_close(&m->pf);
}
