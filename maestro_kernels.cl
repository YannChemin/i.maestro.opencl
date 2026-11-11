/****************************************************************************
 *
 * MODULE:       i.maestro.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      OpenCL kernels for the MAESTRO multimodal encoder: GEMMs,
 *               layer/group norms, softmax, patch embedding and the block
 *               gather/scatter of the modality fusion. Embedded into the
 *               module at build time (see Makefile).
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

/* Tensor layout convention: activations are token-major (row = sequence
 * position, column = channel), which makes every nn.Linear and the
 * patchify convolution a plain row-major GEMM against the PyTorch
 * [out, in] weight matrix.
 *
 * Sequence layout of MAESTRO's group fusion: per modality the tokens of
 * all dates are concatenated along the sequence axis ((B, D, L, C) ->
 * (B, D*L, C)), modalities of one group are concatenated the same way,
 * then the outputs of all groups are concatenated for encoder_inter. */

#define EMBED_DIM 768
#define N_HEADS 12
#define DIM_HEAD 64

#define ACT_NONE 0
#define ACT_GELU 1
#define ACT_RELU 2

/* Storage of GEMM operand B. Built with -DMAESTRO_BHALF
 * (precision=fp16), the weight matrices are IEEE half read with
 * vload_half, which is core OpenCL and needs no cl_khr_fp16; the
 * arithmetic stays float. */
#ifdef MAESTRO_BHALF
#define BTYPE half
#define LDB(p, i) vload_half(0, (p) + (i))
#define LDB4(p) vload_half4(0, (p))
#define LDB8(p) vload_half8(0, (p))
#else
#define BTYPE float
#define LDB(p, i) ((p)[i])
#define LDB4(p) vload4(0, (p))
#define LDB8(p) vload8(0, (p))
#endif

float gelu_erf(float x)
{
    return 0.5f * x * (1.0f + erf(x * 0.70710678118654752f));
}

float apply_act(float v, int act)
{
    if (act == ACT_GELU)
        return gelu_erf(v);
    if (act == ACT_RELU)
        return fmax(v, 0.0f);
    return v;
}

/* Batched GEMM: C = act(alpha * A * op(B) + bias) + R.
 *
 * A is M x K row-major (lda). With transb != 0, B is stored N x K (the
 * PyTorch Linear weight layout, B(k, n) = B[n * ldb + k]); otherwise B
 * is K x N (B(k, n) = B[k * ldb + n]). Batch index b = get_group_id(2)
 * is split into (b / nb1, b % nb1), each part with its own element
 * stride per operand; a stride of 0 shares an operand across that batch
 * axis. Element offsets aoff/boff/coff/roff select sub-matrices of the
 * buffers. R, when used, has C's layout and may alias C for in-place
 * residual updates.
 *
 * Generic version: 64 x 64 output tile per work-group of 16 x 16 items,
 * each item computing a 4 x 4 register block. */
__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void
gemm(const int M, const int N, const int K, __global const float *A,
     const long aoff, const int lda, const long sa0, const long sa1,
     __global const BTYPE *B, const long boff, const int ldb, const long sb0,
     const long sb1, const int transb, __global float *C, const long coff,
     const int ldc, const long sc0, const long sc1, const int nb1,
     __global const float *bias, __global const float *R, const long roff,
     const float alpha, const int act)
{
    __local float As[16][65];
    __local float Bs[16][65];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int tid = ty * 16 + tx;
    const int n0 = get_group_id(0) * 64, m0 = get_group_id(1) * 64;
    const int b = get_group_id(2);
    const long b0 = b / nb1, b1 = b % nb1;
    float acc[4][4];
    int i, j, k, l;

    A += aoff + b0 * sa0 + b1 * sa1;
    B += boff + b0 * sb0 + b1 * sb1;
    C += coff + b0 * sc0 + b1 * sc1;
    if (R)
        R += roff + b0 * sc0 + b1 * sc1;

    for (i = 0; i < 4; i++)
        for (j = 0; j < 4; j++)
            acc[i][j] = 0.0f;

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (l = 0; l < 4; l++) {
            const int idx = tid + l * 256;
            const int r = idx >> 4, c = idx & 15;
            const int gm = m0 + r, gk = k0 + c;

            As[c][r] = (gm < M && gk < K) ? A[(long)gm * lda + gk] : 0.0f;
            if (transb) {
                const int gn = n0 + r;

                Bs[c][r] =
                    (gn < N && gk < K) ? LDB(B, (long)gn * ldb + gk) : 0.0f;
            }
            else {
                const int kk = idx >> 6, nn = idx & 63;
                const int gk2 = k0 + kk, gn = n0 + nn;

                Bs[kk][nn] =
                    (gn < N && gk2 < K) ? LDB(B, (long)gk2 * ldb + gn) : 0.0f;
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (k = 0; k < 16; k++) {
            float a[4], bb[4];

            for (i = 0; i < 4; i++)
                a[i] = As[k][ty + 16 * i];
            for (j = 0; j < 4; j++)
                bb[j] = Bs[k][tx + 16 * j];
            for (i = 0; i < 4; i++)
                for (j = 0; j < 4; j++)
                    acc[i][j] = mad(a[i], bb[j], acc[i][j]);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    for (i = 0; i < 4; i++) {
        const int m = m0 + ty + 16 * i;

        if (m >= M)
            continue;
        for (j = 0; j < 4; j++) {
            const int n = n0 + tx + 16 * j;
            float v;

            if (n >= N)
                continue;
            v = alpha * acc[i][j];
            if (bias)
                v += bias[n];
            v = apply_act(v, act);
            if (R)
                v += R[(long)m * ldc + n];
            C[(long)m * ldc + n] = v;
        }
    }
}

/* Epilogue of gemm_big: one row of an 8 x 8 register block. */
void store_row8(__global float *C, __global const float *R,
                __global const float *bias, const int m, const int n,
                const int M, const int N, const int ldc, const float alpha,
                const int act, const float4 lo, const float4 hi)
{
    const float v[8] = {lo.x, lo.y, lo.z, lo.w, hi.x, hi.y, hi.z, hi.w};

    if (m >= M)
        return;
    for (int j = 0; j < 8; j++) {
        float x;

        if (n + j >= N)
            return;
        x = alpha * v[j];
        if (bias)
            x += bias[n + j];
        x = apply_act(x, act);
        if (R)
            x += R[(long)m * ldc + n + j];
        C[(long)m * ldc + n + j] = x;
    }
}

/* Rank-1 update of the 8 x 8 block row r from A column value s. */
#define ROW_FMA(r, s) c##r##0 = mad((float4)(s), b0, c##r##0); c##r##1 = mad((float4)(s), b1, c##r##1)

/* Same contract as gemm, for large, 16-byte aligned operands: 128 x 128
 * output tile per 16 x 16 work-group, 8 x 8 register block per item
 * held in explicit float4 accumulators (private arrays of that size end
 * up in scratch memory with Mesa, costing a factor of 70), float4 loads.
 * Requires K, lda, ldb and all A/B offsets and batch strides to be
 * multiples of 4 (checked by the host). */
__kernel __attribute__((reqd_work_group_size(16, 16, 1))) void
gemm_big(const int M, const int N, const int K, __global const float *A,
         const long aoff, const int lda, const long sa0, const long sa1,
         __global const BTYPE *B, const long boff, const int ldb,
         const long sb0, const long sb1, const int transb, __global float *C,
         const long coff, const int ldc, const long sc0, const long sc1,
         const int nb1, __global const float *bias, __global const float *R,
         const long roff, const float alpha, const int act)
{
    __local float As[16][132];
    __local float Bs[16][132];
    const int tx = get_local_id(0), ty = get_local_id(1);
    const int tid = ty * 16 + tx;
    const int n0 = get_group_id(0) * 128, m0 = get_group_id(1) * 128;
    const int b = get_group_id(2);
    const long bb0 = b / nb1, bb1 = b % nb1;
    float4 c00 = 0.0f, c01 = 0.0f, c10 = 0.0f, c11 = 0.0f;
    float4 c20 = 0.0f, c21 = 0.0f, c30 = 0.0f, c31 = 0.0f;
    float4 c40 = 0.0f, c41 = 0.0f, c50 = 0.0f, c51 = 0.0f;
    float4 c60 = 0.0f, c61 = 0.0f, c70 = 0.0f, c71 = 0.0f;

    A += aoff + bb0 * sa0 + bb1 * sa1;
    B += boff + bb0 * sb0 + bb1 * sb1;
    C += coff + bb0 * sc0 + bb1 * sc1;
    if (R)
        R += roff + bb0 * sc0 + bb1 * sc1;

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int l = 0; l < 2; l++) {
            const int idx = tid + l * 256;
            const int r = idx >> 2, kq = (idx & 3) * 4;
            const int gm = m0 + r, gk = k0 + kq;
            float4 v = 0.0f;

            if (gm < M && gk < K)
                v = vload4(0, A + (long)gm * lda + gk);
            As[kq][r] = v.x;
            As[kq + 1][r] = v.y;
            As[kq + 2][r] = v.z;
            As[kq + 3][r] = v.w;

            if (transb) {
                const int gn = n0 + r;

                v = 0.0f;
                if (gn < N && gk < K)
                    v = LDB4(B + (long)gn * ldb + gk);
                Bs[kq][r] = v.x;
                Bs[kq + 1][r] = v.y;
                Bs[kq + 2][r] = v.z;
                Bs[kq + 3][r] = v.w;
            }
            else {
                const int kk = idx >> 5, nq = (idx & 31) * 4;
                const int gk2 = k0 + kk, gn = n0 + nq;

                v = 0.0f;
                if (gk2 < K) {
                    __global const BTYPE *bp = B + (long)gk2 * ldb + gn;

                    if (gn + 3 < N)
                        v = LDB4(bp);
                    else {
                        if (gn < N)
                            v.x = LDB(bp, 0);
                        if (gn + 1 < N)
                            v.y = LDB(bp, 1);
                        if (gn + 2 < N)
                            v.z = LDB(bp, 2);
                    }
                }
                vstore4(v, 0, &Bs[kk][nq]);
            }
        }
        barrier(CLK_LOCAL_MEM_FENCE);

        for (int k = 0; k < 16; k++) {
            const float4 a0 = vload4(0, &As[k][ty * 8]);
            const float4 a1 = vload4(0, &As[k][ty * 8 + 4]);
            const float4 b0 = vload4(0, &Bs[k][tx * 8]);
            const float4 b1 = vload4(0, &Bs[k][tx * 8 + 4]);

            ROW_FMA(0, a0.x);
            ROW_FMA(1, a0.y);
            ROW_FMA(2, a0.z);
            ROW_FMA(3, a0.w);
            ROW_FMA(4, a1.x);
            ROW_FMA(5, a1.y);
            ROW_FMA(6, a1.z);
            ROW_FMA(7, a1.w);
        }
        barrier(CLK_LOCAL_MEM_FENCE);
    }

    store_row8(C, R, bias, m0 + ty * 8 + 0, n0 + tx * 8, M, N, ldc, alpha, act,
               c00, c01);
    store_row8(C, R, bias, m0 + ty * 8 + 1, n0 + tx * 8, M, N, ldc, alpha, act,
               c10, c11);
    store_row8(C, R, bias, m0 + ty * 8 + 2, n0 + tx * 8, M, N, ldc, alpha, act,
               c20, c21);
    store_row8(C, R, bias, m0 + ty * 8 + 3, n0 + tx * 8, M, N, ldc, alpha, act,
               c30, c31);
    store_row8(C, R, bias, m0 + ty * 8 + 4, n0 + tx * 8, M, N, ldc, alpha, act,
               c40, c41);
    store_row8(C, R, bias, m0 + ty * 8 + 5, n0 + tx * 8, M, N, ldc, alpha, act,
               c50, c51);
    store_row8(C, R, bias, m0 + ty * 8 + 6, n0 + tx * 8, M, N, ldc, alpha, act,
               c60, c61);
    store_row8(C, R, bias, m0 + ty * 8 + 7, n0 + tx * 8, M, N, ldc, alpha, act,
               c70, c71);
}

/* Epilogue shared by the CPU GEMMs: one output element. */
void store_one(__global float *C, __global const float *R,
               __global const float *bias, const int m, const int n,
               const int ldc, const float alpha, const int act, const float v)
{
    float x = alpha * v;

    if (bias)
        x += bias[n];
    x = apply_act(x, act);
    if (R)
        x += R[(long)m * ldc + n];
    C[(long)m * ldc + n] = x;
}

float hsum8(const float8 v)
{
    const float4 s = v.lo + v.hi;

    return s.x + s.y + s.z + s.w;
}

/* CPU GEMMs (profile=cpu), same contract as gemm. No local memory and no
 * barriers, which PoCL turns into slow scalar loops: each work-item owns
 * a register block and streams the contiguous axis with float8 loads.
 *
 * transb != 0 (B stored N x K, the Linear weight layout): work-item
 * (n / 4, m / 2) computes 2 x 4 outputs, dot products along K. */
__kernel void gemm_cpu_t(const int M, const int N, const int K,
                         __global const float *A, const long aoff,
                         const int lda, const long sa0, const long sa1,
                         __global const BTYPE *B, const long boff,
                         const int ldb, const long sb0, const long sb1,
                         const int transb, __global float *C, const long coff,
                         const int ldc, const long sc0, const long sc1,
                         const int nb1, __global const float *bias,
                         __global const float *R, const long roff,
                         const float alpha, const int act)
{
    const int n0 = get_global_id(0) * 4, m0 = get_global_id(1) * 2;
    const int b = get_global_id(2);
    const long bb0 = b / nb1, bb1 = b % nb1;
    float8 s[2][4];
    float t[2][4];
    int i, j, k;

    if (m0 >= M || n0 >= N)
        return;
    A += aoff + bb0 * sa0 + bb1 * sa1;
    B += boff + bb0 * sb0 + bb1 * sb1;
    C += coff + bb0 * sc0 + bb1 * sc1;
    if (R)
        R += roff + bb0 * sc0 + bb1 * sc1;

    for (i = 0; i < 2; i++)
        for (j = 0; j < 4; j++)
            s[i][j] = 0.0f;
    if (m0 + 1 < M && n0 + 3 < N && (lda % 8 == 0) && (ldb % 8 == 0)) {
        for (k = 0; k + 8 <= K; k += 8) {
            const float8 a0 = vload8(0, A + (long)m0 * lda + k);
            const float8 a1 = vload8(0, A + (long)(m0 + 1) * lda + k);

            for (j = 0; j < 4; j++) {
                const float8 bv = LDB8(B + (long)(n0 + j) * ldb + k);

                s[0][j] = mad(a0, bv, s[0][j]);
                s[1][j] = mad(a1, bv, s[1][j]);
            }
        }
    }
    else
        k = 0;
    for (i = 0; i < 2; i++)
        for (j = 0; j < 4; j++) {
            const int m = m0 + i, n = n0 + j;
            float v = hsum8(s[i][j]);
            int kk;

            if (m >= M || n >= N)
                continue;
            for (kk = k; kk < K; kk++)
                v = mad(A[(long)m * lda + kk], LDB(B, (long)n * ldb + kk), v);
            t[i][j] = v;
            store_one(C, R, bias, m, n, ldc, alpha, act, t[i][j]);
        }
}

/* transb == 0 (B stored K x N): work-item (n / 8, m / 4) computes 4 x 8
 * outputs, one float8 row of B per step of K. */
__kernel void gemm_cpu_n(const int M, const int N, const int K,
                         __global const float *A, const long aoff,
                         const int lda, const long sa0, const long sa1,
                         __global const BTYPE *B, const long boff,
                         const int ldb, const long sb0, const long sb1,
                         const int transb, __global float *C, const long coff,
                         const int ldc, const long sc0, const long sc1,
                         const int nb1, __global const float *bias,
                         __global const float *R, const long roff,
                         const float alpha, const int act)
{
    const int n0 = get_global_id(0) * 8, m0 = get_global_id(1) * 4;
    const int b = get_global_id(2);
    const long bb0 = b / nb1, bb1 = b % nb1;
    float8 c[4];
    int i, j, k;

    if (m0 >= M || n0 >= N)
        return;
    A += aoff + bb0 * sa0 + bb1 * sa1;
    B += boff + bb0 * sb0 + bb1 * sb1;
    C += coff + bb0 * sc0 + bb1 * sc1;
    if (R)
        R += roff + bb0 * sc0 + bb1 * sc1;

    for (i = 0; i < 4; i++)
        c[i] = 0.0f;
    if (n0 + 7 < N && (ldb % 8 == 0) && (n0 % 8 == 0)) {
        for (k = 0; k < K; k++) {
            const float8 bv = LDB8(B + (long)k * ldb + n0);

            for (i = 0; i < 4; i++)
                if (m0 + i < M)
                    c[i] = mad((float8)(A[(long)(m0 + i) * lda + k]), bv, c[i]);
        }
        for (i = 0; i < 4 && m0 + i < M; i++) {
            const float v[8] = {c[i].s0, c[i].s1, c[i].s2, c[i].s3,
                                c[i].s4, c[i].s5, c[i].s6, c[i].s7};

            for (j = 0; j < 8; j++)
                store_one(C, R, bias, m0 + i, n0 + j, ldc, alpha, act, v[j]);
        }
        return;
    }
    for (i = 0; i < 4 && m0 + i < M; i++)
        for (j = 0; j < 8 && n0 + j < N; j++) {
            float v = 0.0f;

            for (k = 0; k < K; k++)
                v = mad(A[(long)(m0 + i) * lda + k],
                        LDB(B, (long)k * ldb + n0 + j), v);
            store_one(C, R, bias, m0 + i, n0 + j, ldc, alpha, act, v);
        }
}

/* The other kernels are not needed in the half-precision GEMM program. */
#ifndef MAESTRO_GEMM_ONLY

/* Sum and max of val over a work-group of n items (power of two). */
float wg_sum(float val, __local float *scratch, const int n)
{
    const int lid = get_local_id(0);

    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = n / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] += scratch[lid + s];
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return val;
}

float wg_max(float val, __local float *scratch, const int n)
{
    const int lid = get_local_id(0);

    scratch[lid] = val;
    barrier(CLK_LOCAL_MEM_FENCE);
    for (int s = n / 2; s > 0; s >>= 1) {
        if (lid < s)
            scratch[lid] = fmax(scratch[lid], scratch[lid + s]);
        barrier(CLK_LOCAL_MEM_FENCE);
    }
    val = scratch[0];
    barrier(CLK_LOCAL_MEM_FENCE);
    return val;
}

/* Row-wise LayerNorm over C channels, Y may alias X. Also the
 * GroupNorm(1, C) of the patch embedding: with a single group over the
 * channel axis of a token-major matrix the two are the same operation.
 * The work-group size (256 for the 768- and 3072-channel rows) is the
 * local size chosen by the host. */
__kernel void layernorm(__global const float *X, __global float *Y,
                        __global const float *g, __global const float *bt,
                        const int C, const float eps, const int act)
{
    __local float scratch[256];
    const int lid = get_local_id(0), n = get_local_size(0);
    const long row = get_group_id(0);
    __global const float *x = X + row * C;
    __global float *y = Y + row * C;
    float s = 0.0f, mean, var;
    int c;

    for (c = lid; c < C; c += n)
        s += x[c];
    mean = wg_sum(s, scratch, n) / C;
    s = 0.0f;
    for (c = lid; c < C; c += n) {
        const float d = x[c] - mean;

        s += d * d;
    }
    var = wg_sum(s, scratch, n) / C;
    s = rsqrt(var + eps);
    for (c = lid; c < C; c += n)
        y[c] = apply_act((x[c] - mean) * s * g[c] + bt[c], act);
}

/* In-place row softmax of attention scores (full sequence attention):
 * one 256-item work-group per row of L keys. Rows are ordered
 * (batch, head, query). */
__kernel __attribute__((reqd_work_group_size(256, 1, 1))) void
softmax_rows(__global float *S, const int L)
{
    __local float scratch[256];
    const int lid = get_local_id(0);
    __global float *s = S + (long)get_group_id(0) * L;
    float m = -INFINITY, sum = 0.0f, v;
    int j;

    for (j = lid; j < L; j += 256) {
        v = s[j];
        m = fmax(m, v);
    }
    m = wg_max(m, scratch, 256);
    for (j = lid; j < L; j += 256) {
        v = exp(s[j] - m);
        s[j] = v;
        sum += v;
    }
    sum = wg_sum(sum, scratch, 256);
    v = 1.0f / sum;
    for (j = lid; j < L; j += 256)
        s[j] *= v;
}

/* Patch extraction of a stack of channels-first images into token-major
 * rows: image (ni, C, side, side) -> out (ni * grid^2, C * p * p) with
 * column order c * p * p + ky * p + kx to match the Conv2d weight
 * layout [E, C, p, p], tokens in row-major grid order. One work-item
 * per output element. */
__kernel void patch_im2col(__global const float *img, __global float *out,
                           const int side, const int grid, const int patch,
                           const int channels)
{
    const int K = channels * patch * patch;
    const int gid = get_global_id(0);
    const int ni = gid / (grid * grid * K);
    const int rem = gid % (grid * grid * K);
    const int t = rem / K, col = rem % K;
    const int c = col / (patch * patch), kp = col % (patch * patch);
    const int ky = kp / patch, kx = kp % patch;
    const int ty = t / grid, tx = t % grid;
    const int i = ni * channels + c, y = ty * patch + ky, x = tx * patch + kx;

    out[gid] = img[((long)i * side + y) * side + x];
}

/* out[i] = a[i] + b[i % bmod]; out may alias a. */
__kernel void add_bcast(__global float *out, __global const float *a,
                        __global const float *b, const int n, const int bmod)
{
    const int i = get_global_id(0);

    if (i < n)
        out[i] = a[i] + b[i % bmod];
}

/* Token-block gather for the group fusion: copy the (B, block, C)
 * modality buffer into the (B, T, C) group sequence at token offset
 * dst_off. One work-item per element of one batch sample's block:
 * gid = b * block * C + r -> out[(b * T + dst_off) * C + r] = in[gid]. */
__kernel void place_block(__global const float *in, __global float *out,
                          const int T, const int dst_off, const int block,
                          const int C)
{
    const int gid = get_global_id(0);
    const int b = gid / (block * C), r = gid % (block * C);

    out[((long)b * T + dst_off) * C + r] = in[gid];
}

/* Token-block scatter, the inverse of place_block: extract the member
 * modality (B, block, C) at token offset src_off of the (B, T, C)
 * sequence. gid = b * block * C + r -> out[gid] = in[(b * T + src_off) *
 * C + r]. */
__kernel void take_block(__global const float *in, __global float *out,
                         const int T, const int src_off, const int block,
                         const int C)
{
    const int gid = get_global_id(0);
    const int b = gid / (block * C), r = gid % (block * C);

    out[gid] = in[((long)b * T + src_off) * C + r];
}

/* Mean over the date axis of a modality embedding: in (B, D, L, C) ->
 * out (B, L, C), out[b, l, c] = mean over d of in[b, d, l, c]. */
__kernel void mean_dates(__global const float *in, __global float *out,
                         const int D, const int L, const int C)
{
    const int gid = get_global_id(0);
    const int b = gid / (L * C), rem = gid % (L * C);
    const int l = rem / C, c = rem % C;
    float s = 0.0f;

    for (int d = 0; d < D; d++)
        s += in[(((long)b * D + d) * L + l) * C + c];
    out[gid] = s / D;
}

#endif /* MAESTRO_GEMM_ONLY */
