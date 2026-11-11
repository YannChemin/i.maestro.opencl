/****************************************************************************
 *
 * MODULE:       i.maestro.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Raster input and output of the MAESTRO encoder: per-tile
 *               native-resolution band reads with the dataset
 *               normalization, exact nearest resampling, and embedding
 *               rasters written tile row by tile row.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <math.h>
#include <string.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include "raster_io.h"

void raster_read_plane(const char *name, const char *mapset,
                       const struct Cell_head *tile, int image,
                       double norm_fac, int log_scale, float *out)
{
    struct Cell_head ch, win;
    int fd, r, c, i, j;
    double *buf;
    const int n = image * image;

    Rast_get_cellhd(name, mapset, &ch);

    /* Map window covering the tile, at the map's native resolution. */
    {
        double west = fmax(tile->west, ch.west);
        double east = fmin(tile->east, ch.east);
        double south = fmax(tile->south, ch.south);
        double north = fmin(tile->north, ch.north);
        long c0, c1, r0, r1;

        if (east <= west || north <= south) {
            for (i = 0; i < n; i++)
                out[i] = 0.0f;
            return;
        }
        c0 = (long)floor((west - ch.west) / ch.ew_res);
        c1 = (long)ceil((east - ch.west) / ch.ew_res);
        r0 = (long)floor((ch.north - north) / ch.ns_res);
        r1 = (long)ceil((ch.north - south) / ch.ns_res);
        if (c0 < 0)
            c0 = 0;
        if (r0 < 0)
            r0 = 0;
        if (c1 > ch.cols)
            c1 = ch.cols;
        if (r1 > ch.rows)
            r1 = ch.rows;
        if (c1 <= c0 || r1 <= r0) {
            for (i = 0; i < n; i++)
                out[i] = 0.0f;
            return;
        }
        win = ch;
        win.west = ch.west + c0 * ch.ew_res;
        win.east = ch.west + c1 * ch.ew_res;
        win.north = ch.north - r0 * ch.ns_res;
        win.south = ch.north - r1 * ch.ns_res;
        win.cols = (int)(c1 - c0);
        win.rows = (int)(r1 - r0);
        win.ew_res = ch.ew_res;
        win.ns_res = ch.ns_res;
    }

    Rast_set_input_window(&win);
    fd = Rast_open_old(name, mapset);
    buf = G_malloc((size_t)win.cols * win.rows * sizeof(double));
    for (r = 0; r < win.rows; r++)
        Rast_get_d_row(fd, buf + (size_t)r * win.cols, r);
    Rast_close(fd);

    /* Nearest resampling of F.interpolate: out[i] = in[floor(i * in / out)]. */
    for (i = 0; i < image; i++) {
        const int si = (int)((long)i * win.rows / image);

        for (j = 0; j < image; j++) {
            const int sj = (int)((long)j * win.cols / image);
            const double v = buf[(size_t)si * win.cols + sj];

            if (Rast_is_d_null_value(&v))
                out[(size_t)i * image + j] = 0.0f;
            else {
                double x = v;

                if (log_scale)
                    x = log(fmax(x, 1e-10));
                out[(size_t)i * image + j] = (float)(x / norm_fac);
            }
        }
    }
    G_free(buf);
}

void raster_out_open(struct raster_out *o, const char *name,
                     const struct Cell_head *win, int g, int tile_cols)
{
    memset(o, 0, sizeof(*o));
    o->g = g;
    o->tile_cols = tile_cols;
    o->cols = win->cols;
    o->cur_row = -1;
    o->name = G_store(name);
    Rast_set_output_window(win);
    o->fd = Rast_open_new(name, FCELL_TYPE);
    o->row = G_malloc((size_t)o->cols * sizeof(float));
    o->rb = G_malloc((size_t)g * o->cols * sizeof(float));
}

void raster_out_add(struct raster_out *o, int tile_row, int tile_col,
                    const float *emb, int stride, int band)
{
    int y, x;

    if (o->cur_row >= 0 && o->cur_row != tile_row)
        raster_out_flush(o);
    o->cur_row = tile_row;
    for (y = 0; y < o->g; y++)
        for (x = 0; x < o->g; x++)
            o->rb[(size_t)y * o->cols + tile_col * o->g + x] =
                emb[(size_t)(y * o->g + x) * stride + band];
}

void raster_out_flush(struct raster_out *o)
{
    int y;

    if (o->cur_row < 0)
        return;
    for (y = 0; y < o->g; y++) {
        memcpy(o->row, o->rb + (size_t)y * o->cols,
               (size_t)o->cols * sizeof(float));
        Rast_put_f_row(o->fd, o->cur_row * o->g + y, o->row);
    }
    o->cur_row = -1;
}

void raster_out_close(struct raster_out *o)
{
    struct History hist;

    raster_out_flush(o);
    Rast_close(o->fd);
    Rast_short_history(o->name, "raster", &hist);
    snprintf(hist.title, sizeof(hist.title), "MAESTRO embeddings");
    snprintf(hist.datsrc_1, sizeof(hist.datsrc_1),
             "MAESTRO multimodal encoder (i.maestro.opencl)");
    Rast_write_history(o->name, &hist);
    G_free(o->row);
    G_free(o->rb);
    G_free(o->name);
    memset(o, 0, sizeof(*o));
    o->cur_row = -1;
}
