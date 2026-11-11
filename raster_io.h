/* SPDX-License-Identifier: Unlicense */

#ifndef I_MAESTRO_OPENCL_RASTER_IO_H
#define I_MAESTRO_OPENCL_RASTER_IO_H

#include <grass/raster.h>

/* Read one band of one tile at the map's native resolution, resample it
 * to image x image with the exact floor(i * in / out) mapping of
 * F.interpolate(mode="nearest"), and normalize: NULL -> 0, else
 * log(max(v, 1e-10)) when log_scale, then v / norm_fac. `tile` holds the
 * tile bounds in the location projection; the band is read over the
 * intersection with the map extent. */
void raster_read_plane(const char *name, const char *mapset,
                       const struct Cell_head *tile, int image,
                       double norm_fac, int log_scale, float *out);

/* An output raster of `bands` channels covering the tiles extent. Tiles
 * are added in row-major order; a complete tile row is written once its
 * last tile arrives. */
struct raster_out {
    int fd, bands, cols, g, tile_cols;
    float *row;  /* one output row, cols values */
    float *rb;   /* bands * g * cols, the tile row being accumulated */
    int cur_row; /* tile row being accumulated, -1 when empty */
    char *name;
};

void raster_out_open(struct raster_out *o, const char *name,
                     const struct Cell_head *win, int bands, int g,
                     int tile_cols);
/* Add the (g*g, bands) embedding of one tile at column block tile_col of
 * tile row tile_row. */
void raster_out_add(struct raster_out *o, int tile_row, int tile_col,
                    const float *emb);
void raster_out_flush(struct raster_out *o);
void raster_out_close(struct raster_out *o);

#endif /* I_MAESTRO_OPENCL_RASTER_IO_H */
