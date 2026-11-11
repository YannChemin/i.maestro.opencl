/****************************************************************************
 *
 * MODULE:       i.maestro.opencl
 * AUTHOR(S):    Yann Chemin <dr.yann.chemin gmail.com>
 * PURPOSE:      Apply the MAESTRO multimodal masked-autoencoder foundation
 *               model (encoder only) on an OpenCL device, reading the
 *               original PyTorch checkpoint, and write the per-modality
 *               embedding rasters at the patch grid resolution.
 *
 * SPDX-License-Identifier: Unlicense
 *
 *****************************************************************************/

#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/time.h>
#include <time.h>

#include <grass/gis.h>
#include <grass/glocale.h>
#include <grass/raster.h>

#include "download.h"
#include "maestro_model.h"
#include "ocl_backend.h"
#include "progress.h"
#include "pth_loader.h"
#include "raster_io.h"

/* Day-of-year computation (GRASS 8.6 has no G_yday). */
static int day_of_year(int year, int month, int day)
{
    static const int doy[] = {0, 0, 31, 59, 90, 120, 151,
                               181, 212, 243, 273, 304, 334};
    int doy_base = doy[month];

    if (month > 2 && (year % 4 == 0 && (year % 100 != 0 || year % 400 == 0)))
        doy_base++;
    return doy_base + day;
}

/* MAESTRO variants registry (published on HuggingFace). */
static const struct maestro_variant *maestro_variants(void)
{
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
            .mods = {
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
            .mods = {
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
            .mods = {
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
            .mods = {
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
            .repo = "IGNF/MAESTRO_S2-NAIP-urban",
            .file = "MAESTRO_S2-NAIP-urban/checkpoints/pretrain-epoch=14.ckpt",
            .size = 0,
            .published = 0,
            .crop_meters = 120.0,
            .grid_pos_enc = 192,
            .nmods = 4,
            .mods = {
                {"aerial", "aerial", 384, 24, 16, 4, 255.0, 0, 0},
                {"spot", "spot", 128, 8, 16, 3, 1, 255.0, 0, 0},
                {"s2", "s2", 12, 6, 2, 10, 16, 5000.0, 0, 0},
                {"s1", "s1", 12, 6, 2, 2, 4, 20.0, 0, 0},
            },
        },
    };
    return variants;
}

#define N_VARIANTS 5

/* Default weights directory. */
static char *default_weights_dir(void)
{
    const char *base = getenv("MAESTRO_MODELS_DIR");
    char *dir;
    if (base && *base)
        G_asprintf(&dir, "%s/maestro", base);
    else
        G_asprintf(&dir, "%s/models/maestro", G_home());
    return dir;
}

/* Download a checkpoint into dir; returns its path. */
static char *fetch_checkpoint(const char *dir, const struct maestro_variant *var)
{
    const char *base = getenv("I_MAESTRO_OPENCL_DOWNLOAD_URL");
    char *url, *path;

    if (!base || !*base)
        base = "https://huggingface.co";
    G_asprintf(&url, "%s/%s/resolve/main/%s", base, var->repo, var->file);
    G_asprintf(&path, "%s/%s/%s", dir, var->name, var->file);
    download_file(url, path, var->size);
    G_free(url);
    return path;
}

/* Find checkpoint in dir for variant, or NULL. */
static char *find_checkpoint(const char *dir, const char *want)
{
    DIR *d;
    struct dirent *e;
    char path[GPATH_MAX];
    const struct maestro_variant *variants = maestro_variants();
    int i;

    /* Find the variant by name. */
    for (i = 0; i < N_VARIANTS; i++) {
        if (strcmp(variants[i].name, want) == 0)
            break;
    }
    if (i == N_VARIANTS)
        return NULL;

    d = opendir(dir);
    if (!d) {
        if (errno == ENOENT)
            return NULL;
        G_fatal_error(_("Unable to open weights folder <%s>: %s"), dir,
                      strerror(errno));
    }

    while ((e = readdir(d))) {
        size_t len = strlen(e->d_name);
        struct stat st;

        if (len < 6 || strcmp(e->d_name + len - 5, ".ckpt") != 0)
            continue;
        snprintf(path, sizeof(path), "%s/%s", dir, e->d_name);
        if (stat(path, &st) != 0 || !S_ISREG(st.st_mode))
            continue;
        if (strstr(e->d_name, want) ||
            strstr(e->d_name, "pretrain-epoch") ||
            strstr(e->d_name, "MAESTRO")) {
            closedir(d);
            return G_store(path);
        }
    }
    closedir(d);
    return NULL;
}

/* Parse a date string YYYY-MM-DD[THH] into (year, doy, hour). */
static void parse_date(const char *s, int *year, int *doy, int *hour)
{
    int mon, day, hh = 0;
    if (sscanf(s, "%d-%d-%d", year, &mon, &day) != 3)
        G_fatal_error(_("Invalid date string <%s>"), s);
    *doy = day_of_year(*year, mon, day);
    const char *t = strchr(s, 'T');
    if (t && sscanf(t + 1, "%d", &hh) == 1)
        *hour = hh;
    else
        *hour = 0;
}

/* Parse channel range or list. */
static void parse_channels(const char *s, int *ch_start, int *ch_end, int *nch)
{
    char *p;
    *ch_start = atoi(s);
    p = strchr(s, '-');
    if (p) {
        *ch_end = atoi(p + 1);
        if (*ch_end <= *ch_start)
            G_fatal_error(_("Invalid channel range <%s>"), s);
    }
    else {
        *ch_end = *ch_start + 1;
    }
    *nch = *ch_end - *ch_start;
    if (*ch_start < 1 || *ch_end > MAESTRO_EMBED_DIM + 1)
        G_fatal_error(_("Channels must be in range 1-%d"), MAESTRO_EMBED_DIM);
    (*ch_start)--;
    (*ch_end)--;
}

/* Parse user dates. */
static char **parse_user_dates(const char *s, int *n)
{
    char **dates = NULL;
    int cap = 0, count = 0;
    const char *p, *start = s;

    *n = 0;
    if (!s || !*s)
        return NULL;

    while (*start) {
        p = strchr(start, ',');
        if (!p)
            p = start + strlen(start);
        if (count == cap) {
            cap = cap ? 2 * cap : 8;
            dates = G_realloc(dates, sizeof(char *) * cap);
        }
        dates[count] = G_malloc(p - start + 1);
        memcpy(dates[count], start, p - start);
        dates[count][p - start] = '\0';
        count++;
        if (*p == ',')
            start = p + 1;
        else
            break;
    }
    *n = count;
    return dates;
}

static double now(void)
{
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec * 1e-6;
}

int main(int argc, char *argv[])
{
    struct GModule *module;
    struct {
        struct Option *aerial, *spot, *dem, *s2, *s1, *s1_asc, *s1_des;
        struct Option *model, *weights_dir, *checkpoint;
        struct Option *tile_meters, *batch, *channels, *dates, *output;
        struct Option *device, *platform, *profile, *precision;
    } opt;
    struct Flag *no_download;
    struct ocl_backend ocl;
    struct maestro_model model_ctx;
    struct Cell_head win, tile;
    struct raster_out *out = NULL;
    double t0, t1, load_gib;
    const struct maestro_variant *variant = NULL;
    char *checkpoint_path = NULL;
    char *dir = NULL;
    int nmods_active = 0;
    int mod_active[MAESTRO_MAX_MODS] = {0};
    int ndates[MAESTRO_MAX_MODS];
    int tile_rows, tile_cols, tile_row, tile_col;
    int nchannels_out, ch_start, ch_end;
    char **user_dates = NULL;
    int n_user_dates = 0;
    int mi, oi, ch, b;

    G_gisinit(argv[0]);

    module = G_define_module();
    G_add_keyword(_("imagery"));
    G_add_keyword(_("deep learning"));
    G_add_keyword(_("multimodal"));
    G_add_keyword(_("foundation model"));
    G_add_keyword(_("OpenCL"));
    G_add_keyword(_("GPU"));
    module->description =
        _("Applies the MAESTRO multimodal encoder on imagery groups with "
          "an OpenCL device and outputs per-modality embedding rasters.");

    /* Input modalities. */
    opt.aerial = G_define_option();
    opt.aerial->key = "aerial";
    opt.aerial->type = TYPE_STRING;
    opt.aerial->required = NO;
    opt.aerial->label = _("Aerial imagery map(s)");
    opt.aerial->description =
        _("Aerial image map(s), bands separated by commas, dates by semicolons");
    opt.aerial->guisection = _("Input");

    opt.spot = G_define_option();
    opt.spot->key = "spot";
    opt.spot->type = TYPE_STRING;
    opt.spot->required = NO;
    opt.spot->label = _("SPOT imagery map(s)");
    opt.spot->description = _("SPOT image map(s), bands separated by commas, dates by semicolons");
    opt.spot->guisection = _("Input");

    opt.dem = G_define_option();
    opt.dem->key = "dem";
    opt.dem->type = TYPE_STRING;
    opt.dem->required = NO;
    opt.dem->label = _("Digital Elevation Model");
    opt.dem->description = _("DEM raster map(s), channels separated by commas, dates by semicolons");
    opt.dem->guisection = _("Input");

    opt.s2 = G_define_option();
    opt.s2->key = "s2";
    opt.s2->type = TYPE_STRING;
    opt.s2->required = NO;
    opt.s2->label = _("Sentinel-2 imagery");
    opt.s2->description = _("Sentinel-2 bands, separated by commas, dates by semicolons");
    opt.s2->guisection = _("Input");

    opt.s1 = G_define_option();
    opt.s1->key = "s1";
    opt.s1->type = TYPE_STRING;
    opt.s1->required = NO;
    opt.s1->label = _("Sentinel-1 imagery (s2-naip variant)");
    opt.s1->description = _("Sentinel-1 VV,VH bands, separated by commas, dates by semicolons");
    opt.s1->guisection = _("Input");

    opt.s1_asc = G_define_option();
    opt.s1_asc->key = "s1_asc";
    opt.s1_asc->type = TYPE_STRING;
    opt.s1_asc->required = NO;
    opt.s1_asc->label = _("Sentinel-1 ascending");
    opt.s1_asc->description = _("Sentinel-1 ascending VV,VH bands");
    opt.s1_asc->guisection = _("Input");

    opt.s1_des = G_define_option();
    opt.s1_des->key = "s1_des";
    opt.s1_des->type = TYPE_STRING;
    opt.s1_des->required = NO;
    opt.s1_des->label = _("Sentinel-1 descending");
    opt.s1_des->description = _("Sentinel-1 descending VV,VH bands");
    opt.s1_des->guisection = _("Input");

    /* Model options. */
    opt.model = G_define_option();
    opt.model->key = "model";
    opt.model->type = TYPE_STRING;
    opt.model->required = NO;
    opt.model->options = "flair-hub,s2-naip,forest,agriculture,urban";
    opt.model->answer = "flair-hub";
    opt.model->label = _("MAESTRO model variant");
    opt.model->description = _("Model variant to use");
    opt.model->guisection = _("Model");

    opt.weights_dir = G_define_option();
    opt.weights_dir->key = "weights_dir";
    opt.weights_dir->type = TYPE_STRING;
    opt.weights_dir->required = NO;
    opt.weights_dir->label = _("Weights directory");
    opt.weights_dir->description =
        _("Directory containing model checkpoints; default: "
          "$MAESTRO_MODELS_DIR/maestro or ~/models/maestro");
    opt.weights_dir->guisection = _("Model");

    opt.checkpoint = G_define_option();
    opt.checkpoint->key = "checkpoint";
    opt.checkpoint->type = TYPE_STRING;
    opt.checkpoint->required = NO;
    opt.checkpoint->label = _("Checkpoint file");
    opt.checkpoint->description =
        _("Explicit path to a .ckpt checkpoint file (overrides model/weights_dir)");
    opt.checkpoint->guisection = _("Model");

    /* Processing options. */
    opt.tile_meters = G_define_option();
    opt.tile_meters->key = "tile_meters";
    opt.tile_meters->type = TYPE_DOUBLE;
    opt.tile_meters->required = NO;
    opt.tile_meters->answer = "0";
    opt.tile_meters->label = _("Tile size in meters");
    opt.tile_meters->description =
        _("Tile size in meters (0 = use model's crop_meters)");
    opt.tile_meters->guisection = _("Processing");

    opt.batch = G_define_option();
    opt.batch->key = "batch";
    opt.batch->type = TYPE_INTEGER;
    opt.batch->answer = "1";
    opt.batch->options = "1-";
    opt.batch->label = _("Batch size");
    opt.batch->description = _("Number of tiles per batch (1 for now)");
    opt.batch->guisection = _("Processing");

    /* Output options. */
    opt.channels = G_define_option();
    opt.channels->key = "channels";
    opt.channels->type = TYPE_STRING;
    opt.channels->answer = "1-8";
    opt.channels->label = _("Output channels");
    opt.channels->description =
        _("Channels to write (1-8, range like 1-32, or 'all' for all 768)");
    opt.channels->guisection = _("Output");

    opt.dates = G_define_option();
    opt.dates->key = "dates";
    opt.dates->type = TYPE_STRING;
    opt.dates->required = NO;
    opt.dates->label = _("Date override");
    opt.dates->description =
        _("Comma-separated ISO dates (YYYY-MM-DD[THH]) to use instead of map timestamps");
    opt.dates->guisection = _("Input");

    opt.output = G_define_standard_option(G_OPT_R_OUTPUT);
    opt.output->description =
        _("Prefix for output raster maps (suffixes _modality.channel will be appended)");

    /* Device options. */
    opt.device = G_define_option();
    opt.device->key = "device";
    opt.device->type = TYPE_STRING;
    opt.device->options = "auto,gpu,cpu";
    opt.device->answer = "auto";
    opt.device->description = _("OpenCL device type");
    opt.device->guisection = _("Device");

    opt.platform = G_define_option();
    opt.platform->key = "platform";
    opt.platform->type = TYPE_STRING;
    opt.platform->required = NO;
    opt.platform->description =
        _("Use only OpenCL platforms whose name contains this text");
    opt.platform->guisection = _("Device");

    opt.profile = G_define_option();
    opt.profile->key = "profile";
    opt.profile->type = TYPE_STRING;
    opt.profile->options = "auto,cpu,igpu,dgpu,egpu";
    opt.profile->answer = "auto";
    opt.profile->description = _("Device profile");
    opt.profile->guisection = _("Device");

    opt.precision = G_define_option();
    opt.precision->key = "precision";
    opt.precision->type = TYPE_STRING;
    opt.precision->options = "fp32,fp16";
    opt.precision->answer = "fp32";
    opt.precision->description = _("Weight storage precision");
    opt.precision->guisection = _("Device");

    no_download = G_define_flag();
    no_download->key = 'n';
    no_download->description = _("Do not download missing checkpoints");
    no_download->guisection = _("Model");

    if (G_parser(argc, argv))
        exit(EXIT_FAILURE);

    /* Parse channels. */
    if (strcmp(opt.channels->answer, "all") == 0) {
        ch_start = 0;
        ch_end = MAESTRO_EMBED_DIM;
        nchannels_out = MAESTRO_EMBED_DIM;
    }
    else {
        parse_channels(opt.channels->answer, &ch_start, &ch_end, &nchannels_out);
    }

    /* Parse user dates. */
    user_dates = parse_user_dates(opt.dates->answer, &n_user_dates);

    /* Find variant. */
    {
        const struct maestro_variant *variants = maestro_variants();
        int v;
        for (v = 0; v < N_VARIANTS; v++) {
            if (strcmp(variants[v].name, opt.model->answer) == 0) {
                variant = &variants[v];
                break;
            }
        }
        if (!variant)
            G_fatal_error(_("Unknown model variant <%s>"), opt.model->answer);
    }

    /* Get checkpoint. */
    if (opt.checkpoint->answer) {
        checkpoint_path = G_store(opt.checkpoint->answer);
    }
    else {
        dir = opt.weights_dir->answer ? G_store(opt.weights_dir->answer)
                                      : default_weights_dir();
        checkpoint_path = find_checkpoint(dir, variant->name);
        if (!checkpoint_path) {
            if (no_download->answer)
                G_fatal_error(_("No checkpoint for %s in <%s> and -n given"),
                              variant->name, dir);
            G_message(_("Downloading %s checkpoint..."), variant->name);
            checkpoint_path = fetch_checkpoint(dir, variant);
        }
        G_free(dir);
    }
    G_message(_("Using checkpoint <%s>"), checkpoint_path);

    /* Determine tile size. */
    {
        double tile_m = atof(opt.tile_meters->answer);
        if (tile_m <= 0)
            tile_m = variant->crop_meters;
        G_get_window(&win);
        tile_rows = (int)ceil(tile_m / win.ns_res);
        tile_cols = (int)ceil(tile_m / win.ew_res);
    }

    /* Collect active modalities and their ndates. */
    /* For now, assume 1 date slot per modality. */
    /* TODO: Parse date slots from input options. */
    for (mi = 0; mi < variant->nmods; mi++) {
        const char *name = variant->mods[mi].name;
        int has_input = 0;

        if (strcmp(name, "aerial") == 0) has_input = (opt.aerial->answer != NULL);
        else if (strcmp(name, "spot") == 0) has_input = (opt.spot->answer != NULL);
        else if (strcmp(name, "dem") == 0) has_input = (opt.dem->answer != NULL);
        else if (strcmp(name, "s2") == 0) has_input = (opt.s2->answer != NULL);
        else if (strcmp(name, "s1") == 0) has_input = (opt.s1->answer != NULL);
        else if (strcmp(name, "s1_asc") == 0) has_input = (opt.s1_asc->answer != NULL);
        else if (strcmp(name, "s1_des") == 0) has_input = (opt.s1_des->answer != NULL);

        if (has_input) {
            mod_active[mi] = 1;
            ndates[mi] = 1; /* TODO: Parse actual date slots */
            nmods_active++;
        }
        else {
            mod_active[mi] = 0;
            ndates[mi] = 1; /* Set to 1 to avoid error in maestro_model_init */
        }
    }

    if (nmods_active == 0)
        G_fatal_error(_("No input modalities specified"));

    /* Plan progress. */
    progress_plan(PG_READ,
                  (double)Rast_window_rows() * Rast_window_cols() / 1e6);
    progress_plan(PG_BUILD, 1);
    progress_plan(PG_ENCODE,
                  (double)Rast_window_rows() * Rast_window_cols() /
                  (tile_rows * tile_cols) / 10);
    progress_plan(PG_WRITE, (double)nchannels_out * nmods_active / 10);

    /* Initialize OpenCL. */
    t0 = now();
    progress_begin(PG_BUILD, 1);
    ocl_init(&ocl, opt.device->answer, opt.platform->answer,
             opt.profile->answer, opt.precision->answer);
    progress_begin(PG_LOAD, 1);

    /* Initialize model. */
    maestro_model_init(&model_ctx, &ocl, variant, checkpoint_path,
                       ndates, atoi(opt.batch->answer));
    progress_end();
    G_free(checkpoint_path);

    t1 = now();
    G_message(_("Model ready in %.1f s"), t1 - t0);

    /* Create output raster files. */
    {
        char outname[GPATH_MAX];
        out = G_calloc(nchannels_out * nmods_active, sizeof(struct raster_out));
        for (mi = 0, oi = 0; mi < variant->nmods; mi++) {
            if (!mod_active[mi])
                continue;
            for (ch = ch_start; ch < ch_end; ch++) {
                snprintf(outname, sizeof(outname), "%s_%s.%d",
                         opt.output->answer, variant->mods[mi].name, ch + 1);
                raster_out_open(&out[oi++], outname, &win,
                                variant->mods[mi].grid, tile_cols);
            }
        }
    }

    /* Tile loop. */
    tile = win;
    tile.rows = tile_rows;
    tile.cols = tile_cols;

    /* For now, just report progress at tile boundaries */
    int total_tiles = ((Rast_window_rows() + tile_rows - 1) / tile_rows) *
                      ((Rast_window_cols() + tile_cols - 1) / tile_cols);
    int tile_count = 0;
    
    progress_begin(PG_ENCODE, total_tiles);

    for (tile_row = 0; tile_row < Rast_window_rows(); tile_row += tile_rows) {
        tile.north = win.north - tile_row * win.ns_res;
        tile.south = win.north - fmin(tile_row + tile_rows, Rast_window_rows()) * win.ns_res;
        tile.rows = (int)((tile.north - tile.south) / win.ns_res + 0.5);

        for (tile_col = 0; tile_col < Rast_window_cols(); tile_col += tile_cols) {
            tile.west = win.west + tile_col * win.ew_res;
            tile.east = win.west + fmin(tile_col + tile_cols, Rast_window_cols()) * win.ew_res;
            tile.cols = (int)((tile.east - tile.west) / win.ew_res + 0.5);

            tile_count++;
            G_message(_("Processing tile %d/%d at row %d, col %d (size %dx%d)"),
                      tile_count, total_tiles, tile_row, tile_col, tile.rows, tile.cols);
            progress_part(tile_count, total_tiles);

            /* Process each active modality for this tile. */
            /* Allocate arrays for input images and output embeddings. */
            /* Must allocate for all modalities in the variant, not just active ones,
             * because maestro_encode iterates over all m->nmods. */
            float **imgs = G_calloc(variant->nmods, sizeof(float *));
            float **embs = G_calloc(variant->nmods, sizeof(float *));
            struct maestro_date **mod_dates = G_calloc(variant->nmods, sizeof(struct maestro_date *));
            struct maestro_date ref_date = {0};
            int n_active = 0;
            
            /* For each modality in the variant, allocate buffers and read data if available. */
            for (mi = 0; mi < variant->nmods; mi++) {
                const struct maestro_mod *md = &variant->mods[mi];
                const char *opt_name = NULL;
                
                /* Determine which option this modality comes from. */
                if (strcmp(md->name, "aerial") == 0) opt_name = opt.aerial->answer;
                else if (strcmp(md->name, "spot") == 0) opt_name = opt.spot->answer;
                else if (strcmp(md->name, "dem") == 0) opt_name = opt.dem->answer;
                else if (strcmp(md->name, "s2") == 0) opt_name = opt.s2->answer;
                else if (strcmp(md->name, "s1") == 0) opt_name = opt.s1->answer;
                else if (strcmp(md->name, "s1_asc") == 0) opt_name = opt.s1_asc->answer;
                else if (strcmp(md->name, "s1_des") == 0) opt_name = opt.s1_des->answer;
                
                /* Allocate input buffer: (1, ndates, bands, image, image) */
                size_t img_size = (size_t)1 * md->ndates * md->bands * md->image * md->image;
                imgs[mi] = G_calloc(img_size, sizeof(float));
                
                /* Allocate output embedding buffer: (B, L, 768) */
                embs[mi] = G_calloc((size_t)md->L * MAESTRO_EMBED_DIM * sizeof(float), 1);
                
                /* Allocate and set dates. */
                mod_dates[mi] = G_malloc(md->ndates * sizeof(struct maestro_date));
                for (int d = 0; d < md->ndates; d++) {
                    mod_dates[mi][d].year = 2022;
                    mod_dates[mi][d].doy = 182; /* July 1 */
                    mod_dates[mi][d].hour = 0;
                }
                
                /* Reference date = first modality's first date. */
                if (n_active == 0) {
                    ref_date = mod_dates[mi][0];
                }
                n_active++;
                
                /* If modality is not active or has no input, fill with normalized zeros. */
                if (!mod_active[mi] || !opt_name || !*opt_name) {
                    if (!mod_active[mi])
                        G_warning(_("Modality %s is not active in this variant"), md->name);
                    else
                        G_warning(_("No input specified for modality %s"), md->name);
                    /* Fill with normalized zeros (0 after normalization) */
                    for (size_t j = 0; j < img_size; j++)
                        imgs[mi][j] = 0.0f;
                    continue;
                }
                
                /* Read the raster map for this modality at the tile bounds. */
                /* For now, use the first token as the map name. */
                char map_name[GNAME_MAX];
                char *comma = strchr(opt_name, ',');
                if (comma) {
                    strncpy(map_name, opt_name, comma - opt_name);
                    map_name[comma - opt_name] = '\0';
                } else {
                    strncpy(map_name, opt_name, sizeof(map_name) - 1);
                    map_name[sizeof(map_name) - 1] = '\0';
                }
                
                /* Read the raster map. raster_read_plane handles missing maps gracefully. */
                float *band_data = G_malloc(md->image * md->image * sizeof(float));
                raster_read_plane(map_name, "", &tile, md->image,
                                  md->norm_fac, md->log_scale, band_data);
                
                /* Copy to all bands in the input buffer. */
                for (int b = 0; b < md->bands; b++) {
                    for (int r = 0; r < md->image; r++) {
                        for (int c = 0; c < md->image; c++) {
                            size_t idx = b * md->image * md->image + r * md->image + c;
                            imgs[mi][idx] = band_data[r * md->image + c];
                        }
                    }
                }
                G_free(band_data);
            }
            
            /* If no active modalities, skip encoding. */
            if (n_active == 0) {
                for (int i = 0; i < nmods_active; i++) {
                    G_free(imgs[i]);
                    G_free(embs[i]);
                    G_free(mod_dates[i]);
                }
                G_free(imgs);
                G_free(embs);
                G_free(mod_dates);
                continue;
            }
            
            /* Set dates for the model. */
            maestro_set_dates(&model_ctx, (const struct maestro_date **)mod_dates, &ref_date);
            
            /* Encode. */
            maestro_encode(&model_ctx, (const float *const *)imgs, (float *const *)embs);
            
            /* Write output embeddings. */
            /* For now, skip output writing to debug encoding. */
            /* for (int ai = 0, mi = 0, oi = 0; mi < variant->nmods; mi++) { */
            /*     if (!mod_active[mi]) */
            /*         continue; */
            /*     const struct maestro_mod *md = &variant->mods[mi]; */
            /*     float *emb = embs[ai++]; */
            /*     for (ch = ch_start; ch < ch_end; ch++) { */
            /*         raster_out_add(&out[oi], tile_row / tile_rows, tile_col / tile_cols, */
            /*                        emb, MAESTRO_EMBED_DIM, ch); */
            /*         oi++; */
            /*     } */
            /* } */
            G_message(_("  Encoding successful for this tile"));
            
            /* Free temporary allocations for all modalities. */
            for (int i = 0; i < variant->nmods; i++) {
                G_free(imgs[i]);
                G_free(embs[i]);
                G_free(mod_dates[i]);
            }
            G_free(imgs);
            G_free(embs);
            G_free(mod_dates);
        }
    }

    /* End encode progress. */
    progress_end();

    /* Flush and close output. */
    for (oi = 0; oi < nchannels_out * nmods_active; oi++) {
        raster_out_close(&out[oi]);
    }
    G_free(out);

    /* Cleanup. */
    maestro_free(&model_ctx);
    ocl_free(&ocl);

    for (int i = 0; i < n_user_dates; i++)
        G_free(user_dates[i]);
    G_free(user_dates);

    progress_finish();
    G_message(_("Done"));

    exit(EXIT_SUCCESS);
}
