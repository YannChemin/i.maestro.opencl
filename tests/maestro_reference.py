#!/usr/bin/env python3
"""Reference implementation of the MAESTRO encoder forward pass.

Standalone re-implementation (no maestro/hydra imports) of the exact inference
path of IGNF/MAESTRO_{FLAIR-HUB,S2-NAIP-urban}_base checkpoints, used by the
tests of i.maestro.opencl to validate the C + OpenCL translation:

    raw inputs (dataset normalization)          -> resize_and_rescale (nearest)
    -> Patchify (Conv2d k=stride=patch + GroupNorm(1,768))
    -> + sincos pos encoding (block-mean of G x G grid)
    -> + date encoding ([zeros(760), diff*4, sin/cos doy, sin/cos hour])
    -> group_mods (concat per group, seq dim)
    -> encoder[group] (9 vit_pytorch Transformer layers + final LN)
    -> encoder_inter (3 layers + final LN) over concat of all groups
    -> ungroup -> per-modality embeddings (B, D, L, 768)

Only the encoder stack of the released pretrain checkpoints is usable (they
carry no decoder transformer and untrained heads), so the output is the
post-norm encoder embedding.

Usage:
    python3 tests/maestro_reference.py --variant s2naip|flair \
        [--ckpt PATH] [--outdir DIR] [--seed N] [--batch B]

Defaults read the local study copies in $HOME/dev/MAESTRO_weights and write
to /tmp/maestro_reference/<variant>/.
"""

from __future__ import annotations

import argparse
import json
import pickle
import types
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import nn
from vit_pytorch.vit import Transformer

EMBED_DIM = 768
DATE_DIM = 8
FAC_ABS_ENC = 1.0
FAC_DATE_ENC = 1.0
TEMPERATURE = 10000
INTER_DEPTH = 3
HEADS = 12
DIM_HEAD = 64
MLP_DIM = EMBED_DIM * 4

VARIANTS = {
    "s2naip": {
        "ckpt": "MAESTRO_S2-NAIP-urban_base/checkpoints/pretrain-epoch=14.ckpt",
        "grid_pos_enc": 192,
        "crop_meters": 120.0,
        "mods": {  # image, patch, bands, dates, norm_fac, log, group
            "aerial": dict(image=384, patch=16, bands=4, dates=1,
                           norm_fac=255.0, log=False, group="aerial"),
            "spot": dict(image=128, patch=16, bands=3, dates=1,
                         norm_fac=255.0, log=False, group="spot"),
            "s2": dict(image=12, patch=2, bands=10, dates=16,
                       norm_fac=5000.0, log=False, group="s2"),
            "s1": dict(image=12, patch=2, bands=2, dates=4,
                       norm_fac=20.0, log=False, group="s1"),
        },
        "mod_order": ["aerial", "spot", "s2", "s1"],
        "group_order": ["aerial", "spot", "s2", "s1"],
        "native": {"aerial": 372, "spot": 132, "s2": 14, "s1": 14},
        "raw_max": {"aerial": 255.0, "spot": 255.0, "s2": 10000.0, "s1": 20.0},
    },
    "flair": {
        "ckpt": "MAESTRO_FLAIR-HUB_base/checkpoints/pretrain-epoch=99.ckpt",
        "grid_pos_enc": 160,
        "crop_meters": 102.4,
        "mods": {
            "aerial": dict(image=512, patch=16, bands=4, dates=1,
                           norm_fac=255.0, log=False, group="aerial"),
            "spot": dict(image=128, patch=16, bands=3, dates=1,
                         norm_fac=2000.0, log=False, group="spot"),
            "dem": dict(image=512, patch=32, bands=2, dates=1,
                        norm_fac=1000.0, log=False, rescale_elev=True,
                        group="dem"),
            "s2": dict(image=10, patch=2, bands=10, dates=16,
                       norm_fac=5000.0, log=False, group="s2"),
            "s1_asc": dict(image=10, patch=2, bands=2, dates=4,
                           norm_fac=5.0, log=True, group="s1"),
            "s1_des": dict(image=10, patch=2, bands=2, dates=4,
                           norm_fac=5.0, log=True, group="s1"),
        },
        "mod_order": ["aerial", "spot", "dem", "s2", "s1_asc", "s1_des"],
        "group_order": ["aerial", "spot", "dem", "s2", "s1"],
        "native": {"aerial": 500, "spot": 132, "dem": 508, "s2": 14,
                   "s1_asc": 14, "s1_des": 14},
        "raw_max": {"aerial": 255.0, "spot": 2000.0, "dem": 300.0,
                    "s2": 5000.0, "s1_asc": 10.0, "s1_des": 10.0},
    },
}

# Deterministic test dates: (year, day of year, hour), as produced by
# maestro.dataset.utils.dates_numpy (date-only strings => hour 0).
REF_DATE = (2022, 182, 0)  # 2022-07-01, the aerial/spot/dem date
S2_DOYS = [15, 37, 59, 81, 104, 127, 151, 174, 197, 221, 244, 267, 290, 314,
           337, 359]
S1_DOYS = [74, 166, 258, 350]


class _Stub:
    """Swallow unpickled maestro/hydra objects we do not need."""

    def __init__(self, *args, **kwargs):
        pass

    def __setstate__(self, state):
        pass


class _StubUnpickler(pickle.Unpickler):
    SAFE = ("torch", "collections", "builtins", "_codecs", "numpy")

    def find_class(self, module, name):
        if module.split(".")[0] not in self.SAFE:
            return _Stub
        return super().find_class(module, name)


_stub_pickle = types.ModuleType("maestro_stub_pickle")
_stub_pickle.Unpickler = _StubUnpickler


def load_state_dict(ckpt_path):
    ck = torch.load(ckpt_path, map_location="cpu", pickle_module=_stub_pickle,
                    weights_only=False)
    sd = ck["state_dict"]
    return {k[len("model."):]: v for k, v in sd.items()
            if k.startswith("model.")}


def posemb_sincos_2d(h, w, dim=EMBED_DIM, date_dim=DATE_DIM,
                     temperature=TEMPERATURE):
    y, x = torch.meshgrid(torch.arange(h), torch.arange(w), indexing="ij")
    n = (dim - date_dim) // 4
    omega = torch.arange(n) / (n - 1)
    omega = 1.0 / (temperature ** omega)
    y = y[:, :, None] * omega[None, None, :]
    x = x[:, :, None] * omega[None, None, :]
    return torch.cat([x.sin(), x.cos(), y.sin(), y.cos(),
                      torch.zeros((h, w, date_dim))], dim=-1).float()


def reshape_encoding(encoding, grid_size):
    """Exact port of maestro.layers.utils.reshape_encoding."""
    encoding = encoding.reshape((1,) * (5 - encoding.ndim) + encoding.shape)
    encoding = encoding.expand(
        (*encoding.shape[:-3],
         max(grid_size, encoding.shape[-3]),
         max(grid_size, encoding.shape[-2]),
         encoding.shape[-1]))
    if encoding.shape[-3] % grid_size:
        b = encoding.shape[0]
        resize = grid_size * round(encoding.shape[-3] / grid_size)
        encoding = rearrange(encoding, "b d h w c -> (b d) c h w")
        encoding = F.interpolate(encoding, (resize, resize), mode="bilinear")
        encoding = rearrange(encoding, "(b d) c h w -> b d h w c", b=b)
    encoding = encoding.unflatten(-3, (grid_size, -1))
    encoding = encoding.unflatten(-2, (grid_size, -1))
    return encoding.mean(dim=(-2, -4)).flatten(-3, -2)


def encode_dates(dates, ref_date, grid_size, len_bands=1, dim=EMBED_DIM,
                 date_dim=DATE_DIM, fac_date_enc=FAC_DATE_ENC):
    """Exact port of maestro.layers.utils.encode_dates."""
    dates = dates.reshape(dates.shape + (1,) * (5 - dates.ndim))
    ref_date = ref_date.reshape(ref_date.shape + (1,) * (5 - ref_date.ndim))

    year = dates[:, :, 0]
    doy = dates[:, :, 1] / 365.25
    hour = dates[:, :, 2] / 24.0

    year_ref = ref_date[:, :, 0]
    doy_ref = ref_date[:, :, 1] / 365.25
    diff = (year + doy) - (year_ref + doy_ref)

    doy = 2 * np.pi * doy
    hour = 2 * np.pi * hour
    date_encoding = torch.stack(
        [diff, torch.sin(doy), torch.cos(doy), torch.sin(hour),
         torch.cos(hour)], dim=-1)
    date_encoding = reshape_encoding(date_encoding, grid_size=grid_size)
    date_encoding = date_encoding * fac_date_enc

    diff, date_encoding = torch.split(date_encoding, [1, 4], dim=-1)
    pad_zeros = torch.zeros_like(diff).expand(-1, -1, -1, dim - date_dim)
    pad_diff = diff.expand(-1, -1, -1, date_dim - 4)
    date_encoding = torch.cat([pad_zeros, pad_diff, date_encoding], dim=-1)

    if len_bands == 1:
        return date_encoding
    date_encoding = date_encoding.unflatten(1, (1, -1))
    date_encoding = date_encoding.expand(-1, len_bands, -1, -1, -1)
    return date_encoding.flatten(1, 2)


class PatchifyBands(nn.Module):
    def __init__(self, in_chans, patch_size):
        super().__init__()
        self.conv = nn.Conv2d(in_chans, EMBED_DIM, patch_size, patch_size)
        self.norm = nn.GroupNorm(1, EMBED_DIM)

    def forward(self, x):
        _, d, _, h, w = x.shape
        x = rearrange(x, "b d c h w -> (b d) c h w")
        x = self.conv(x)
        x = self.norm(x)
        return rearrange(x, "(b d) c h w -> b d (h w) c", d=d)


def group_mods(x, groups):
    """fusion_mode='group': per mod flatten(1,2), concat per group."""
    x = {m: t.flatten(1, 2) for m, t in x.items()}
    out = {}
    for m, g in groups:
        out.setdefault(g, []).append(x[m])
    return {g: torch.cat(v, dim=1) for g, v in out.items()}


def ungroup_mods(x_group, groups, num_dates, grid_size):
    out = {}
    group_mods_ = {}
    for m, g in groups:
        group_mods_.setdefault(g, []).append(m)
    for g, seq in x_group.items():
        splits = [num_dates[m] * grid_size[m] ** 2 for m in group_mods_[g]]
        for m, t in zip(group_mods_[g], torch.split(seq, splits, dim=1)):
            out[m] = t
    for m in out:
        out[m] = out[m].unflatten(1, (num_dates[m], -1))
    return out


def mod_dates(name, spec):
    if name in ("aerial", "spot", "dem"):
        return [(2022, REF_DATE[1], 0)]
    if name == "s2":
        return [(2022, d, 0) for d in S2_DOYS[: spec["dates"]]]
    return [(2022, d, 0) for d in S1_DOYS[: spec["dates"]]]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--variant", choices=sorted(VARIANTS), required=True)
    ap.add_argument("--ckpt", default=None)
    ap.add_argument("--outdir", default=None)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch", type=int, default=1)
    args = ap.parse_args()
    torch.set_grad_enabled(False)

    v = VARIANTS[args.variant]
    ckpt = args.ckpt or str(Path.home() / "dev/MAESTRO_weights" / v["ckpt"])
    outdir = Path(args.outdir or f"/tmp/maestro_reference/{args.variant}")
    outdir.mkdir(parents=True, exist_ok=True)

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    sd = load_state_dict(ckpt)

    mods = v["mods"]
    grids = {m: spec["image"] // spec["patch"] for m, spec in mods.items()}
    groups = [(m, spec["group"]) for m, spec in mods.items()]
    num_dates = {m: spec["dates"] for m, spec in mods.items()}
    len_bands = {m: 1 for m in mods}

    # checkpoint-loaded modules
    patch_embed = {}
    for m, spec in mods.items():
        pe = PatchifyBands(spec["bands"], spec["patch"])
        prefix = f"patch_embed.{m}.patchify_bands.0."
        pe.load_state_dict({k[len(prefix):]: sd[k] for k in sd
                            if k.startswith(prefix)}, strict=True)
        patch_embed[m] = pe

    encoders = {}
    for g in v["group_order"]:
        enc = Transformer(dim=EMBED_DIM, depth=DEPTH_LOCAL, heads=HEADS,
                          dim_head=DIM_HEAD, mlp_dim=MLP_DIM)
        prefix = f"encoder.{g}."
        enc.load_state_dict({k[len(prefix):]: sd[k] for k in sd
                             if k.startswith(prefix)}, strict=True)
        encoders[g] = enc
    enc_inter = Transformer(dim=EMBED_DIM, depth=INTER_DEPTH, heads=HEADS,
                            dim_head=DIM_HEAD, mlp_dim=MLP_DIM)
    prefix = "encoder_inter."
    enc_inter.load_state_dict({k[len(prefix):]: sd[k] for k in sd
                               if k.startswith(prefix)}, strict=True)

    # encodings (non-persistent buffers in the original)
    pos_grid = posemb_sincos_2d(v["grid_pos_enc"], v["grid_pos_enc"]) \
        * FAC_ABS_ENC

    # ---- synthetic raw inputs, dataset normalization (dataset.py order) ----
    batch = {}
    dates = {}
    for m in v["mod_order"]:
        spec = mods[m]
        d, c, n = spec["dates"], spec["bands"], v["native"][m]
        raw = rng.uniform(0.0, v["raw_max"][m],
                          size=(args.batch, d, c, n, n)).astype(np.float32)
        x = torch.from_numpy(raw)
        if spec.get("log"):
            x = torch.log(torch.clamp(x, min=1e-10))
        x = x / spec["norm_fac"]
        batch[m] = x
        dates[m] = torch.tensor([mod_dates(m, spec)] * args.batch)
        np.save(outdir / f"raw_{m}.npy", raw)
        np.save(outdir / f"dates_{m}.npy",
                np.array([mod_dates(m, spec)], dtype=np.int64))
    ref_date = torch.tensor([[REF_DATE]] * args.batch)
    np.save(outdir / "ref_date.npy", np.array([REF_DATE], dtype=np.int64))

    # ---- model forward: resize_and_rescale ----
    for m in v["mod_order"]:
        spec = mods[m]
        b = batch[m]
        b = F.interpolate(b.flatten(0, 1), size=(spec["image"],) * 2,
                          mode="nearest").unflatten(0, (-1, spec["dates"]))
        if spec.get("rescale_elev"):
            b[:, :, 1:] = 30 * (b[:, :, :1] - b[:, :, 1:])
        batch[m] = b
        np.save(outdir / f"resized_{m}.npy", b.numpy())

    # ---- embed (Patchify) ----
    x_enc = {}
    for m in v["mod_order"]:
        x_enc[m] = patch_embed[m](batch[m])
        np.save(outdir / f"patch_{m}.npy", x_enc[m].numpy())

    # ---- enc_add_encodings ----
    for m in v["mod_order"]:
        pos = reshape_encoding(pos_grid, grids[m])           # (1,1,L,C)
        date = encode_dates(dates[m], ref_date, grids[m],
                            len_bands=len_bands[m])           # (B,D,L,C)
        np.save(outdir / f"posenc_{m}.npy", pos[0, 0].numpy())
        np.save(outdir / f"dateenc_{m}.npy", date.numpy())
        x_enc[m] = x_enc[m] + pos + date

    x_group = group_mods(x_enc, groups)
    for g, t in x_group.items():
        np.save(outdir / f"xgroup_{g}.npy", t.numpy())

    # ---- encode: per-group encoders ----
    for g in v["group_order"]:
        x_group[g] = encoders[g](x_group[g])
        np.save(outdir / f"encout_{g}.npy", x_group[g].numpy())

    # ---- encoder_inter over the concatenation of all groups ----
    order = v["group_order"]
    splits = [x_group[g].shape[1] for g in order]
    inter_in = torch.cat([x_group[g] for g in order], dim=1)
    np.save(outdir / "inter_in.npy", inter_in.numpy())
    inter_out = enc_inter(inter_in)
    np.save(outdir / "inter_out.npy", inter_out.numpy())
    for g, t in zip(order, inter_out.split(splits, dim=1)):
        x_group[g] = t

    # ---- ungroup: per-modality embeddings ----
    emb = ungroup_mods(x_group, groups, num_dates, grids)
    for m in v["mod_order"]:
        np.save(outdir / f"emb_{m}.npy", emb[m].numpy())
        e = emb[m].numpy()
        print(f"emb_{m}: shape {e.shape} mean {e.mean():+.6f} "
              f"std {e.std():.6f} absmax {np.abs(e).max():.6f}")

    meta = {
        "variant": args.variant,
        "ckpt": ckpt,
        "seed": args.seed,
        "batch": args.batch,
        "grid_pos_enc": v["grid_pos_enc"],
        "crop_meters": v["crop_meters"],
        "mod_order": v["mod_order"],
        "group_order": v["group_order"],
        "mods": {m: dict(spec, grid=grids[m]) for m, spec in mods.items()},
        "groups": groups,
        "ref_date": list(REF_DATE),
    }
    (outdir / "meta.json").write_text(json.dumps(meta, indent=2))
    print(f"reference written to {outdir}")


DEPTH_LOCAL = 9  # depth 12 - inter_depth 3, per-group encoder layers


if __name__ == "__main__":
    main()
