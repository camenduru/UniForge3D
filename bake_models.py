"""Download every Hub artifact UniForge3D needs at inference time.

Mirrors exactly what the code fetches (see uniforge3d/models/loaders.py and
uniforge3d/pipelines/uniforge3d_image_to_3d.py), so a container baked with
this script runs with HF_HUB_OFFLINE=1 and never hits the network.

Background removal uses the public camenduru/RMBG-2.0 mirror (same weights
as the gated briaai/RMBG-2.0) — no token needed.

  python bake_models.py
"""
import argparse

from pathlib import Path

from huggingface_hub import hf_hub_download, snapshot_download
from huggingface_hub.constants import HF_HUB_CACHE as _DEFAULT_CACHE
import os
from huggingface_hub.constants import HF_HUB_CACHE as _DEFAULT_CACHE
import os


def _hub_root() -> str:
    return os.environ.get("HF_HUB_CACHE", _DEFAULT_CACHE)


def _cached_snapshot(repo_id: str, revision: str | None = None) -> bool:
    """True when a complete snapshot for repo[/revision] is already cached
    (e.g. pre-seeded via COPY for gated repos) — skip the Hub call entirely."""
    d = os.path.join(_hub_root(), "models--" + repo_id.replace("/", "--"), "snapshots")
    if not os.path.isdir(d):
        return False
    snaps = os.listdir(d)
    if revision:
        return any(s.startswith(revision) or revision.startswith(s) for s in snaps)
    return len(snaps) > 0


UNI_REV = None  # xibushu/UniForge3D: default branch (pinned by Hub, immutable .pt files)
T2_4B_REV = "af44b45f2e35a493886929c6d786e563ec68364d"
TIMG_REV = "25e0d31ffbebe4b5a97464dd851910efc3002d96"


def bake():
    print("== xibushu/UniForge3D (pipeline + stage weights)", flush=True)
    snapshot_download(
        "xibushu/UniForge3D",
        allow_patterns=["pipeline.json", "uniforge3d/**/*.pt"],
    )

    print("== microsoft/TRELLIS.2-4B (DiT base + shape/texture decoders)", flush=True)
    t2b = dict(repo_id="microsoft/TRELLIS.2-4B", revision=T2_4B_REV)
    hf_hub_download(filename="ckpts/ss_flow_img_dit_1_3B_64_bf16.safetensors", **t2b)
    for dec in ("ckpts/shape_dec_next_dc_f16c32_fp16",
                "ckpts/tex_dec_next_dc_f16c32_fp16"):
        # loaders.py resolves decoder checkpoints as <name>.json + <name>.safetensors
        snapshot_download(allow_patterns=[f"{dec}.json", f"{dec}.safetensors"], **t2b)

    print("== microsoft/TRELLIS-image-large (sparse-structure decoder)", flush=True)
    snapshot_download(
        repo_id="microsoft/TRELLIS-image-large", revision=TIMG_REV,
        allow_patterns=["ckpts/ss_dec_conv3d_16l8_fp16.json",
                        "ckpts/ss_dec_conv3d_16l8_fp16.safetensors"],
    )

    print("== facebook/dinov3-vitl16-pretrain-lvd1689m (frozen encoder)", flush=True)
    if _cached_snapshot("facebook/dinov3-vitl16-pretrain-lvd1689m"):
        print("-- already cached (pre-seeded, gated repo), skipping download", flush=True)
    else:
        snapshot_download("facebook/dinov3-vitl16-pretrain-lvd1689m")

    print("== camenduru/RMBG-2.0 (background removal, public mirror)", flush=True)
    snapshot_download("camenduru/RMBG-2.0")

    print("BAKE_DONE", flush=True)


if __name__ == "__main__":
    bake()
