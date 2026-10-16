"""Generic single-case runner with seed support (infer.py has no --seed flag)."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, "/content/UniForge3D")

from uniforge3d import UniForge3DImageTo3DPipeline


def _install_rembg_model_override() -> None:
    """The xibushu snapshot's own pipeline.json names the GATED
    briaai/RMBG-2.0, which cannot be fetched offline. The baked public
    camenduru/RMBG-2.0 mirror has identical weights, so force it here."""
    from uniforge3d.utils import preprocessing as pp

    orig_init = pp.BackgroundRemover.__init__

    def init(self, checkpoint, device, **kwargs):
        return orig_init(self, "camenduru/RMBG-2.0", device, **kwargs)

    pp.BackgroundRemover.__init__ = init


def _install_skip_rembg_guard() -> None:
    """Make --skip-rembg fail clearly on opaque inputs instead of a
    'NoneType' error, and leave the pipeline untouched otherwise.

    Importing the class here would shadow the module-level name inside the
    caller, so reach it through the module object.
    """
    import numpy as np
    from PIL import Image
    from uniforge3d.pipelines import uniforge3d_image_to_3d as pipe_mod

    pipeline_cls = pipe_mod.UniForge3DImageTo3DPipeline
    orig_preprocess = pipeline_cls._preprocess_images

    def preprocess_images(self, images):
        for path in images:
            path = Path(path)
            with Image.open(path) as handle:
                alpha = np.asarray(handle.convert("RGBA"))[..., 3]
            if bool(np.all(alpha == 255)):
                raise RuntimeError(
                    f"{path.name} has no transparency. Background removal "
                    "was skipped (--skip-rembg), which only accepts "
                    "transparent RGBA PNGs. Upload transparent images, or "
                    "rerun without --skip-rembg to remove the background "
                    "with RMBG-2.0."
                )
        return orig_preprocess(self, images)

    pipeline_cls._preprocess_images = preprocess_images


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", default="xibushu/UniForge3D")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument("--rembg-model", default="camenduru/RMBG-2.0",
                        help="Background-removal checkpoint (public mirror, no login needed).")
    parser.add_argument("--skip-rembg", action="store_true",
                        help="Do not load RMBG-2.0 at all. Only valid for transparent "
                             "(RGBA, non-opaque alpha) inputs; errors clearly otherwise.")
    args = parser.parse_args()

    if args.skip_rembg:
        _install_skip_rembg_guard()
    else:
        _install_rembg_model_override()

    pipeline = UniForge3DImageTo3DPipeline.from_pretrained(
        args.model, device=args.device, low_vram=args.low_vram
    )
    result = pipeline.run(args.images, args.output, seed=args.seed, overwrite=args.overwrite)
    print("Saved %s" % result, flush=True)


if __name__ == "__main__":
    main()
