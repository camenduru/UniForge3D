import argparse
from pathlib import Path
from uniforge3d import UniForge3DImageTo3DPipeline

ROOT = Path(__file__).resolve().parent
IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".tif", ".tiff"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=ROOT / "assets")
    parser.add_argument("--output", type=Path, default=ROOT / "outputs")
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--model", default="xibushu/UniForge3D")
    source.add_argument("--config", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--low-vram", action=argparse.BooleanOptionalAction, default=True)
    args = parser.parse_args()

    cases = sorted(path for path in args.input.iterdir() if path.is_dir())
    cases = [case for case in cases if args.overwrite or not (args.output / case.name / "prediction.glb").is_file()]
    if not cases:
        return

    pipeline = (
        UniForge3DImageTo3DPipeline.from_config(args.config, low_vram=args.low_vram)
        if args.config else
        UniForge3DImageTo3DPipeline.from_pretrained(args.model, low_vram=args.low_vram)
    )
    for case in cases:
        images = sorted(path for path in case.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
        pipeline.run(images, args.output / case.name / "prediction.glb", overwrite=args.overwrite)


if __name__ == "__main__":
    main()
