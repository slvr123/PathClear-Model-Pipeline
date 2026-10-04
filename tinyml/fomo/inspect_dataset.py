"""Validate annotation mistakes and save original-frame box/center previews."""
import argparse
import json
import hashlib
from itertools import zip_longest

from PIL import Image, ImageDraw

from config import CLASS_NAMES, SETTINGS, assert_branch, output_directory
from dataset import inspect_manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True, help="JSON manifest; image paths relative to it")
    parser.add_argument("--samples", type=int, default=6)
    parser.add_argument("--verify-sources", action="store_true", help="Re-read all original pixels and compare prepared inputs")
    parser.add_argument("--review-note", help="Record completed visual review after opening the generated previews")
    parser.add_argument("--output", help="New directory inside tinyml/fomo/artifacts")
    args = parser.parse_args()
    if args.samples < 0:
        parser.error("--samples must be nonnegative")
    _, records, report = inspect_manifest(args.manifest, args.verify_sources)
    directory = output_directory(args.output, "inspection")
    (directory / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    # Round-robin splits so previews include held-out data when it exists.
    buckets = [[r for r in records if r["split"] == s] for s in ("train", "validation", "test")]
    # Spread previews across cities/scenes rather than only use the first records.
    import numpy as np
    buckets = [[b[i] for i in np.linspace(0, len(b)-1, min(len(b), args.samples), dtype=int)] if b else [] for b in buckets]
    ordered = [r for row in zip_longest(*buckets) for r in row if r]
    for index, record in enumerate(ordered[:args.samples]):
        with Image.open(record["path"]) as source:
            image = source.convert("RGB")
        scale = min(1.0, 1024/image.width)
        image = image.resize((round(image.width*scale), round(image.height*scale)))
        draw = ImageDraw.Draw(image)
        for annotation in record["annotations"]:
            x, y, w, h = [v*scale for v in annotation["bbox"]]
            color = "lime" if annotation["class_id"] == 0 else "cyan"
            draw.rectangle((x, y, x + w, y + h), outline=color, width=2)
            cx, cy = x + w / 2, y + h / 2
            draw.ellipse((cx - 3, cy - 3, cx + 3, cy + 3), fill="red")
            draw.text((x, max(0,y-12)), CLASS_NAMES[annotation["class_id"]].upper(), fill=color)
        image.save(directory / f"sample-{index}-{record['split']}.png")
    print(json.dumps(report, indent=2))
    print(f"Previews and report: {directory}")
    print("Counts describe labels, not model quality. Red dots are box centers.")
    if args.review_note:
        if report["errors"]:
            raise ValueError("Cannot approve an invalid dataset")
        from pathlib import Path
        manifest_path = Path(args.manifest).resolve()
        assert_branch()
        receipt = {"manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
                   "valid": True, "visual_review_note": args.review_note, "inspection": str(directory),
                   "source_pixels_reverified": args.verify_sources}
        receipt_path = manifest_path.parent / "review.json"
        # This generated receipt is created once, never used to overwrite a previous review.
        with receipt_path.open("x", encoding="utf-8") as handle:
            json.dump(receipt, handle, indent=2)
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
