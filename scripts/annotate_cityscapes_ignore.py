"""Review Cityscapes images and draw polygons to exclude from MicroSeg loss."""
import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from dataset_config import CITYSCAPES_ROOT
from microseg.dataset import collect_cityscapes_pairs

_DEFAULT_IGNORE_ROOT = CITYSCAPES_ROOT / "ignore_masks"
_MAX_CANVAS_SIZE = (1280, 760)


def ignore_mask_path(image_path, ignore_root):
    """Return the reviewed-mask path mirroring the Cityscapes image folders."""
    image_path = Path(image_path)
    image_root = CITYSCAPES_ROOT / "leftImg8bit"
    try:
        relative = image_path.relative_to(image_root)
    except ValueError as error:
        raise ValueError(
            f"Cityscapes image must be under {image_root}: {image_path}"
        ) from error
    stem = image_path.stem.removesuffix("_leftImg8bit")
    return Path(ignore_root) / relative.parent / f"{stem}_ignore.png"


def rasterize_polygons(size, polygons):
    """Create a binary 0/255 mask from polygons in source-image coordinates."""
    mask = Image.new("L", size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        if len(polygon) < 3:
            raise ValueError("Ignore polygons must contain at least three points.")
        draw.polygon(polygon, fill=255)
    return mask


def save_reviewed_mask(path, mask):
    """Atomically save a reviewed 8-bit binary PNG mask."""
    array = np.asarray(mask)
    if array.ndim != 2 or not np.isin(array, (0, 255)).all():
        raise ValueError("Reviewed ignore masks must be 2D and contain only 0 or 255.")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.stem + ".pending.png")
    try:
        Image.fromarray(array.astype(np.uint8), mode="L").save(temporary, format="PNG")
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def review_state_path(candidate_path):
    return Path(candidate_path).with_suffix(".review.json")


def load_review_state(candidate_path, split, root):
    """Read progress tied to the exact selector report, if it already exists."""
    candidate_path = Path(candidate_path)
    state_path = review_state_path(candidate_path)
    if not state_path.is_file():
        return state_path, {}
    state = json.loads(state_path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(candidate_path.read_bytes()).hexdigest()
    if (
        state.get("schema_version") != 1
        or state.get("split") != split
        or Path(state.get("dataset_root", "")).resolve() != Path(root).resolve()
        or state.get("candidate_sha256") != expected_hash
        or not isinstance(state.get("reviews"), dict)
    ):
        raise ValueError(
            f"Review progress does not match the candidate report: {state_path}"
        )
    if any(
        not isinstance(name, str)
        or not isinstance(record, dict)
        or record.get("status") not in {"confirmed", "rejected"}
        for name, record in state["reviews"].items()
    ):
        raise ValueError(f"Malformed candidate review progress: {state_path}")
    return state_path, state["reviews"]


def save_review_state(candidate_path, split, root, state_path, reviews):
    candidate_bytes = Path(candidate_path).read_bytes()
    state = {
        "schema_version": 1,
        "dataset_root": str(Path(root).resolve()),
        "split": split,
        "candidate_sha256": hashlib.sha256(candidate_bytes).hexdigest(),
        "reviewed_count": len(reviews),
        "confirmed_count": sum(
            record["status"] == "confirmed" for record in reviews.values()
        ),
        "rejected_count": sum(
            record["status"] == "rejected" for record in reviews.values()
        ),
        "reviews": reviews,
    }
    temporary = state_path.with_name(state_path.stem + ".pending.json")
    try:
        temporary.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
        temporary.replace(state_path)
    finally:
        temporary.unlink(missing_ok=True)


class IgnoreMaskAnnotator:
    def __init__(
        self,
        pairs,
        ignore_root,
        split,
        start_index=0,
        scores=None,
        review_callback=None,
    ):
        import tkinter as tk
        from tkinter import messagebox
        from PIL import ImageTk

        self.tk = tk
        self.messagebox = messagebox
        self.image_tk = ImageTk
        self.pairs = pairs
        self.ignore_root = ignore_root
        self.split = split
        self.index = start_index
        self.scores = scores or {}
        self.review_callback = review_callback
        self.base_mask = None
        self.polygons = []
        self.active_polygon = []
        self.dirty = False
        self.current_reviewed = False
        self.scale = 1.0
        self.photo = None
        self.reviewed_count = sum(
            ignore_mask_path(image_path, ignore_root).is_file()
            for image_path, _ in pairs
        )

        self.window = self.tk.Tk()
        self.window.title(f"Cityscapes ignore-mask review — {split}")
        self.status = self.tk.StringVar()
        self.tk.Label(
            self.window,
            textvariable=self.status,
            anchor="w",
            justify="left",
            padx=8,
            pady=6,
        ).pack(fill="x")
        self.canvas = self.tk.Canvas(
            self.window,
            width=_MAX_CANVAS_SIZE[0],
            height=_MAX_CANVAS_SIZE[1],
            background="#202020",
            highlightthickness=0,
        )
        self.canvas.pack(fill="both", expand=True)
        self.canvas.bind("<Button-1>", self.add_point)
        self.canvas.bind("<Button-3>", lambda _event: self.finish_polygon())
        self.canvas.bind("<Configure>", self.on_resize)
        self.window.bind("<Return>", lambda _event: self.finish_polygon())
        self.window.bind("<Control-s>", lambda _event: self.save_current())

        controls = self.tk.Frame(self.window, padx=8, pady=8)
        controls.pack(fill="x")
        self.tk.Button(controls, text="Previous", command=self.previous).pack(side="left")
        self.tk.Button(controls, text="Skip / next", command=self.next).pack(
            side="left", padx=(6, 0)
        )
        self.tk.Button(controls, text="Undo polygon", command=self.undo_polygon).pack(
            side="left", padx=(6, 0)
        )
        self.tk.Button(controls, text="Clear mask", command=self.clear_mask).pack(
            side="left", padx=(6, 0)
        )
        self.tk.Button(controls, text="Save positive mask", command=self.save_current).pack(
            side="left", padx=(6, 0)
        )
        self.tk.Button(controls, text="Save & next", command=self.save_and_next).pack(
            side="left", padx=(6, 0)
        )
        self.tk.Label(
            controls,
            text="Click polygon points; right-click or Enter to finish. Skip images with no region.",
            anchor="e",
        ).pack(side="right", fill="x", expand=True, padx=(12, 0))

        self.show_current()

    def output_path(self):
        return ignore_mask_path(self.pairs[self.index][0], self.ignore_root)

    def show_current(self):
        image_path, _ = self.pairs[self.index]
        image = Image.open(image_path).convert("RGB")
        output_path = self.output_path()
        if output_path.is_file():
            with Image.open(output_path) as saved:
                saved = saved.convert("L")
                if saved.size != image.size:
                    raise ValueError(
                        f"Existing mask dimensions {saved.size} do not match "
                        f"image dimensions {image.size}: {output_path}"
                    )
                values = np.asarray(saved)
                if not np.isin(values, (0, 255)).all():
                    raise ValueError(
                        f"Existing mask must contain only 0 or 255: {output_path}"
                    )
                self.base_mask = values.copy()
        else:
            self.base_mask = np.zeros((image.height, image.width), dtype=np.uint8)
        self.original = image
        self.polygons = []
        self.active_polygon = []
        self.dirty = False
        self.current_reviewed = False
        reviewed_status = "POSITIVE MASK" if output_path.is_file() else "NO MASK"
        relative_image = Path(image_path).relative_to(CITYSCAPES_ROOT).as_posix()
        score = self.scores.get(relative_image)
        score_text = f" — candidate score {score:.3f}" if score is not None else ""
        self.status.set(
            f"{self.split} image {self.index + 1}/{len(self.pairs)} "
            f"({self.reviewed_count} mask files present) — "
            f"{reviewed_status}{score_text}\n"
            f"{image_path}\n"
            "Draw over the full visible crossing footprint "
            "that conflicts with Cityscapes road labels."
        )
        self.render()

    def render(self):
        if not hasattr(self, "original"):
            return
        mask = self.base_mask.copy()
        if self.polygons:
            additions = np.asarray(
                rasterize_polygons(
                    self.original.size,
                    self.polygons,
                )
            )
            mask = np.maximum(mask, additions)

        overlay = Image.new("RGB", self.original.size, (255, 40, 40))
        composite = Image.composite(
            overlay,
            self.original,
            Image.fromarray(mask).point(lambda value: 100 if value else 0),
        )
        width = max(1, self.canvas.winfo_width())
        height = max(1, self.canvas.winfo_height())
        self.scale = min(
            width / self.original.width,
            height / self.original.height,
            1.0,
        )
        size = (
            max(1, round(self.original.width * self.scale)),
            max(1, round(self.original.height * self.scale)),
        )
        preview = composite.resize(size, Image.Resampling.LANCZOS)
        self.photo = self.image_tk.PhotoImage(preview)
        self.canvas.delete("all")
        x_offset = (width - size[0]) // 2
        y_offset = (height - size[1]) // 2
        self.canvas_image_origin = (x_offset, y_offset)
        self.canvas.create_image(x_offset, y_offset, image=self.photo, anchor="nw")
        if self.active_polygon:
            points = [
                (
                    x_offset + x * self.scale,
                    y_offset + y * self.scale,
                )
                for x, y in self.active_polygon
            ]
            if len(points) > 1:
                flattened = [coordinate for point in points for coordinate in point]
                self.canvas.create_line(*flattened, fill="#00ffff", width=2)
            for x, y in points:
                self.canvas.create_oval(
                    x - 3, y - 3, x + 3, y + 3,
                    fill="#00ffff", outline="#003333",
                )

    def on_resize(self, _event):
        self.render()

    def add_point(self, event):
        x_offset, y_offset = self.canvas_image_origin
        x = round((event.x - x_offset) / self.scale)
        y = round((event.y - y_offset) / self.scale)
        if 0 <= x < self.original.width and 0 <= y < self.original.height:
            self.active_polygon.append((x, y))
            self.dirty = True
            self.render()

    def finish_polygon(self):
        if len(self.active_polygon) < 3:
            if self.active_polygon:
                self.messagebox.showinfo(
                    "Polygon needs more points",
                    "Add at least three points before finishing a polygon.",
                    parent=self.window,
                )
            return
        self.polygons.append(self.active_polygon)
        self.active_polygon = []
        self.render()

    def undo_polygon(self):
        if self.active_polygon:
            self.active_polygon.pop()
        elif self.polygons:
            self.polygons.pop()
        else:
            return
        self.dirty = True
        self.render()

    def clear_mask(self):
        self.base_mask.fill(0)
        self.polygons = []
        self.active_polygon = []
        self.dirty = True
        self.render()

    def save_current(self):
        if self.active_polygon:
            self.messagebox.showwarning(
                "Finish polygon",
                "Finish or undo the active polygon before saving.",
                parent=self.window,
            )
            return False
        mask = self.base_mask.copy()
        if self.polygons:
            mask = np.maximum(
                mask,
                np.asarray(rasterize_polygons(self.original.size, self.polygons)),
            )
        output_path = self.output_path()
        was_reviewed = output_path.is_file()
        if not np.any(mask):
            if was_reviewed:
                output_path.unlink()
                self.reviewed_count -= 1
                self.mark_review("rejected")
                self.base_mask = mask
                self.polygons = []
                self.dirty = False
                self.show_current()
                self.current_reviewed = True
                return True
            self.messagebox.showinfo(
                "No ignore region",
                "No mask is needed for this image. Use Skip / next to continue.",
                parent=self.window,
            )
            return False
        save_reviewed_mask(output_path, mask)
        if not was_reviewed:
            self.reviewed_count += 1
        self.mark_review("confirmed")
        self.base_mask = mask
        self.polygons = []
        self.dirty = False
        self.show_current()
        self.current_reviewed = True
        return True

    def save_and_next(self):
        if self.save_current():
            self.next()

    def move(self, offset):
        if self.dirty:
            self.messagebox.showwarning(
                "Unsaved annotation",
                "Save the reviewed mask or undo/clear the changes before moving on.",
                parent=self.window,
            )
            return
        next_index = self.index + offset
        if offset > 0 and not self.current_reviewed:
            status = (
                "confirmed"
                if self.output_path().is_file() and np.any(self.base_mask)
                else "rejected"
            )
            self.mark_review(status)
        if 0 <= next_index < len(self.pairs):
            self.index = next_index
            self.show_current()
        elif offset > 0:
            self.status.set(
                f"{self.split} candidate review queue complete "
                f"({len(self.pairs)} images). Progress is saved."
            )

    def mark_review(self, status):
        if self.review_callback is not None:
            self.review_callback(self.pairs[self.index][0], status)
        self.current_reviewed = True

    def next(self):
        self.move(1)

    def previous(self):
        self.move(-1)

    def run(self):
        self.window.mainloop()


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--split", choices=("train", "val"), required=True)
    parser.add_argument(
        "--ignore-root",
        type=Path,
        default=_DEFAULT_IGNORE_ROOT,
        help=f"Output root (default: {_DEFAULT_IGNORE_ROOT})",
    )
    parser.add_argument(
        "--start-index",
        type=int,
        help="Zero-based index in the selected review queue",
    )
    parser.add_argument(
        "--candidates-json",
        type=Path,
        help="Review only images in a candidate JSON created by the selector",
    )
    parser.add_argument(
        "--review-rejected-sample",
        type=int,
        default=0,
        help="Instead review this many deterministically sampled non-candidates",
    )
    parser.add_argument("--seed", type=int, default=42)
    return parser.parse_args(argv)


def load_candidate_pairs(
    candidate_path,
    split,
    root,
    cityscapes_pairs,
    reviewed=None,
    rejected_sample_size=0,
    seed=42,
):
    """Resolve a selector report into checked Cityscapes image/mask pairs."""
    report = json.loads(candidate_path.read_text(encoding="utf-8"))
    if report.get("schema_version") != 1 or report.get("split") != split:
        raise ValueError(
            f"Candidate report must be schema version 1 for split {split!r}: "
            f"{candidate_path}"
        )
    if Path(report.get("dataset_root", "")).resolve() != Path(root).resolve():
        raise ValueError(
            f"Candidate report root does not match Cityscapes root {root}."
        )
    by_image = {
        Path(image).relative_to(root).as_posix(): (image, mask)
        for image, mask in cityscapes_pairs
    }
    candidates = report.get("candidates")
    reviewed = reviewed or {}
    if not isinstance(candidates, list) or not candidates:
        if rejected_sample_size == 0:
            raise ValueError(f"Candidate report has no candidates: {candidate_path}")
        candidates = []
    selected = []
    scores = {}
    seen = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or not isinstance(candidate.get("image"), str):
            raise ValueError(f"Malformed candidate record: {candidate!r}")
        relative = Path(candidate["image"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError(f"Candidate image must be relative to {root}: {candidate!r}")
        key = relative.as_posix()
        if key in seen:
            raise ValueError(f"Duplicate candidate image: {key}")
        if key not in by_image:
            raise ValueError(f"Candidate image is not in Cityscapes {split}: {key}")
        seen.add(key)
        if candidate.get("candidate", True):
            if key not in reviewed:
                selected.append(by_image[key])
        score = candidate.get("score")
        if isinstance(score, (int, float)):
            scores[key] = float(score)
    if rejected_sample_size:
        scored_images = report.get("scored_images")
        if not isinstance(scored_images, list):
            raise ValueError(
                "Candidate report does not include all image scores needed to "
                "sample rejected images."
            )
        candidates = [
            item for item in scored_images
            if (
                isinstance(item, dict)
                and not item.get("candidate", False)
                and item.get("image") not in reviewed
            )
        ]
        if rejected_sample_size > len(candidates):
            raise ValueError(
                f"Requested {rejected_sample_size} rejected images, but only "
                f"{len(candidates)} are available."
            )
        sampled = random.Random(seed).sample(candidates, rejected_sample_size)
        selected = []
        for candidate in sampled:
            relative = Path(candidate["image"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError(
                    f"Candidate image must be relative to {root}: {candidate!r}"
                )
            key = relative.as_posix()
            if key not in by_image:
                raise ValueError(
                    f"Rejected image is not in Cityscapes {split}: {key}"
                )
            selected.append(by_image[key])
            scores[key] = float(candidate["score"])
        selected.sort(key=lambda pair: pair[0])
    return selected, scores


def main(argv=None):
    args = parse_args(argv)
    ignore_root = args.ignore_root.resolve()
    try:
        ignore_root.relative_to(CITYSCAPES_ROOT.resolve())
    except ValueError as error:
        raise ValueError(
            f"Ignore-mask output must be inside {CITYSCAPES_ROOT}: {ignore_root}"
        ) from error

    image_paths, mask_paths = collect_cityscapes_pairs(args.split)
    pairs = list(zip(image_paths, mask_paths))
    scores = {}
    review_callback = None
    if args.candidates_json is not None:
        if args.review_rejected_sample < 0:
            raise ValueError("--review-rejected-sample cannot be negative.")
        state_path, reviews = load_review_state(
            args.candidates_json, args.split, CITYSCAPES_ROOT
        )
        pairs, scores = load_candidate_pairs(
            args.candidates_json,
            args.split,
            CITYSCAPES_ROOT,
            pairs,
            reviewed=reviews,
            rejected_sample_size=args.review_rejected_sample,
            seed=args.seed,
        )
        def review_callback(image, status):
            reviews[Path(image).relative_to(CITYSCAPES_ROOT).as_posix()] = {
                "status": status
            }
            save_review_state(
                args.candidates_json,
                args.split,
                CITYSCAPES_ROOT,
                state_path,
                reviews,
            )
        if not pairs:
            print("No unreviewed images remain in this review queue.")
            return
    elif args.review_rejected_sample:
        raise ValueError("--review-rejected-sample requires --candidates-json.")
    if args.start_index is None:
        start_index = next(
            (
                index for index, (image_path, _) in enumerate(pairs)
                if not ignore_mask_path(image_path, ignore_root).is_file()
            ),
            0,
        )
    else:
        start_index = args.start_index
    if not 0 <= start_index < len(pairs):
        raise ValueError(
            f"--start-index must be between 0 and {len(pairs) - 1}, "
            f"got {start_index}."
        )
    IgnoreMaskAnnotator(
        pairs,
        ignore_root,
        args.split,
        start_index,
        scores=scores,
        review_callback=review_callback,
    ).run()


if __name__ == "__main__":
    main()
