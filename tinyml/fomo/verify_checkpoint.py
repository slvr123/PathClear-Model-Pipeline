"""Read-only integrity check for the canonical Cityscapes v2 checkpoint."""
import hashlib
import json
from pathlib import Path


def verify(directory=None):
    root = Path(directory) if directory else Path(__file__).resolve().parent / "checkpoints" / "cityscapes_v2"
    expected = {}
    for line in (root / "SHA256SUMS.txt").read_text(encoding="utf-8").splitlines():
        digest, name = line.split("  ", 1)
        if Path(name).name != name or name in expected:
            raise ValueError(f"Invalid checksum entry: {name}")
        expected[name] = digest
    actual_files = {p.name for p in root.iterdir() if p.is_file() and p.name != "SHA256SUMS.txt"}
    if actual_files != set(expected):
        raise ValueError("Checkpoint inventory differs from SHA256SUMS.txt")
    for name, digest in expected.items():
        with (root / name).open("rb") as stream:
            actual = hashlib.file_digest(stream, "sha256").hexdigest()
        if actual != digest:
            raise ValueError(f"SHA-256 mismatch: {name}")
    config = json.loads((root / "config.json").read_text(encoding="utf-8"))
    for key, value in {"version": "cityscapes_v2", "object_weight": 10, "decoder": "cells",
                       "confidence_threshold": 0.70, "nms_enabled": False,
                       "input_width": 160, "input_height": 160, "channels": 3,
                       "grid_width": 20, "grid_height": 20, "model_size_bytes": 70736}.items():
        if config[key] != value:
            raise ValueError(f"Unexpected frozen setting: {key}")
    if (root / "model.int8.tflite").stat().st_size != config["model_size_bytes"]:
        raise ValueError("INT8 size differs from frozen configuration")
    return {"version": config["version"], "verified_files": len(expected),
            "int8_sha256": expected["model.int8.tflite"], "int8_bytes": config["model_size_bytes"]}


if __name__ == "__main__":
    print(json.dumps(verify(), indent=2))
