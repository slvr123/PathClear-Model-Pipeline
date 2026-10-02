"""Tests for Kaggle FPV Crosswalk image/mask pairing."""
from pathlib import Path

import pytest

import micro_dataset_loader


def _write(path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"test")
    return path


def test_collect_crosswalk_pairs_matches_separate_mirrored_folders(
    tmp_path, monkeypatch
):
    root = tmp_path / "FPVCrosswalk2025"
    image = _write(root / "synthetic" / "rainy" / "images" / "frame_001.jpg")
    mask = _write(root / "synthetic" / "rainy" / "masks" / "frame_001.png")
    monkeypatch.setattr(micro_dataset_loader, "KAGGLE_CROSSWALK_ROOT", root)

    image_paths, mask_paths = micro_dataset_loader._collect_crosswalk_pairs()

    assert image_paths == [str(image)]
    assert mask_paths == [str(mask)]


def test_collect_crosswalk_pairs_matches_sibling_mask_suffix_and_mixed_extensions(
    tmp_path, monkeypatch
):
    root = tmp_path / "FPVCrosswalk2025"
    image = _write(root / "real" / "sunny" / "frame_002.jpeg")
    mask = _write(root / "real" / "sunny" / "frame_002_mask.jpg")
    monkeypatch.setattr(micro_dataset_loader, "KAGGLE_CROSSWALK_ROOT", root)

    image_paths, mask_paths = micro_dataset_loader._collect_crosswalk_pairs()

    assert image_paths == [str(image)]
    assert mask_paths == [str(mask)]


def test_collect_crosswalk_pairs_matches_role_suffixes_in_folder_names(
    tmp_path, monkeypatch
):
    root = tmp_path / "FPVCrosswalk2025"
    image = _write(root / "synthetic_images" / "rainy" / "frame.jpg")
    mask = _write(root / "synthetic_masks" / "rainy" / "frame.png")
    monkeypatch.setattr(micro_dataset_loader, "KAGGLE_CROSSWALK_ROOT", root)

    image_paths, mask_paths = micro_dataset_loader._collect_crosswalk_pairs()

    assert image_paths == [str(image)]
    assert mask_paths == [str(mask)]


def test_collect_crosswalk_pairs_keeps_duplicate_names_in_separate_conditions(
    tmp_path, monkeypatch
):
    root = tmp_path / "FPVCrosswalk2025"
    rainy_image = _write(root / "real" / "rainy" / "images" / "frame.jpg")
    sunny_image = _write(root / "real" / "sunny" / "images" / "frame.jpg")
    rainy_mask = _write(root / "real" / "rainy" / "masks" / "frame.png")
    sunny_mask = _write(root / "real" / "sunny" / "masks" / "frame.png")
    monkeypatch.setattr(micro_dataset_loader, "KAGGLE_CROSSWALK_ROOT", root)

    image_paths, mask_paths = micro_dataset_loader._collect_crosswalk_pairs()

    assert image_paths == [str(rainy_image), str(sunny_image)]
    assert mask_paths == [str(rainy_mask), str(sunny_mask)]


def test_collect_crosswalk_pairs_rejects_ambiguous_masks(tmp_path, monkeypatch):
    root = tmp_path / "FPVCrosswalk2025"
    _write(root / "real" / "images" / "frame.jpg")
    _write(root / "real" / "masks" / "frame.png")
    _write(root / "real" / "annotations" / "frame.png")
    monkeypatch.setattr(micro_dataset_loader, "KAGGLE_CROSSWALK_ROOT", root)

    with pytest.raises(RuntimeError, match="Ambiguous FPV Crosswalk"):
        micro_dataset_loader._collect_crosswalk_pairs()


def test_collect_crosswalk_pairs_rejects_unpaired_images(tmp_path, monkeypatch):
    root = tmp_path / "FPVCrosswalk2025"
    _write(root / "real" / "images" / "frame.jpg")
    monkeypatch.setattr(micro_dataset_loader, "KAGGLE_CROSSWALK_ROOT", root)

    with pytest.raises(RuntimeError, match="images without masks"):
        micro_dataset_loader._collect_crosswalk_pairs()
