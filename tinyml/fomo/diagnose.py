"""Reproduce decoding losses on saved predictions without changing model weights."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

import numpy as np

from config import CLASS_NAMES, SETTINGS, output_directory
from dataset import centroid_target, validated_records
from evaluate import match_centroids, scores, save_previews
from inference import decode_grid


def cell_detections(grid, width, height, threshold=.60):
    """Diagnostic alternative: each winning foreground cell is one center."""
    g = grid.shape[0]
    winners = grid.argmax(axis=-1)
    detections = []
    for row, col in zip(*np.where((winners > 0) & (grid.max(axis=-1) >= threshold))):
        channel = int(winners[row, col])
        detections.append(dict(centroid_x=float((col+.5)/g*width),
                               centroid_y=float((row+.5)/g*height),
                               coarse_class_id=channel-1, class_name=CLASS_NAMES[channel-1],
                               confidence=float(grid[row, col, channel])))
    return sorted(detections, key=lambda d: (-d['confidence'], d['coarse_class_id'],
                                            d['centroid_y'], d['centroid_x']))


def summarize(records, predictions):
    totals = {i: Counter(tp=0, fp=0, fn=0) for i in range(len(CLASS_NAMES))}
    sizes = {name: Counter(matched=0, total=0) for name in ('tiny_lt2', 'small_2_to8', 'larger_ge8')}
    top3 = Counter(tp=0, fp=0, fn=0)
    ignored_top3 = raw = 0
    for record, detections in zip(records, predictions):
        annotations, width, height = record['annotations'], record['width'], record['height']
        counts, _ = match_centroids(detections, annotations, width, height,
                                    ignore_cells=record.get('ignore_cells'))
        capped, _ = match_centroids(detections[:3], annotations, width, height,
                                    ignore_cells=record.get('ignore_cells'))
        for class_id, count in counts.items():
            totals[class_id].update(count)
        for count in capped.values():
            top3.update(count)
            ignored_top3 += count.get('ignored', 0)
        raw += len(detections)
        # Match ALL objects together before attributing recall to size bins.
        # Filtering GT first would incorrectly let a detection match multiple size bins.
        used = set()
        for detection in sorted(detections, key=lambda d: -d['confidence']):
            candidates = []
            for i, a in enumerate(annotations):
                if i in used or a['class_id'] != detection['coarse_class_id']:
                    continue
                x, y, w, h = a['bbox']
                distance = np.hypot((detection['centroid_x']-x-w/2)*160/width,
                                    (detection['centroid_y']-y-h/2)*160/height)
                if distance <= SETTINGS.match_radius:
                    candidates.append((distance, i))
            if candidates:
                used.add(min(candidates)[1])
        assert len(used) == sum(c['tp'] for c in counts.values())
        for i, a in enumerate(annotations):
            side = min(a['bbox'][2]*160/width, a['bbox'][3]*160/height)
            group = 'tiny_lt2' if side < 2 else 'small_2_to8' if side < 8 else 'larger_ge8'
            sizes[group]['total'] += 1
            sizes[group]['matched'] += i in used
    total = Counter(tp=0, fp=0, fn=0)
    for count in totals.values():
        total.update(count)
    return dict(micro=scores(total), per_class={CLASS_NAMES[i]: scores(c) for i, c in totals.items()},
                size_recall={k: dict(v, recall=v['matched']/v['total'] if v['total'] else None)
                             for k, v in sizes.items()},
                raw_detections=raw, legacy_top3=scores(top3), ignored_in_top3=ignored_top3)


def oracle_grid(record, grid_size):
    """Positive target geometry only; unknown and conflict cells emit background."""
    target = np.zeros((grid_size, grid_size), np.int32)
    occupied = {}
    counts = Counter()
    for annotation in record['annotations']:
        x, y, w, h = annotation['bbox']
        cell = (int((y+h/2)/record['height']*grid_size), int((x+w/2)/record['width']*grid_size))
        channel = annotation['class_id']+1
        counts[cell] += 1
        occupied[cell] = channel if cell not in occupied or occupied[cell] == channel else -1
        target[cell] = occupied[cell]
    grid = np.eye(len(CLASS_NAMES)+1, dtype=np.float32)[np.maximum(target, 0)]
    return grid, dict(extra_centers=sum(n-1 for n in counts.values()),
                      centers_in_conflicting_cells=sum(counts[c] for c, v in occupied.items() if v == -1))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--evaluation', required=True, type=Path, help='Existing evaluation directory')
    parser.add_argument('--output')
    args = parser.parse_args()
    records, _ = validated_records(args.manifest)
    original_report = json.loads((args.evaluation/'report.json').read_text())
    records = [r for r in records if r['split'] == original_report['split']]
    saved = json.loads((args.evaluation/'detections.json').read_text())
    if [r['id'] for r in records] != [r['id'] for r in saved]:
        raise ValueError('Saved predictions do not match manifest record order')
    grids = np.load(args.evaluation/'probabilities.npy', allow_pickle=False)
    if grids.shape != (len(records), 20, 20, 3):
        raise ValueError('Expected saved 20x20 Person/Vehicle grids')
    directory = output_directory(args.output, 'diagnosis')
    connected = [decode_grid(g, r['width'], r['height'], .6, 0, 'connected') for r, g in zip(records, grids)]
    cells = [cell_detections(g, r['width'], r['height']) for r, g in zip(records, grids)]
    cell_nms = [decode_grid(g, r['width'], r['height'], .6, 0, 'cell_nms') for r, g in zip(records, grids)]
    assert all(a == decode_grid(g, r['width'], r['height'], .6, 0, 'cells')
               for a, r, g in zip(cells, records, grids))
    report = dict(manifest_sha256=hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                  probabilities_sha256=hashlib.sha256((args.evaluation/'probabilities.npy').read_bytes()).hexdigest(),
                  evaluation=str(args.evaluation.resolve()), images=len(records),
                  scope='Exploratory development validation; previously used for checkpoint selection.',
                  threshold=.6, radius_input_pixels=8, connected=summarize(records, connected),
                  cells=summarize(records, cells), cell_nms=summarize(records, cell_nms), oracle={})
    # The reproduction must agree with the previously saved end-to-end evaluation.
    if report['connected']['micro'] != original_report['uncapped_micro']:
        raise RuntimeError('Connected decoder no longer reproduces saved baseline metrics')
    for g in (20, 40):
        oracles, counts = [], Counter()
        for record in records:
            grid, count = oracle_grid(record, g)
            counts.update(count)
            oracles.append(grid)
        entry = dict(collisions=dict(counts), cells=summarize(records, [
            cell_detections(grid, r['width'], r['height']) for r, grid in zip(records, oracles)]))
        if g == 20:
            expected = [np.eye(3)[np.maximum(centroid_target(r['annotations'], r['width'], r['height'],
                                                           r.get('ignore_cells')), 0)] for r in records]
            assert all(np.array_equal(a, b) for a, b in zip(expected, oracles))
            entry['connected'] = summarize(records, [decode_grid(grid, r['width'], r['height'], .6, 0, 'connected')
                                                     for r, grid in zip(records, oracles)])
        report['oracle'][str(g)] = entry
    ignored = np.zeros((20, 20), dtype=np.int64)
    false_region = np.zeros((20, 20), dtype=np.int64)
    for record, grid in zip(records, grids):
        for row, col in record.get('ignore_cells', []):
            ignored[row, col] += 1
            false_region[row, col] += int(grid[row, col].argmax() > 0 and grid[row, col].max() >= .6)
    report['ignored_regions'] = dict(ignored_cell_observations=int(ignored.sum()),
                                    foreground_cell_observations=int(false_region.sum()),
                                    caveat='Unknown labels, not proven false positives; never masked at inference.')
    np.savez(directory/'ignore_heatmaps.npz', ignored=ignored, foreground=false_region)
    save_previews(directory, records, cell_nms, count=12)
    (directory/'report.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    print(json.dumps(dict(output=str(directory), connected=report['connected'], cells=report['cells'], cell_nms=report['cell_nms'],
                          oracle=report['oracle'], ignored_regions=report['ignored_regions']), indent=2))


if __name__ == '__main__':
    main()
