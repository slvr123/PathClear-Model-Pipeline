"""Create a new center-mask training variant while preserving all evaluation labels."""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from functools import partial
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import output_directory
from dataset import resolve_path, validated_records
from prepare_cityscapes import center_ignore_cells


def refine_record(record, grid_size=20):
    with Image.open(resolve_path(record['source_mask'])) as image:
        mask = np.asarray(image)
    result = {k: v for k, v in record.items() if k not in ('path', 'manifest_dir')}
    result['target_ignore_cells'] = center_ignore_cells(mask, grid_size)
    result['target_grid_size'] = grid_size
    result['cache_file'] = resolve_path(record['cache_file'], record['manifest_dir']).as_posix()
    old = set(map(tuple, record['ignore_cells']))
    new = set(map(tuple, result['target_ignore_cells']))
    if grid_size == 20 and not new <= old:
        raise ValueError('Center ignore mask must be a subset of conservative any-pixel mask')
    counts = Counter(original_20_grid_ignored_cells=len(old), target_ignored_cells=len(new))
    if grid_size == 20:
        counts['newly_supervised_cells'] = len(old-new)
    return result, counts


def save_target_preview(record, destination):
    cache = np.load(resolve_path(record['cache_file']), mmap_mode='r')
    rgb = cache[record['cache_index']]
    canvas = Image.new('RGB', (960, 520), 'white')
    for index, key in enumerate(('ignore_cells', 'target_ignore_cells')):
        image = Image.fromarray(rgb).resize((480, 480))
        overlay = Image.new('RGBA', image.size, (0, 0, 0, 0))
        draw = ImageDraw.Draw(overlay)
        cell_size = 480 / (20 if key == 'ignore_cells' else record.get('target_grid_size',20))
        for row, col in record[key]:
            draw.rectangle((col*cell_size, row*cell_size, (col+1)*cell_size-1, (row+1)*cell_size-1), fill=(255, 0, 255, 95))
        image = Image.alpha_composite(image.convert('RGBA'), overlay).convert('RGB')
        draw = ImageDraw.Draw(image)
        for a in record['annotations']:
            x, y, w, h = a['bbox']
            cx, cy = (x+w/2)/record['width']*480, (y+h/2)/record['height']*480
            draw.ellipse((cx-2, cy-2, cx+2, cy+2), fill='lime' if a['class_id']==0 else 'cyan')
        canvas.paste(image, (index*480, 40))
    ImageDraw.Draw(canvas).text((8, 8), 'Original any-pixel mask (left) | Center mask (right). Magenta = unknown; dots = labeled centers.', fill='black')
    canvas.save(destination)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True, type=Path)
    parser.add_argument('--output')
    parser.add_argument('--grid-size', type=int, choices=(20,40), default=20)
    args = parser.parse_args()
    records, _ = validated_records(args.manifest)
    manifest = json.loads(args.manifest.read_text())
    if any('target_ignore_cells' in r for r in records):
        raise ValueError('Expected original any-pixel manifest, not an already refined variant')
    directory = output_directory(args.output, 'center-targets')
    totals, revised = Counter(), []
    with ThreadPoolExecutor(4) as pool:
        for record, counts in pool.map(partial(refine_record, grid_size=args.grid_size), records):
            revised.append(record)
            totals.update(counts)
    manifest.update(reviewed=False, images=revised, training_ignore_policy='semantic label at grid cell center',
                    target_grid_size=args.grid_size,
                    evaluation_ignore_policy='original any-pixel void/group cells, unchanged',
                    parent_manifest=str(args.manifest.resolve()),
                    parent_manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest())
    (directory/'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    (directory/'changes.json').write_text(json.dumps(dict(totals), indent=2), encoding='utf-8')
    for split in ('train', 'validation'):
        subset = [r for r in revised if r['split'] == split]
        for i in np.linspace(0, len(subset)-1, 3, dtype=int):
            save_target_preview(subset[i], directory/f'target-{split}-{i}.png')
    print(json.dumps(dict(output=str(directory), **totals), indent=2))


if __name__ == '__main__':
    main()
