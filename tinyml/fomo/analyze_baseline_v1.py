"""Audit frozen INT8 predictions without training or changing the baseline."""
import argparse
from collections import Counter
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path, PurePosixPath
from urllib.parse import quote

import numpy as np

from config import ROOT, CLASS_NAMES, output_directory
from dataset import centroid_target, resolve_path, validated_records
from evaluate import match_centroids, scores
from inference import decode_grid

ART = ROOT / 'artifacts'
EVALUATION = ART / 'diagnosis-center-int8-evaluation-20261004'
TRAINING = ART / 'diagnosis-center-training-20261004'
EXPORT = ART / 'diagnosis-center-export-20261004'
MANIFEST = ART / 'cityscapes-data-20261004/manifest.json'
TARGETS = ART / 'cityscapes-center-targets-20261004/manifest.json'
FINE_BINS = ('lt2', '2to4', '4to8', '8to16', 'ge16')


def dump(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def size_bin(side):
    return FINE_BINS[next((i for i, edge in enumerate((2, 4, 8, 16)) if side < edge), 4)]


def assignments(record, detections):
    """Return review identities using exactly the production evaluator's ordering."""
    used, labeled = set(), []
    ignored = set(map(tuple, record.get('ignore_cells', [])))
    for detection in sorted(detections, key=lambda d: -d['confidence']):
        candidates = []
        for index, annotation in enumerate(record['annotations']):
            if index in used or annotation['class_id'] != detection['coarse_class_id']:
                continue
            x, y, w, h = annotation['bbox']
            distance = math.hypot((detection['centroid_x']-x-w/2)*160/record['width'],
                                  (detection['centroid_y']-y-h/2)*160/record['height'])
            if distance <= 8:
                candidates.append((distance, index))
        if candidates:
            _, index = min(candidates)
            used.add(index)
            labeled.append(dict(kind='tp', annotation_index=index, detection=detection))
        else:
            cell = (min(19, int(detection['centroid_y']/record['height']*20)),
                    min(19, int(detection['centroid_x']/record['width']*20)))
            labeled.append(dict(kind='ignored' if cell in ignored else 'fp', detection=detection))
    # Guard the attribution layer against accidental changes in official matching.
    expected, _ = match_centroids(detections, record['annotations'], record['width'], record['height'],
                                  ignore_cells=record.get('ignore_cells'))
    for class_id, counts in expected.items():
        for kind in ('tp', 'fp', 'ignored'):
            assert counts.get(kind, 0) == sum(e['kind'] == kind and e['detection']['coarse_class_id'] == class_id for e in labeled)
    return used, labeled


def measure(records, grids, threshold, details=False, decoder='cell_nms'):
    totals = {name: Counter(tp=0, fp=0, fn=0, ignored=0) for name in CLASS_NAMES}
    bins = {name: {c: Counter(total=0, matched=0) for c in CLASS_NAMES} for name in FINE_BINS}
    events = []
    for ri, (r, grid) in enumerate(zip(records, grids)):
        detections = decode_grid(grid, r['width'], r['height'], threshold, 0, decoder)
        used, labeled = assignments(r, detections)
        for event in labeled:
            totals[event['detection']['class_name']][event['kind']] += 1
            if details:
                events.append(dict(record_index=ri, image_id=r['id'], **event))
        centers = np.array([[(a['bbox'][0]+a['bbox'][2]/2)*160/r['width'],
                             (a['bbox'][1]+a['bbox'][3]/2)*160/r['height']] for a in r['annotations']])
        for ai, a in enumerate(r['annotations']):
            x, y, w, h = a['bbox']
            side = min(w*160/r['width'], h*160/r['height'])
            name = CLASS_NAMES[a['class_id']]
            bins[size_bin(side)][name].update(total=1, matched=int(ai in used))
            if ai not in used:
                totals[name]['fn'] += 1
                if details:
                    center = centers[ai]
                    row, col = min(19, int(center[1]/8)), min(19, int(center[0]/8))
                    neighbors = int(np.sum(np.linalg.norm(centers-center, axis=1) <= 8)-1)
                    edge = min(x*160/r['width'], y*160/r['height'],
                               (r['width']-x-w)*160/r['width'], (r['height']-y-h)*160/r['height']) <= 4
                    winning = int(grid[row, col].argmax())
                    events.append(dict(kind='fn', record_index=ri, image_id=r['id'], annotation_index=ai,
                                       class_name=name, minimum_dimension=side, size_bin=size_bin(side),
                                       edge_within_4px=bool(edge), nearby_centers_within_8px=neighbors,
                                       center_cell_winner=('background', *CLASS_NAMES)[winning],
                                       center_class_probability=float(grid[row, col, a['class_id']+1]),
                                       occlusion='unavailable in retained box annotations'))
    overall = Counter(tp=0, fp=0, fn=0, ignored=0)
    for counts in totals.values():
        overall.update(counts)
    size_results = {}
    for name, classes in bins.items():
        total = sum(c['total'] for c in classes.values())
        matched = sum(c['matched'] for c in classes.values())
        size_results[name] = dict(total=total, matched=matched, recall=matched/total if total else None,
                                  per_class={c: dict(v, recall=v['matched']/v['total'] if v['total'] else None) for c, v in classes.items()})
    return dict(threshold=threshold, micro=scores(overall), per_class={c: scores(v) for c, v in totals.items()}, size_recall=size_results), events


def coarse_sizes(result):
    groups = {'lt2': ('lt2',), '2to8': ('2to4', '4to8'), 'ge8': ('8to16', 'ge16')}
    return {name: dict(total=sum(result['size_recall'][k]['total'] for k in keys),
                       matched=sum(result['size_recall'][k]['matched'] for k in keys),
                       recall=sum(result['size_recall'][k]['matched'] for k in keys)/sum(result['size_recall'][k]['total'] for k in keys)) for name, keys in groups.items()}


def balance(records):
    raw, effective = Counter(), Counter()
    for r in records:
        target = centroid_target(r['annotations'], r['width'], r['height'], r.get('target_ignore_cells', r.get('ignore_cells')))
        weight = np.where(target < 0, 0., np.where(target > 0, 20., 1.))
        weight /= max(weight.mean(), 1e-8)
        for channel, name in ((-1, 'ignored'), (0, 'background'), (1, 'person'), (2, 'vehicle')):
            mask = target == channel
            raw[name] += int(mask.sum())
            effective[name] += float(weight[mask].sum())
    positive = raw['person']+raw['vehicle']
    return dict(images=len(records), cells=dict(raw), normalized_weight_mass=dict(effective),
                background_per_positive=raw['background']/positive,
                nominal_weighted_object_to_background=20*positive/raw['background'],
                normalized_object_to_background=(effective['person']+effective['vehicle'])/effective['background'])


def review_page(cards):
    """Render URLs relative to review/index.html, including legacy Windows records."""
    page='<html><meta charset="utf-8"><title>Baseline v1 error review</title><style>body{background:#171b24;color:white;font-family:Arial;max-width:1100px;margin:auto}img{max-width:100%}article{margin:24px 0}</style><h1>Baseline v1 review at threshold 0.60</h1>'
    page+=f'<p>{len(cards)} stratified examples. These illustrate patterns; they are not a random prevalence sample. FP means unmatched under the centroid protocol, not proof that nothing exists there. Occlusion is not automatically labeled.</p>'
    for i, card in enumerate(cards):
        relative=PurePosixPath(card['file'].replace('\\', '/')).relative_to('review')
        url=quote(relative.as_posix(), safe='/')
        label=html.escape(f"{i}: {card['kind'].upper()}")
        page+=f'<article><h2>{label}</h2><img loading="lazy" alt="{label}" src="{url}"></article>'
    return page+'</html>'


def review(directory, records, grids, events):
    from PIL import Image, ImageDraw, ImageFont
    font_path = Path('C:/Windows/Fonts/arial.ttf')
    font = ImageFont.truetype(str(font_path), 17) if font_path.exists() else ImageFont.load_default()
    chosen = []
    # At most one example per image within each class/bin; deterministic and stratified, not a prevalence sample.
    for kind in ('tp', 'fp', 'fn'):
        for name in CLASS_NAMES:
            for bin_name in (FINE_BINS if kind == 'fn' else ('all',)):
                pool = [e for e in events if e['kind'] == kind and
                        (e.get('class_name') or e['detection']['class_name']) == name and
                        (bin_name == 'all' or e['size_bin'] == bin_name)]
                image_ids = set()
                target_count = 3 if kind == 'fn' else 5
                for event in pool:
                    if event['image_id'] in image_ids:
                        continue
                    chosen.append(event)
                    image_ids.add(event['image_id'])
                    if len(image_ids) >= target_count:
                        break
    output = directory/'review'
    output.mkdir()
    cards = []
    for index, e in enumerate(chosen):
        r = records[e['record_index']]
        with Image.open(r['path']) as im:
            original = im.convert('RGB')
        full = original.resize((960, 480), Image.Resampling.BILINEAR)
        draw = ImageDraw.Draw(full)
        for a in r['annotations']:
            x,y,w,h = a['bbox']
            draw.rectangle((x/r['width']*960,y/r['height']*480,(x+w)/r['width']*960,(y+h)/r['height']*480), outline='#777777')
        for d in decode_grid(grids[e['record_index']], r['width'], r['height'], .6):
            cx,cy=d['centroid_x']/r['width']*960,d['centroid_y']/r['height']*480
            draw.ellipse((cx-3,cy-3,cx+3,cy+3), fill='cyan')
        color = {'fn':'#ff5050','fp':'#ffad32','tp':'#50ef70'}[e['kind']]
        if 'annotation_index' in e:
            a=r['annotations'][e['annotation_index']]
            x,y,w,h=a['bbox']; cx,cy=x+w/2,y+h/2
            draw.rectangle((x/r['width']*960,y/r['height']*480,(x+w)/r['width']*960,(y+h)/r['height']*480), outline=color, width=3)
            side=min(w*160/r['width'], h*160/r['height'])
        else:
            cx,cy=e['detection']['centroid_x'],e['detection']['centroid_y']; w,h=96,96; side=None
            draw.ellipse((cx/r['width']*960-10,cy/r['height']*480-10,cx/r['width']*960+10,cy/r['height']*480+10), outline=color, width=3)
        half=max(w,h,128)*.8
        box=(max(0,int(cx-half)),max(0,int(cy-half)),min(r['width'],int(cx+half)),min(r['height'],int(cy+half)))
        crop=original.crop(box); crop.thumbnail((380,380))
        # Reuse the exact cached bilinear input, then nearest-neighbor enlarge for visibility.
        cache=np.load(resolve_path(r['cache_file'],r['manifest_dir']),mmap_mode='r')[r['cache_index']]
        model_view=Image.fromarray(cache).resize((380,380),Image.Resampling.NEAREST)
        dd=ImageDraw.Draw(model_view)
        dd.ellipse((cx/r['width']*380-6,cy/r['height']*380-6,cx/r['width']*380+6,cy/r['height']*380+6),outline=color,width=2)
        canvas=Image.new('RGB',(960,970),'#171b24'); canvas.paste(full,(0,90))
        canvas.paste(crop,(35,600)); canvas.paste(model_view,(545,585))
        dd=ImageDraw.Draw(canvas)
        name=e.get('class_name') or e['detection']['class_name']
        dd.text((15,12),f"{index:02d} {e['kind'].upper()} / {name} / {r['id']}",font=font,fill=color)
        label=f"Minimum dimension at model input: {side:.2f} px" if side is not None else f"Unmatched prediction confidence: {e['detection']['confidence']:.3f}"
        dd.text((15,39),label+' | threshold 0.60',font=font,fill='white')
        dd.text((15,64),'Selected target highlighted; gray = labels; cyan = predicted centers',font=font,fill='white')
        dd.text((35,574),'Source detail (enlarged)',font=font,fill='white')
        dd.text((545,564),'Actual 160 x 160 input (enlarged)',font=font,fill='white')
        folder=output/e['kind']; folder.mkdir(exist_ok=True)
        path=folder/f'{index:02d}-{name}-{e.get("size_bin","example")}.jpg'
        canvas.save(path,quality=90)
        cards.append(dict(**e, file=path.relative_to(directory).as_posix()))
    dump(output/'index.json',cards)
    (output/'index.html').write_text(review_page(cards),encoding='utf-8')
    return cards


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=output_directory(args.output,'baseline-v1-analysis')
    records,_=validated_records(MANIFEST)
    records=[r for r in records if r['split']=='validation']
    saved=json.loads((EVALUATION/'detections.json').read_text())
    assert [r['id'] for r in records]==[r['id'] for r in saved]
    grids=np.load(EVALUATION/'probabilities.npy',allow_pickle=False)
    assert grids.shape==(500,20,20,3)
    old=json.loads((EVALUATION/'report.json').read_text())
    reference,events=measure(records,grids,.6,True)
    assert reference['micro']==old['uncapped_micro'] and reference['per_class']==old['per_class']
    print('Exact baseline metrics reproduced.',flush=True)
    summary=dict(name='baseline_v1',model='local FOMO-inspired MobileNetV2 alpha 0.35',dataset='Cityscapes official train/validation',
                 classes=list(CLASS_NAMES),input_size=[160,160,3],grid_size=[20,20],evaluation=reference,
                 size_stratified_recall=coarse_sizes(reference),model_size=(EXPORT/'model.int8.tflite').stat().st_size,
                 checkpoint_path=str(TRAINING/'best.keras'),int8_path=str(EXPORT/'model.int8.tflite'),
                 training_configuration=json.loads((TRAINING/'best.json').read_text()),
                 protected_files={str(p.relative_to(ROOT)):sha(p) for folder in (TRAINING,EXPORT,EVALUATION,ART/'diagnosis-center-evaluation-20261004') for p in folder.rglob('*') if p.is_file()},
                 training_manifest_sha256=sha(TARGETS),evaluation_manifest_sha256=sha(MANIFEST),
                 protocol='Same-class confidence-greedy one-to-one, Euclidean match <=8 input pixels; same-class Chebyshev NMS <=8; all objects; fixed original ignore cells; no output cap',
                 scope='Development validation already used for model selection, not independent test accuracy')
    dump(out/'baseline_v1.json',summary)
    (out/'BASELINE_V1.md').write_text('# My frozen baseline_v1\n\nI identify the existing saved 20 x 20 model as `baseline_v1`.\nI preserve its original files in place and record their SHA-256 fingerprints in `baseline_v1.json`.\nI do not retrain or overwrite it.\n\n'+f"- Checkpoint: `{summary['checkpoint_path']}`.\n- INT8: `{summary['int8_path']}`.\n- Dataset: Cityscapes, 2,975 train and 500 validation images.\n- Classes: Person and Vehicle; 160 x 160 RGB input, 20 x 20 grid.\n- Precision / recall / F1: 70.65% / 25.43% / 37.40% at confidence 0.60.\n- Person: 63.56% / 22.85% / 33.62%.\n- Vehicle: 75.16% / 27.07% / 39.80%.\n- Recall below 2 px / 2 to below 8 px / at least 8 px: 12.37% / 21.53% / 50.64%.\n- INT8 size: 70,736 bytes.\n- Training: ImageNet initialization, seed 42, batch 32, object weight 20, horizontal flips, initial learning rate 0.001, maximum 20 epochs, best epoch 19.\n- Full training configuration and original history: `baseline_v1.json`.\n",encoding='utf-8')
    sweep=[]
    for t in range(20,81,5):
        result,_=measure(records,grids,t/100)
        result['coarse_size_recall']=coarse_sizes(result)
        sweep.append(result)
        print(json.dumps(dict(threshold=t/100,**result['micro'])),flush=True)
    dump(out/'threshold_sweep.json',sweep)
    rows=[]
    for r in sweep:
        row={'threshold':r['threshold'],**r['micro']}
        for c in CLASS_NAMES:
            row.update({f'{c}_{k}':v for k,v in r['per_class'][c].items()})
        row.update({f'{k}_recall':v['recall'] for k,v in r['coarse_size_recall'].items()})
        row.update({f'{k}_count':v['total'] for k,v in r['coarse_size_recall'].items()})
        rows.append(row)
    with (out/'threshold_sweep.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    dump(out/'size_analysis.json',reference['size_recall'])
    fn=[e for e in events if e['kind']=='fn']
    counts=dict(false_negatives=len(fn),class_counts=dict(Counter(e['class_name'] for e in fn)),
                size_counts=dict(Counter(e['size_bin'] for e in fn)),edge_within_4px=sum(e['edge_within_4px'] for e in fn),
                crowded_nearby_center_within_8px=sum(e['nearby_centers_within_8px']>0 for e in fn),
                center_cell_winner=dict(Counter(e['center_cell_winner'] for e in fn)),
                occlusion='not measured: retained annotations do not supply visibility/occlusion; inspect examples manually',
                caveat='Size and class counts partition misses; edge and crowd tags overlap and are geometric proxies, not causal diagnoses.')
    dump(out/'error_counts.json',counts); dump(out/'error_events.json',events)
    cards=review(out,records,grids,events)
    targets,_=validated_records(TARGETS)
    balancing={split:balance([r for r in targets if r['split']==split]) for split in ('train','validation')}
    dump(out/'loss_balance.json',balancing)
    history=json.loads((TRAINING/'history.json').read_text())
    dump(out/'training_history_analysis.json',dict(history=history,training_f1=None,
         training_f1_note='Not recorded per epoch; cannot reconstruct a training F1 curve from a single best checkpoint.',
         best_epoch=summary['training_configuration']['best_epoch'],stopping='20-epoch budget reached, not early stopping',
         minimum_validation_loss_epoch=min(history,key=lambda r:r['validation_loss'])['epoch']))
    os.environ['MPLCONFIGDIR']=str(out/'matplotlib-cache')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for filename,x,y,xlabel,ylabel in (
        ('precision_recall.png',[r['micro']['recall']*100 for r in sweep],[r['micro']['precision']*100 for r in sweep],'Recall (%)','Precision (%)'),
        ('f1_threshold.png',[r['threshold'] for r in sweep],[r['micro']['f1']*100 for r in sweep],'Confidence threshold','F1 (%)')):
        fig,ax=plt.subplots(figsize=(8,5),layout='constrained'); ax.plot(x,y,'o-')
        for r,xx,yy in zip(sweep,x,y):
            if filename=='precision_recall.png' and r['threshold'] not in (.2,.45,.55,.6,.7,.8):
                continue
            label='0.20-0.35' if filename=='precision_recall.png' and r['threshold']==.2 else f"{r['threshold']:.2f}"
            ax.annotate(label,(xx,yy),xytext=(-50,8) if r['threshold']==.2 else (5,4),textcoords='offset points',fontsize=8)
        ax.set(xlabel=xlabel,ylabel=ylabel,title='Frozen baseline_v1 INT8 - all 500 validation images'); ax.grid(alpha=.25)
        fig.savefig(out/filename,dpi=160); plt.close(fig)
    fig,axes=plt.subplots(1,3,figsize=(13,4),layout='constrained')
    epochs=[r['epoch'] for r in history]
    for key,label in (('train_loss','Training'),('validation_loss','Validation')):
        axes[0].plot(epochs,[r[key] for r in history],label=label)
    axes[0].set(title='Weighted loss',xlabel='Epoch'); axes[0].legend()
    axes[1].plot(epochs,[r['validation_f1']*100 for r in history]); axes[1].set(title='Validation centroid F1 (%)',xlabel='Epoch')
    axes[2].step(epochs,[r['learning_rate'] for r in history]); axes[2].set(title='Learning rate',xlabel='Epoch')
    for ax in axes: ax.grid(alpha=.2); ax.axvline(19,color='gray',linestyle=':')
    fig.savefig(out/'training_curves.png',dpi=160); plt.close(fig)
    print(json.dumps(dict(output=str(out),errors=counts,balance=balancing,review_examples=len(cards)),indent=2),flush=True)


if __name__=='__main__':
    main()
