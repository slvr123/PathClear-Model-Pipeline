"""Object-center precision/recall, not bounding-box IoU or background accuracy."""
import argparse
from collections import Counter
import json
import math

import numpy as np

from config import CLASS_NAMES, SETTINGS, output_directory
from dataset import load_rgb, load_arrays, validated_records
from inference import DECODERS, decoding_policy, decode_grid


def match_centroids(detections, annotations, width, height, radius=SETTINGS.match_radius, ignore_cells=None):
    """Confidence-ordered one-to-one matching, same class, distance in resized-input pixels."""
    if not math.isfinite(radius) or radius <= 0:
        raise ValueError("Matching radius must be finite and positive")
    counts = {i: Counter(tp=0, fp=0, fn=0) for i in range(len(CLASS_NAMES))}
    used, distances = set(), []
    for detection in sorted(detections, key=lambda d: -d["confidence"]):
        class_id = detection["coarse_class_id"]
        candidates = []
        for index, annotation in enumerate(annotations):
            if index in used or annotation["class_id"] != class_id:
                continue
            x, y, w, h = annotation["bbox"]
            dx = (detection["centroid_x"] - (x + w / 2)) * SETTINGS.image_size / width
            dy = (detection["centroid_y"] - (y + h / 2)) * SETTINGS.image_size / height
            distance = math.hypot(dx, dy)
            if distance <= radius:
                candidates.append((distance, index))
        if candidates:
            distance, index = min(candidates)
            used.add(index)
            distances.append(distance)
            counts[class_id]["tp"] += 1
        elif (min(SETTINGS.grid_size-1, int(detection["centroid_y"]/height*SETTINGS.grid_size)),
              min(SETTINGS.grid_size-1, int(detection["centroid_x"]/width*SETTINGS.grid_size))) in set(map(tuple, ignore_cells or [])):
            counts[class_id]["ignored"] += 1
        else:
            counts[class_id]["fp"] += 1
    for index, annotation in enumerate(annotations):
        if index not in used:
            counts[annotation["class_id"]]["fn"] += 1
    return counts, distances


def scores(counts):
    tp, fp, fn = counts["tp"], counts["fp"], counts["fn"]
    precision = tp / (tp + fp) if tp + fp else None
    recall = tp / (tp + fn) if tp + fn else None
    f1 = 2 * tp / (2 * tp + fp + fn) if 2 * tp + fp + fn else None
    return {"tp": tp, "fp": fp, "fn": fn, "ignored": counts.get("ignored", 0),
            "precision": precision, "recall": recall, "f1": f1}


def score_grids(records, grids, threshold=SETTINGS.confidence_threshold, radius=SETTINGS.match_radius,
                decoder=None):
    if len(records) != len(grids):
        raise ValueError("Record/prediction counts must match")
    totals = {i: Counter(tp=0, fp=0, fn=0) for i in range(len(CLASS_NAMES))}
    errors, confidences, matched_confidences, capped_totals = [], [], [], Counter(tp=0, fp=0, fn=0)
    all_detections = []
    for record, grid in zip(records, grids):
        detections = decode_grid(grid, record["width"], record["height"], threshold, max_detections=0, decoder=decoder)
        all_detections.append(detections)
        matched, distances = match_centroids(detections, record["annotations"], record["width"], record["height"], radius, record.get("ignore_cells"))
        for class_id, count in matched.items():
            totals[class_id].update(count)
        capped, _ = match_centroids(detections[:3], record["annotations"],
                                    record["width"], record["height"], radius, record.get("ignore_cells"))
        for count in capped.values():
            capped_totals.update(count)
        errors.extend(distances)
        confidences.extend(d["confidence"] for d in detections)
    micro = Counter(tp=0, fp=0, fn=0)
    for count in totals.values():
        micro.update(count)
    report = {"images": len(records), "confidence_threshold": threshold, "radius_input_pixels": radius,
              "uncapped_micro": scores(micro), "legacy_top3_micro": scores(capped_totals),
              "decoding": dict(decoding_policy(decoder), threshold=threshold),
              "per_class": {CLASS_NAMES[i]: scores(v) for i, v in totals.items()},
              "matched_center_error_input_pixels_mean": float(np.mean(errors)) if errors else None,
              "matched_center_error_input_pixels_median": float(np.median(errors)) if errors else None,
              "matched_center_error_input_pixels_p95": float(np.percentile(errors,95)) if errors else None,
              "detection_confidence_mean": float(np.mean(confidences)) if confidences else None,
              "detection_confidence_quantiles": np.quantile(confidences,[0,.25,.5,.75,1]).tolist() if confidences else None,
              "detection_count": len(confidences)}
    return report, all_detections


def save_previews(directory, records, detections, count=12):
    from PIL import Image, ImageDraw
    indices = np.linspace(0, len(records)-1, min(count, len(records)), dtype=int)
    for index in indices:
        record = records[index]
        frame = Image.fromarray(load_rgb(record["path"]))
        scale = min(1.0, 1024/frame.width)
        frame = frame.resize((round(frame.width*scale), round(frame.height*scale)))
        draw = ImageDraw.Draw(frame)
        for annotation in record["annotations"]:
            x,y,w,h = [v*scale for v in annotation["bbox"]]
            draw.rectangle((x,y,x+w,y+h),outline="gray",width=1)
        for detection in detections[index]:
            x,y = detection["centroid_x"]*scale, detection["centroid_y"]*scale
            color = "lime" if detection["coarse_class_id"] == 0 else "cyan"
            draw.ellipse((x-4,y-4,x+4,y+4),fill=color)
            draw.text((x+5,y-12),f"{detection['class_name'].upper()} {detection['confidence']:.2f}",fill=color)
        frame.save(directory / f"prediction-{index:04d}.png")


def main():
    import time
    from pathlib import Path
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--split", choices=("validation", "test"), default="validation")
    parser.add_argument("--allow-synthetic", action="store_true")
    parser.add_argument("--threshold", type=float, default=SETTINGS.confidence_threshold)
    parser.add_argument("--decoder", choices=DECODERS, default=SETTINGS.decoder)
    parser.add_argument("--radius", type=float, default=SETTINGS.match_radius)
    parser.add_argument("--samples", type=int, default=12)
    parser.add_argument("--output")
    args=parser.parse_args()
    if not 0<=args.threshold<=1 or not math.isfinite(args.radius) or args.radius<=0:
        parser.error("threshold must be in [0,1], radius finite and positive")
    records,dataset_report=validated_records(args.manifest,args.allow_synthetic)
    records=[r for r in records if r['split']==args.split]
    if not records:
        parser.error(f"No {args.split} records; Cityscapes test labels are withheld")
    from model import load_model
    model,metadata=load_model(args.model)
    if metadata['synthetic'] and not args.allow_synthetic:
        parser.error("Synthetic model requires --allow-synthetic")
    images,_=load_arrays(records)
    import tensorflow as tf
    is_int8=Path(args.model).suffix=='.tflite'
    if is_int8:
        invoke=lambda x: np.asarray(model(x,training=False))
        batch_size=1
    else:
        @tf.function(input_signature=[tf.TensorSpec((None,160,160,3),tf.float32)])
        def forward(x):
            return model(x,training=False)
        invoke=lambda x: forward(x).numpy()
        batch_size=SETTINGS.batch_size
    invoke(images[:1].astype(np.float32)/255)
    grids=[]
    started=time.perf_counter()
    for i in range(0,len(images),batch_size):
        grids.append(invoke(images[i:i+batch_size].astype(np.float32)/255))
    grids=np.concatenate(grids)
    total_seconds=time.perf_counter()-started
    latencies=[]
    for i in range(min(30,len(images))):
        sample=images[i:i+1].astype(np.float32)/255
        start=time.perf_counter()
        invoke(sample)
        latencies.append((time.perf_counter()-start)*1000)
    report,detections=score_grids(records,grids,args.threshold,args.radius,args.decoder)
    report.update({'split':args.split,'synthetic':dataset_report['synthetic'] or metadata['synthetic'],
                   'sanity_model':metadata['sanity'],'model_path':str(Path(args.model).resolve()),
                   'model_bytes':Path(args.model).stat().st_size,
                   'matching':'same-class confidence-greedy one-to-one, input-pixel distance; unmatched predictions in source void/group cells ignored',
                   'pc_single_frame_model_call_ms_median':float(np.median(latencies)),
                   'pc_single_frame_model_call_ms_p95':float(np.percentile(latencies,95)),
                   'pc_batch_inference_seconds':total_seconds,'timing_samples':len(latencies),
                   'runtime':'TFLite CPU, one thread' if is_int8 else 'TensorFlow compiled inference',
                   'devices':[str(d) for d in tf.config.list_physical_devices()],
                   'timing_scope':'normalized input already prepared; includes model call and output copy; excludes camera, decode, MicroSeg, and decision logic',
                   'threshold_diagnostics':{str(t):score_grids(records,grids,t,args.radius,args.decoder)[0]['uncapped_micro'] for t in (.3,.6,.8)}})
    directory=output_directory(args.output,'evaluation')
    (directory/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    (directory/'detections.json').write_text(json.dumps([{'id':r['id'],'detections':d} for r,d in zip(records,detections)],indent=2),encoding='utf-8')
    np.save(directory/'probabilities.npy',grids)
    save_previews(directory,records,detections,args.samples)
    print(json.dumps(report,indent=2))
    print(f'Report: {directory}')


if __name__ == '__main__':
    main()
