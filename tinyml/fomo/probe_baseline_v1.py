"""Supplement frozen prediction analysis with saved-checkpoint inference, never training."""
import argparse
from collections import Counter
import json
import os
import numpy as np

from analyze_baseline_v1 import ART, EVALUATION, EXPORT, MANIFEST, TARGETS, TRAINING, assignments, dump, measure
from config import ROOT, output_directory
from dataset import load_arrays, validated_records
from diagnose import oracle_grid
from evaluate import score_grids
from inference import decode_grid


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True)
    args=parser.parse_args()
    out=output_directory(args.output,'baseline-probe')
    os.environ['KERAS_HOME']=str(ART/'keras-cache')
    records,_=validated_records(MANIFEST)
    val=[r for r in records if r['split']=='validation']
    grids=np.load(EVALUATION/'probabilities.npy')
    reasons=Counter(); own=Counter(); collisions=Counter()
    for r,g in zip(val,grids):
        kept=decode_grid(g,r['width'],r['height'],.6)
        raw=decode_grid(g,r['width'],r['height'],.6,decoder='cells')
        floor=decode_grid(g,r['width'],r['height'],0,decoder='cells')
        used,_=assignments(r,kept)
        def nearby(a, candidates):
            x,y,w,h=a['bbox']
            return any(d['coarse_class_id']==a['class_id'] and np.hypot((d['centroid_x']-x-w/2)*160/r['width'],(d['centroid_y']-y-h/2)*160/r['height'])<=8 for d in candidates)
        for i,a in enumerate(r['annotations']):
            if i in used: continue
            if nearby(a,kept): reason='eligible_prediction_already_matched_to_neighbor'
            elif nearby(a,raw): reason='eligible_raw_cell_removed_by_nms'
            elif nearby(a,floor): reason='same_class_winning_cell_only_below_threshold'
            else: reason='no_same_class_winning_cell_within_match_radius'
            reasons[reason]+=1
            x,y,w,h=a['bbox']; row=int((y+h/2)/r['height']*20); col=int((x+w/2)/r['width']*20)
            winner=int(g[row,col].argmax())
            own['background' if winner==0 else 'correct_class' if winner==a['class_id']+1 else 'wrong_object_class']+=1
        _,count=oracle_grid(r,20); collisions.update(count)
    oracle=np.stack([oracle_grid(r,20)[0] for r in val])
    oracle_report,_=score_grids(val,oracle)
    dump(out/'miss_mechanisms.json',dict(exclusive_reasons=dict(reasons),center_cell_winners=dict(own),
         annotation_collisions=dict(collisions),perfect_target_grid_with_current_decoder=oracle_report,
         caveat='Mechanistic associations at threshold 0.60, not causal interventions. The perfect target-grid result is illustrative, not a strict upper bound for arbitrary learned outputs.'))
    # Spot-check that cached INT8 grids really originate from the frozen model now on disk.
    from model import load_model
    model,_=load_model(EXPORT/'model.int8.tflite')
    picks=[val[i] for i in (0,99,249,399,499)]
    inputs,_=load_arrays(picks)
    diffs=[]
    for index,pixels in zip((0,99,249,399,499),inputs):
        current=np.asarray(model(pixels[None].astype(np.float32)/255))[0]
        diffs.append(float(np.max(np.abs(current-grids[index]))))
    assert max(diffs)==0, diffs
    print('Frozen INT8 live inference matches five saved grids exactly.',flush=True)
    # Recover training F1 only for the best float checkpoint, not a historical curve.
    import tensorflow as tf
    checkpoint,_=load_model(TRAINING/'best.keras')
    @tf.function(input_signature=[tf.TensorSpec((None,160,160,3),tf.float32)])
    def forward(batch): return checkpoint(batch,training=False)
    targets,_=validated_records(TARGETS)
    results={}
    for split in ('train','validation'):
        subset=[r for r in targets if r['split']==split]
        pixels,_=load_arrays(subset)
        predicted=np.concatenate([forward(pixels[i:i+32].astype(np.float32)/255).numpy() for i in range(0,len(pixels),32)])
        result,_=measure(subset,predicted,.6)
        results[split]=result
        print(json.dumps(dict(split=split,**result['micro'])),flush=True)
    dump(out/'checkpoint_fit.json',dict(checkpoint=str(TRAINING/'best.keras'),best_epoch=19,
         evaluation='Best float checkpoint, unaugmented inputs, inference mode, same centroid protocol',results=results,
         live_int8_max_absolute_differences=diffs, historical_training_f1_curve=None))


if __name__=='__main__': main()
