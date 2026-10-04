"""Paired host timing of baseline policies and an optional separate INT8 candidate."""
import argparse
import json
import time
import numpy as np
from analyze_baseline_v1 import EXPORT, MANIFEST, dump, sha
from benchmark_grids import distribution
from config import output_directory
from dataset import load_arrays, validated_records
from inference import decode_grid


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',required=True)
    parser.add_argument('--candidate-model')
    parser.add_argument('--candidate-threshold',type=float,default=.7)
    args=parser.parse_args(); out=output_directory(args.output,'policy-benchmark')
    records,_=validated_records(MANIFEST); records=[r for r in records if r['split']=='validation'][:100]
    pixels,_=load_arrays(records); inputs=pixels.astype(np.float32)/255
    from model import load_model
    model,_=load_model(EXPORT/'model.int8.tflite')
    models={'baseline_v1':model,'v2_inference_candidate':model}
    policies=[dict(name='baseline_v1',decoder='cell_nms',threshold=.6),dict(name='v2_inference_candidate',decoder='cells',threshold=.8)]
    if args.candidate_model:
        from pathlib import Path
        candidate,_=load_model(args.candidate_model)
        models['v2_weight10_candidate']=candidate
        policies.append(dict(name='v2_weight10_candidate',decoder='cells',threshold=args.candidate_threshold,
                             model_sha256=sha(Path(args.candidate_model))))
    samples={p['name']:{k:[] for k in ('model_call_ms','decode_ms','total_ms')} for p in policies}
    for loaded in {id(m):m for m in models.values()}.values():
        for i in range(20): loaded(inputs[i:i+1])
    for i in range(200):
        index=i%100; r=records[index]
        for p in (policies if i%2==0 else policies[::-1]):
            start=time.perf_counter(); grid=np.asarray(models[p['name']](inputs[index:index+1]))[0]; mid=time.perf_counter()
            decode_grid(grid,r['width'],r['height'],p['threshold'],0,p['decoder']); end=time.perf_counter()
            target=samples[p['name']]
            target['model_call_ms'].append((mid-start)*1000); target['decode_ms'].append((end-mid)*1000); target['total_ms'].append((end-start)*1000)
    dump(out/'report.json',dict(model_sha256=sha(EXPORT/'model.int8.tflite'),warmup=20,iterations_per_policy=200,
         input_ids=[r['id'] for r in records],
         runtime='Desktop TFLite CPU, one thread per model, alternating policy order; same prepared inputs',
         scope='Includes quantize, invoke, output copy and decode. Excludes camera, resize, decision logic and MicroSeg.',
         results=[dict(**p,**{k:distribution(v) for k,v in samples[p['name']].items()}) for p in policies],
         embedded_peak_ram=None,embedded_latency=None,embedded_fps=None))
    print((out/'report.json').read_text())


if __name__=='__main__': main()
