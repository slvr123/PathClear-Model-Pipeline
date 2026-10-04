"""Compare INT8 grid models on identical inputs; never infer embedded RAM from PC timing."""
import argparse
import hashlib
import json
from pathlib import Path
import platform
import time

import numpy as np

from config import output_directory
from dataset import load_arrays, validated_records
from inference import decode_grid, decoding_policy


def distribution(values):
    return {'median':float(np.median(values)), 'p95':float(np.percentile(values,95)),
            'samples':len(values)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True, action='append', help='Repeat for each INT8 model')
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--iterations', type=int, default=200)
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.iterations < 1:
        parser.error('iterations must be positive')
    records, _ = validated_records(args.manifest)
    records = [r for r in records if r['split']=='validation'][:100]
    pixels, _ = load_arrays(records)
    inputs = pixels.astype(np.float32)/255
    from model import load_model
    import tensorflow as tf
    loaded = []
    for path in args.model:
        if Path(path).suffix != '.tflite':
            parser.error('Only INT8 TFLite models can be compared by this benchmark')
        model, metadata = load_model(path)
        for i in range(20):
            model(inputs[i%len(inputs):i%len(inputs)+1])
        loaded.append((path,model,metadata,[],[],[]))
    # Alternate model order each iteration to reduce systematic warm/cache-order bias.
    for iteration in range(args.iterations):
        index = iteration % len(inputs)
        sample = inputs[index:index+1]
        record = records[index]
        for path,model,metadata,calls,decodes,combined in (loaded if iteration%2==0 else loaded[::-1]):
            start = time.perf_counter()
            grid = np.asarray(model(sample))[0]
            predicted = time.perf_counter()
            decode_grid(grid, record['width'], record['height'])
            ended = time.perf_counter()
            calls.append((predicted-start)*1000)
            decodes.append((ended-predicted)*1000)
            combined.append((ended-start)*1000)
    results = []
    for path,model,metadata,calls,decodes,combined in loaded:
        tensors = model.interpreter.get_tensor_details()
        sizes = [int(np.prod(t['shape']))*np.dtype(t['dtype']).itemsize for t in tensors]
        g = metadata['contract']['grid_size']
        results.append({'model':str(Path(path).resolve()), 'sha256':hashlib.sha256(Path(path).read_bytes()).hexdigest(),
                        'grid_size':g, 'model_bytes':Path(path).stat().st_size,
                        'model_call_ms':distribution(calls), 'decode_ms':distribution(decodes),
                        'model_plus_decode_ms':distribution(combined),
                        'int8_input_tensor_bytes':int(np.prod(model.input['shape'])),
                        'int8_output_tensor_bytes':int(np.prod(model.output['shape'])),
                        'host_float_input_bytes':int(inputs[0].nbytes),
                        'host_float_output_bytes':g*g*3*4,
                        'largest_listed_tensor_bytes':max(sizes),
                        'embedded_peak_ram_bytes':None, 'embedded_latency_ms':None, 'embedded_fps':None})
    directory = output_directory(args.output,'grid-benchmark')
    report = {'runtime':f'TensorFlow {tf.__version__}, TFLite CPU XNNPACK, one thread per model',
              'platform':platform.platform(), 'warmup_calls_per_model':20,
              'input_ids':[r['id'] for r in records], 'decoding':decoding_policy(),
              'scope':'Prepared RGB input; model call includes quantization, output copy/dequantization; excludes camera, resize, MicroSeg and decision logic.',
              'memory_limit':'Tensor buffer sizes are observations, not total peak RAM or a TFLite Micro arena estimate. Camera/runtime/operator scratch buffers and firmware remain unmeasured.',
              'results':results}
    (directory/'report.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(json.dumps(report,indent=2))
    print(f'Report: {directory}')


if __name__ == '__main__':
    main()
