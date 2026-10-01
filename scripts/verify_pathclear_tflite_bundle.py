"""Run the packaged golden input without needing the Keras model or dataset."""
import argparse
import hashlib
import json
from pathlib import Path
import time

import numpy as np
from PIL import Image
try:
    from ai_edge_litert.interpreter import Interpreter
except ImportError:
    from tensorflow.lite.python.interpreter import Interpreter


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--bundle', type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    bundle = args.bundle.resolve()
    info = json.loads((bundle / 'model_interface.json').read_text())
    model_path = bundle / info['model_file']
    assert hashlib.sha256(model_path.read_bytes()).hexdigest() == info['sha256'], 'Model hash mismatch'
    raw = bundle / info['golden_input']
    assert raw.stat().st_size == 6291456, 'Input byte count mismatch'
    x = np.fromfile(raw, dtype='<f4').reshape(1, 512, 1024, 3)
    assert np.isfinite(x).all() and x.min() >= 0 and x.max() <= 1
    interpreter = Interpreter(model_path=str(model_path), num_threads=4)
    interpreter.allocate_tensors()
    inputs, outputs = interpreter.get_input_details(), interpreter.get_output_details()
    assert len(inputs) == len(outputs) == 1
    inp, out = inputs[0], outputs[0]
    assert inp['shape'].tolist() == [1, 512, 1024, 3] and inp['dtype'] == np.float32
    assert out['shape'].tolist() == [1, 512, 1024, 10] and out['dtype'] == np.float32
    interpreter.set_tensor(inp['index'], x)
    tick = time.perf_counter()
    interpreter.invoke()
    elapsed_ms = (time.perf_counter() - tick) * 1000
    logits = interpreter.get_tensor(out['index'])
    assert np.isfinite(logits).all()
    assert np.all(logits[..., 7:] == np.float32(-1e9))
    predicted = logits[0].argmax(axis=-1).astype(np.uint8)
    assert predicted.max() < 7
    expected = np.asarray(Image.open(bundle / 'golden_test/expected_tflite_ids.png'))
    agreement = float(np.mean(predicted == expected))
    assert agreement >= .999, f'Golden labels mismatch: {agreement:.6%}'
    print(json.dumps({'status': 'passed', 'label_agreement': agreement,
                      'different_pixels': int(np.count_nonzero(predicted != expected)),
                      'invoke_ms_including_first_call': elapsed_ms,
                      'input_shape': inp['shape'].tolist(), 'output_shape': out['shape'].tolist()}, indent=2))


if __name__ == '__main__':
    main()
