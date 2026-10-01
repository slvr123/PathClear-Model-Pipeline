"""Export and check the preserved epoch-16 model without training or quantization."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time

# Use CPU for reproducible converter/runtime comparisons without GPU allocations.
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
import keras
import numpy as np
from PIL import Image
import tensorflow as tf
from tensorflow.lite.python import schema_py_generated as schema

ROOT = Path(__file__).resolve().parents[1]
ARCHIVE = ROOT / 'models/pathclear_cityscapes_epoch16'
sys.path.insert(0, str(ARCHIVE / 'source'))
import pathclear_stages  # Registers the archived custom objects.
from pathclear_taxonomy import CLASS_NAMES, COLORS


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def save_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2) + '\n', encoding='utf-8')


def tensor_info(detail):
    return {'name': detail['name'], 'index': int(detail['index']),
            'shape': detail['shape'].tolist(), 'shape_signature': detail['shape_signature'].tolist(),
            'dtype': np.dtype(detail['dtype']).name,
            'bytes': int(np.prod(detail['shape']) * np.dtype(detail['dtype']).itemsize)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    output = args.output_dir.resolve()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    checkpoint = ARCHIVE / 'pathclear_cityscapes_epoch16.keras'
    info = json.loads((ARCHIVE / 'model_info.json').read_text(encoding='utf-8-sig'))
    protected = [checkpoint, ROOT / info['source_checkpoint'],
                 ROOT / 'checkpoints/baseline-5epochs/best.keras']
    protected += list((ARCHIVE / 'source').glob('*.py'))
    hashes = {str(p.relative_to(ROOT)): digest(p) for p in protected}
    assert digest(checkpoint) == info['sha256']
    assert info['epoch'] == 16
    assert Path(pathclear_stages.__file__).resolve().parent == ARCHIVE / 'source'
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    model = keras.models.load_model(checkpoint, compile=False)
    assert tuple(model.get_layer('pathclear_active_classes').active_ids) == tuple(range(7))
    before_weights = model.get_weights()
    output.mkdir(parents=True)
    started = time.perf_counter()
    print('Exporting fixed-batch inference graph with seven active classes.', flush=True)
    with tempfile.TemporaryDirectory(prefix='pathclear-tflite-') as temporary:
        # ExportArchive tracks Keras variables explicitly so conversion freezes
        # their learned values, rather than emitting unresolved resource reads.
        archive = keras.export.ExportArchive()
        archive.track(model)
        archive.add_endpoint(
            name='serve', fn=lambda image: {'logits': model(image, training=False)},
            input_signature=[tf.TensorSpec([1, 512, 1024, 3], tf.float32, name='image')])
        archive.write_out(temporary, verbose=False)
        converter = tf.lite.TFLiteConverter.from_saved_model(temporary)
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS]
        converter.optimizations = []
        converter.allow_custom_ops = False
        buffer = converter.convert()
    print(f'Converted {len(buffer):,} bytes. Checking interpreter and predictions.', flush=True)
    interpreter = tf.lite.Interpreter(model_content=buffer, num_threads=4)
    interpreter.allocate_tensors()
    ins, outs = interpreter.get_input_details(), interpreter.get_output_details()
    assert len(ins) == len(outs) == 1
    inp, out = ins[0], outs[0]
    assert inp['shape'].tolist() == [1, 512, 1024, 3]
    assert out['shape'].tolist() == [1, 512, 1024, 10]
    assert inp['dtype'] == out['dtype'] == np.float32
    assert inp['shape_signature'].tolist() == inp['shape'].tolist()
    assert out['shape_signature'].tolist() == out['shape'].tolist()

    flat = schema.Model.GetRootAsModel(buffer, 0)
    builtin_names = {v: k for k, v in vars(schema.BuiltinOperator).items() if isinstance(v, int)}
    operators = []
    for i in range(flat.OperatorCodesLength()):
        op = flat.OperatorCodes(i)
        assert op.BuiltinCode() != schema.BuiltinOperator.CUSTOM
        assert builtin_names.get(op.BuiltinCode()) not in {'VAR_HANDLE', 'READ_VARIABLE', 'ASSIGN_VARIABLE'}
        operators.append({'name': builtin_names.get(op.BuiltinCode(), str(op.BuiltinCode())),
                          'version': op.Version()})

    manifest = ROOT / 'checkpoints/pathclear_10class/cityscapes_epochs12_16_2026-09-29/dataset_manifest.json'
    selected = {}
    for pair in json.loads(manifest.read_text())['val']:
        path = Path(pair['image'])
        selected.setdefault(path.parent.name, path)
    assert len(selected) == 3
    samples = []
    warm_times = []
    golden = output / 'golden_test'
    golden.mkdir()

    def compare(name, x, image_path=None, make_golden=False):
        reference = model(x, training=False).numpy()
        interpreter.set_tensor(inp['index'], x)
        t0 = time.perf_counter()
        interpreter.invoke()
        invocation_ms = (time.perf_counter() - t0) * 1000
        actual = interpreter.get_tensor(out['index'])
        assert np.isfinite(actual).all()
        np.testing.assert_array_equal(actual[..., 7:], reference[..., 7:])
        ref_mask = np.argmax(reference[0], axis=-1).astype(np.uint8)
        actual_mask = np.argmax(actual[0], axis=-1).astype(np.uint8)
        assert actual_mask.max() < 7
        error = np.abs(actual[..., :7] - reference[..., :7])
        agreement = float(np.mean(ref_mask == actual_mask))
        # Conversion checks, not dataset accuracy or a mobile acceptance guarantee.
        assert agreement >= .999, (name, agreement)
        assert float(error.mean()) < .001 and float(error.max()) < .01, (name, error.max())
        record = {'name': name, 'source_image': str(image_path) if image_path else None,
                  'pixel_label_agreement': agreement,
                  'different_pixels': int(np.count_nonzero(ref_mask != actual_mask)),
                  'active_logit_mean_abs_error': float(error.mean()),
                  'active_logit_max_abs_error': float(error.max()),
                  'reserved_logits_exact': True, 'predicted_ids': np.unique(actual_mask).tolist(),
                  'cpu_invoke_ms': invocation_ms}
        samples.append(record)
        if make_golden:
            x.astype('<f4').tofile(golden / 'input_rgb_float32_le.bin')
            shutil.copy2(image_path, golden / 'source_rgb.png')
            Image.fromarray(ref_mask).save(golden / 'expected_keras_ids.png')
            Image.fromarray(actual_mask).save(golden / 'expected_tflite_ids.png')
            Image.fromarray(COLORS[actual_mask]).save(golden / 'expected_color_preview.png')
            for _ in range(5):
                interpreter.set_tensor(inp['index'], x)
                tick = time.perf_counter()
                interpreter.invoke()
                warm_times.append((time.perf_counter() - tick) * 1000)
        print(json.dumps(record), flush=True)

    # Warming the interpreter is separate from measured sample calls.
    interpreter.set_tensor(inp['index'], np.zeros((1, 512, 1024, 3), np.float32))
    interpreter.invoke()
    for i, (city, path) in enumerate(sorted(selected.items())):
        rgb = tf.io.decode_png(tf.io.read_file(str(path)), channels=3)
        x = (tf.image.resize(rgb, [512, 1024], method='bilinear') / 255.0).numpy()[None]
        compare(city, x, path, make_golden=i == 0)
    compare('synthetic_black', np.zeros((1, 512, 1024, 3), np.float32))
    yy, xx = np.mgrid[0:512, 0:1024]
    ramp = np.stack([xx / 1023, yy / 511, ((xx // 32 + yy // 32) % 2)], axis=-1).astype(np.float32)
    compare('synthetic_rgb_ramp', ramp[None])
    for old, new in zip(before_weights, model.get_weights()):
        np.testing.assert_array_equal(old, new)
    assert all(digest(ROOT / path) == value for path, value in hashes.items())
    model_path = output / 'pathclear_epoch16_float32.tflite'
    # Publish only validated bytes; avoid rename visibility delays on WSL/OneDrive.
    model_path.write_bytes(buffer)
    assert digest(model_path) == hashlib.sha256(buffer).hexdigest()
    labels = [{'id': i, 'name': name, 'active': i < 7, 'color_rgb': COLORS[i].tolist()}
              for i, name in enumerate(CLASS_NAMES)]
    save_json(output / 'labels.json', labels)
    (output / 'labels.txt').write_text('\n'.join(CLASS_NAMES) + '\n', encoding='utf-8')
    save_json(output / 'model_interface.json', {
        'interface_version': 'pathclear-epoch16-fp32-v1', 'model_file': model_path.name,
        'sha256': digest(model_path), 'source_keras_sha256': info['sha256'],
        'input': tensor_info(inp), 'output': tensor_info(out),
        'signature_list': interpreter.get_signature_list(), 'operators': operators,
        'layout': 'NHWC: batch,height,width,channels', 'input_color_order': 'RGB',
        'normalization': 'float32(rgb_byte) / 255.0',
        'resize': 'Direct bilinear to width=1024,height=512; TensorFlow half-pixel centers; no crop, padding or uint8 rounding after resize',
        'orientation': 'Upright image in walking direction; apply camera rotation once before resize; no mirror',
        'decoding': 'argmax of 10 raw logits at each pixel; no softmax required for class IDs',
        'active_ids': list(range(7)), 'disabled_ids': [7, 8, 9], 'disabled_logit': -1e9,
        'ignore_label': 255, 'ignore_is_model_output': False,
        'quantization': 'none; float32 weights, input and output',
        'builtins_only': True, 'flex_or_custom_ops_required': False,
        'tensorflow_converter_version': tf.__version__, 'keras_version': keras.__version__,
        'tested_runtime': f'tf.lite.Interpreter in TensorFlow {tf.__version__}, WSL CPU, 4 threads',
        'phone_runtime_and_delegates_verified': False,
        'golden_input': 'golden_test/input_rgb_float32_le.bin',
        'golden_input_encoding': 'Little-endian float32, row-major NHWC, exactly 6291456 bytes',
        'scope': 'Seven-active-class integration model; local-only classes untrained; not a finalized mobile interface agreement',
    })
    save_json(output / 'validation_report.json', {
        'status': 'passed', 'model_sha256': digest(model_path), 'bytes': len(buffer),
        'elapsed_seconds': time.perf_counter() - started, 'samples': samples,
        'selection': 'First validation filename in each Cityscapes city plus black and RGB-ramp synthetic probes',
        'thresholds': {'minimum_label_agreement': .999, 'maximum_mean_abs_logit_error': .001,
                       'maximum_abs_logit_error': .01, 'reserved_logits': 'exact'},
        'warm_cpu_invoke_ms': warm_times, 'warm_cpu_median_ms': float(np.median(warm_times)),
        'timing_scope': 'Five warm invokes on first image, laptop CPU, 4 threads; excludes preprocessing and output copying; not phone FPS',
        'source_weights_unchanged': True, 'protected_hashes_unchanged': hashes,
        'limitations': 'Diagnostic conversion parity only; no full TFLite accuracy evaluation, phone test, local accuracy, or training.'
    })
    print(f'PASS: {model_path}; {len(buffer) / (1024 * 1024):.2f} MiB; all five parity checks passed.', flush=True)


if __name__ == '__main__':
    main()
