"""Reviewed Cityscapes training with a bounded sanity gate and validation-F1 checkpoints."""
import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import time

import numpy as np

from config import ROOT, CLASS_NAMES, SETTINGS, contract, output_directory
from dataset import load_arrays, validated_records


def weights_for(targets, object_weight=SETTINGS.object_weight):
    if not np.isfinite(object_weight) or object_weight <= 0:
        raise ValueError('object_weight must be finite and positive')
    weights = np.where(targets < 0, 0.0, np.where(targets > 0, object_weight, 1.0)).astype(np.float32)
    # Normalize each image's weights; ignored cells always contribute zero loss.
    return weights / np.maximum(weights.mean(axis=(1, 2), keepdims=True), 1e-8)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', required=True)
    parser.add_argument('--sanity', action='store_true', help='12 train images, 30 updates, no accuracy claim')
    parser.add_argument('--epochs', type=int, default=SETTINGS.epochs)
    parser.add_argument('--batch-size', type=int, default=SETTINGS.batch_size)
    parser.add_argument('--grid-size', type=int, choices=(20,40), default=20)
    parser.add_argument('--object-weight', type=float, default=SETTINGS.object_weight)
    parser.add_argument('--backbone-weights', default='imagenet', help='imagenet, random, or a compatible local weights file')
    parser.add_argument('--output')
    args = parser.parse_args()
    if args.epochs < 1 or args.batch_size < 1:
        parser.error('epochs and batch-size must be positive')
    if not np.isfinite(args.object_weight) or args.object_weight <= 0:
        parser.error('object-weight must be finite and positive')
    records, dataset_report = validated_records(args.manifest)
    directory = output_directory(args.output, 'cityscapes-sanity' if args.sanity else 'cityscapes-baseline')
    # Keep Keras configuration and downloaded public weights inside the isolated workspace.
    cache = ROOT / 'artifacts' / 'keras-cache'
    cache.mkdir(exist_ok=True)
    os.environ['KERAS_HOME'] = str(cache)
    import tensorflow as tf
    tf.keras.utils.set_random_seed(SETTINGS.seed)
    tf.config.experimental.enable_op_determinism()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        tf.config.set_logical_device_configuration(gpus[0], [tf.config.LogicalDeviceConfiguration(memory_limit=2300)])
    from model import build_model, load_model
    from evaluate import score_grids
    from inference import decoding_policy
    from dataset import load_rgb

    train = [r for r in records if r['split'] == 'train']
    validation = [r for r in records if r['split'] == 'validation']
    if args.sanity:
        train = [train[i] for i in np.linspace(0,len(train)-1,min(12,len(train)),dtype=int)]
        validation = [validation[i] for i in np.linspace(0,len(validation)-1,min(12,len(validation)),dtype=int)]
    x_train, y_train = load_arrays(train, args.grid_size)
    x_val, y_val = load_arrays(validation, args.grid_size)
    batch_size = min(8, args.batch_size) if args.sanity else args.batch_size
    train_weights, val_weights = weights_for(y_train, args.object_weight), weights_for(y_val, args.object_weight)
    y_train, y_val = np.maximum(y_train, 0), np.maximum(y_val, 0)

    def make_dataset(x, y, w, training=False):
        ds = tf.data.Dataset.from_tensor_slices((x,y,w))
        if training:
            ds = ds.shuffle(len(x),seed=SETTINGS.seed,reshuffle_each_iteration=True)
        def transform(image, label, weight):
            image = tf.cast(image,tf.float32)/255.0
            # Flip image and target together; no geometry-changing random crops.
            if training and not args.sanity:
                flip = tf.random.uniform(()) < .5
                image = tf.cond(flip,lambda:tf.reverse(image,[1]),lambda:image)
                label = tf.cond(flip,lambda:tf.reverse(label,[1]),lambda:label)
                weight = tf.cond(flip,lambda:tf.reverse(weight,[1]),lambda:weight)
            return image,label,weight
        ds = ds.map(transform,num_parallel_calls=1).batch(batch_size).prefetch(1)
        options = tf.data.Options()
        options.threading.private_threadpool_size = 2
        options.experimental_deterministic = True
        return ds.with_options(options)

    model = build_model(args.backbone_weights, args.grid_size)
    @tf.function(input_signature=[tf.TensorSpec((None,160,160,3),tf.float32)], reduce_retracing=True)
    def infer(batch):
        return model(batch,training=False)
    def grids(images):
        return np.concatenate([infer(images[i:i+batch_size].astype(np.float32)/255).numpy()
                               for i in range(0,len(images),batch_size)])
    def loss_for(images, labels, weights):
        p = grids(images)
        losses = tf.keras.losses.sparse_categorical_crossentropy(labels,p).numpy()
        return float(np.mean(losses*weights))

    initial_loss = loss_for(x_train[:batch_size],y_train[:batch_size],train_weights[:batch_size])
    before = [w.numpy().copy() for w in model.trainable_weights] if args.sanity else None
    metadata = {'contract':contract(args.grid_size),'synthetic':False,'sanity':args.sanity,
                'manifest':str(Path(args.manifest).resolve()),
                'manifest_sha256':hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
                'settings':dict(asdict(SETTINGS), stride=SETTINGS.image_size//args.grid_size, object_weight=args.object_weight),
                'batch_size':batch_size,'requested_epochs':args.epochs,
                'initialization':args.backbone_weights,'tensorflow':tf.__version__,'keras':tf.keras.__version__,
                'devices':[str(g) for g in gpus], 'parameters':model.count_params(),
                'train_images':len(train),'validation_images':len(validation),
                'initial_batch_loss':initial_loss,'checkpoint_selection':'maximum validation centroid F1 at threshold 0.60',
                'decoding':decoding_policy()}
    metadata['source_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in ROOT.glob('*.py')}
    metadata['pretrained_weight_sha256']={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in cache.rglob('*.h5')}
    checkpoint = directory/'best.keras'
    checkpoint.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    history=[]
    started=time.perf_counter()
    class Monitor(tf.keras.callbacks.Callback):
        def __init__(self):
            super().__init__()
            self.best=-1.0
            self.wait=0
            self.lr_wait=0
            self.best_epoch=0
        def on_epoch_end(self,epoch,logs=None):
            logs=logs or {}
            result,_=score_grids(validation,grids(x_val))
            f1=result['uncapped_micro']['f1'] or 0.0
            row={'epoch':epoch+1,'train_loss':float(logs['loss']),
                 'validation_loss':float(logs['val_loss']),'validation_f1':f1,
                 'validation_precision':result['uncapped_micro']['precision'],
                 'validation_recall':result['uncapped_micro']['recall'],
                 'per_class':result['per_class'],
                 'learning_rate':float(tf.keras.backend.get_value(model.optimizer.learning_rate)),
                 'elapsed_seconds':time.perf_counter()-started}
            if not all(np.isfinite(row[k]) for k in ('train_loss','validation_loss')):
                raise RuntimeError('Nonfinite loss, refusing checkpoint')
            if f1>self.best+1e-6:
                self.best=f1; self.best_epoch=epoch+1; self.wait=0; self.lr_wait=0
                model.save(checkpoint)
            else:
                self.wait+=1; self.lr_wait+=1
                if self.lr_wait>=SETTINGS.lr_patience:
                    model.optimizer.learning_rate.assign(max(SETTINGS.min_learning_rate,row['learning_rate']*.5))
                    self.lr_wait=0
                if self.wait>=SETTINGS.patience:
                    model.stop_training=True
            history.append(row)
            (directory/'history.json').write_text(json.dumps(history,indent=2),encoding='utf-8')
            print(json.dumps(row),flush=True)
    monitor=Monitor()
    training_data=make_dataset(x_train,y_train,train_weights,True)
    model.fit(training_data.repeat() if args.sanity else training_data,
              validation_data=make_dataset(x_val,y_val,val_weights),
              epochs=1 if args.sanity else args.epochs,
              steps_per_epoch=30 if args.sanity else None,
              callbacks=[monitor],verbose=0)
    max_change=max(float(np.max(np.abs(w.numpy()-old))) for w,old in zip(model.trainable_weights,before)) if args.sanity else None
    restored,_=load_model(checkpoint)
    # Report actual same-threshold detections from saved weights, including held-out images.
    final_grids=np.concatenate([np.asarray(restored(x_val[i:i+batch_size].astype(np.float32)/255,training=False))
                               for i in range(0,len(x_val),batch_size)])
    metrics,detections=score_grids(validation,final_grids)
    if args.sanity and (max_change<=0 or metrics['detection_count']==0):
        metadata['sanity_gate_passed']=False
    else:
        metadata['sanity_gate_passed']=True
    metadata.update({'duration_seconds':time.perf_counter()-started,'history':history,
                     'best_epoch':monitor.best_epoch,'max_trainable_weight_change':max_change,
                     'updates':int(model.optimizer.iterations.numpy()),'metrics':metrics,
                     'best_model_bytes':checkpoint.stat().st_size})
    checkpoint.with_suffix('.json').write_text(json.dumps(metadata,indent=2),encoding='utf-8')
    from evaluate import save_previews
    save_previews(directory,validation,detections,count=6 if args.sanity else 12)
    print(json.dumps({'output':str(directory),'sanity_gate_passed':metadata['sanity_gate_passed'],
                      'best_epoch':monitor.best_epoch,'duration_seconds':metadata['duration_seconds'],
                      'metrics':metrics},indent=2),flush=True)
    if args.sanity and not metadata['sanity_gate_passed']:
        raise SystemExit('Sanity gate failed: inspect gradients/detections before baseline training.')


if __name__=='__main__':
    main()
