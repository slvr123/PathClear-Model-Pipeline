"""Evaluate the frozen epoch-19 model; never calls fit or an optimizer update."""
import csv
import hashlib
import json
import time
from pathlib import Path

import keras
import numpy as np
import tensorflow as tf
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image

import segmentation_metrics  # Register the saved custom metric.
from dataset_loader import collect_pairs, load_pair, IMAGE_HEIGHT, IMAGE_WIDTH, NUM_CLASSES

ROOT = Path(__file__).resolve().parent
CHECKPOINT = ROOT / 'checkpoints/baseline-5epochs/best.keras'
OUT = CHECKPOINT.parent / 'evaluation'
NAMES = ['road','sidewalk','building','wall','fence','pole','traffic light','traffic sign',
         'vegetation','terrain','sky','person','rider','car','truck','bus','train','motorcycle','bicycle']
COLORS = np.array([(128,64,128),(244,35,232),(70,70,70),(102,102,156),(190,153,153),
    (153,153,153),(250,170,30),(220,220,0),(107,142,35),(152,251,152),(70,130,180),
    (220,20,60),(255,0,0),(0,0,142),(0,0,70),(0,60,100),(0,80,100),(0,0,230),(119,11,32)],dtype=np.uint8)


def digest(path):
    with path.open('rb') as f:
        return hashlib.file_digest(f,'sha256').hexdigest()


def write_csv(path, rows, fields=None):
    with path.open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=fields or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def scores(cm):
    tp=np.diag(cm).astype(np.float64)
    support=cm.sum(axis=1)
    predicted=cm.sum(axis=0)
    union=support+predicted-tp
    iou=np.divide(tp,union,out=np.zeros(19),where=union>0)
    dice=np.divide(2*tp,support+predicted,out=np.zeros(19),where=(support+predicted)>0)
    return support,predicted,iou,dice,union>0


def save_matrix(path, matrix):
    with path.open('w',newline='') as f:
        w=csv.writer(f)
        w.writerow(['true_class / predicted_class']+NAMES)
        for name,row in zip(NAMES,matrix):
            w.writerow([name]+row.tolist())


def main():
    start=time.perf_counter()
    if OUT.exists() and any(OUT.iterdir()):
        raise FileExistsError(f'Refusing to overwrite {OUT}')
    OUT.mkdir(parents=True,exist_ok=True)
    predictions=OUT/'predictions'
    qualitative=OUT/'qualitative'
    predictions.mkdir()
    qualitative.mkdir()
    before_hash=digest(CHECKPOINT)
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu,True)
    model=keras.models.load_model(CHECKPOINT)
    assert model.input_shape==(None,512,1024,3)
    assert model.output_shape==(None,512,1024,19)
    assert NUM_CLASSES==19 and model.count_params()==1160707
    assert int(model.optimizer.iterations.numpy())==56525
    assert not model.losses, 'Unexpected regularization loss; review loss comparison.'
    before_weights=model.get_weights()
    before_optimizer=[v.numpy().copy() for v in model.optimizer.variables]
    print('Loaded epoch 19: input',model.input_shape,'output',model.output_shape,'optimizer steps 56525',flush=True)
    print('Evaluation only: model(..., training=False).',flush=True)
    image_paths,mask_paths=collect_pairs('val')
    assert len(image_paths)==len(mask_paths)==500
    data=tf.data.Dataset.from_tensor_slices((image_paths,mask_paths)).map(load_pair,num_parallel_calls=tf.data.AUTOTUNE).batch(1).prefetch(1)

    @tf.function(input_signature=[tf.TensorSpec((1,512,1024,3),tf.float32)])
    def infer(images):
        return model(images,training=False)

    cm=np.zeros((19,19),dtype=np.int64)
    per_image=[]
    image_cms=[]
    loss_sum=0.0
    ignored=0
    infer_start=time.perf_counter()
    first_device=None
    for i,(images,targets,weights) in enumerate(data):
        logits=infer(images)
        if first_device is None:
            first_device=logits.device
        tf.debugging.assert_all_finite(logits,'Non-finite logits')
        loss=float(model.loss(targets,logits,sample_weight=weights).numpy())
        assert np.isfinite(loss)
        prediction=tf.argmax(logits[0],axis=-1,output_type=tf.int32).numpy()
        truth=targets[0].numpy()
        valid=weights[0].numpy()>0
        assert np.all((truth[valid]>=0)&(truth[valid]<19))
        local=np.bincount(truth[valid].astype(np.int64)*19+prediction[valid],minlength=19*19).reshape(19,19)
        assert local.sum()==int(valid.sum())
        cm+=local
        loss_sum+=loss
        ignored+=int((~valid).sum())
        support,predicted,iou,dice,present=scores(local)
        image_cms.append(local)
        per_image.append({'index':i,'image':image_paths[i],'mask':mask_paths[i],
            'valid_pixels':int(valid.sum()),'ignored_pixels':int((~valid).sum()),
            'pixel_accuracy':float(np.trace(local)/local.sum()),
            'mean_iou':float(iou[present].mean()),'macro_dice':float(dice[present].mean()),'loss':loss})
        Image.fromarray(prediction.astype(np.uint8)).save(predictions/f'{i:03d}.png')
        if (i+1)%50==0:
            print(f'Evaluated {i+1}/500 images; elapsed {time.perf_counter()-infer_start:.1f}s',flush=True)
    eval_seconds=time.perf_counter()-infer_start
    assert len(per_image)==500
    assert int(cm.sum())+ignored==500*512*1024
    support,predicted,iou,dice,present=scores(cm)
    overall={'pixel_accuracy':float(np.trace(cm)/cm.sum()),'mean_iou':float(iou[present].mean()),
             'macro_dice':float(dice[present].mean()),'loss':loss_sum/500}
    assert present.all()
    # Independently cross-check the registered Keras metric reduction on the same exact counts.
    for metric,key in [(segmentation_metrics.make_metrics()[1],'mean_iou'),(segmentation_metrics.make_metrics()[2],'macro_dice')]:
        metric.total_cm.assign(cm)
        np.testing.assert_allclose(float(metric.result()),overall[key],atol=1e-6)
    classes=[]
    for c,name in enumerate(NAMES):
        classes.append({'class_id':c,'class_name':name,'iou':float(iou[c]),'dice':float(dice[c]),
            'support_pixels':int(support[c]),'support_percent':float(100*support[c]/cm.sum()),
            'predicted_pixels':int(predicted[c]),'true_positive_pixels':int(cm[c,c]),
            'recall':float(cm[c,c]/support[c]) if support[c] else 0.0,
            'precision':float(cm[c,c]/predicted[c]) if predicted[c] else 0.0,
            'images_with_class':sum(int(x[c].sum()>0) for x in image_cms)})
    write_csv(OUT/'per_class_metrics.csv',classes)
    write_csv(OUT/'per_class_metrics_by_iou.csv',sorted(classes,key=lambda r:r['iou']))
    write_csv(OUT/'per_image_metrics.csv',per_image)
    save_matrix(OUT/'confusion_matrix.csv',cm)
    normalized=cm/support[:,None]
    save_matrix(OUT/'confusion_matrix_normalized.csv',normalized)
    fig,ax=plt.subplots(figsize=(15,13))
    img=ax.imshow(normalized*100,cmap='Blues',vmin=0,vmax=100)
    ax.set_xticks(range(19),NAMES,rotation=60,ha='right')
    ax.set_yticks(range(19),NAMES)
    ax.set_xlabel('Predicted class')
    ax.set_ylabel('True class')
    ax.set_title('Epoch 19 - all 500 validation images\nRow-normalized confusion (% of each true class; void excluded)')
    for r in range(19):
        for c in range(19):
            value=normalized[r,c]*100
            if value>=0.5:
                ax.text(c,r,f'{value:.1f}',ha='center',va='center',fontsize=7,color='white' if value>55 else 'black')
    fig.colorbar(img,ax=ax,label='% of true-class pixels',fraction=0.04)
    fig.tight_layout()
    fig.savefig(OUT/'confusion_matrix.png',dpi=160)
    plt.close(fig)
    pairs=[]
    for r in range(19):
        for c in range(19):
            if r!=c and cm[r,c]>0:
                pairs.append({'true_class':NAMES[r],'predicted_class':NAMES[c],'pixels':int(cm[r,c]),
                    'percent_of_true_class':float(normalized[r,c]*100)})
    write_csv(OUT/'confusion_pairs_by_count.csv',sorted(pairs,key=lambda r:r['pixels'],reverse=True))
    write_csv(OUT/'confusion_pairs_by_rate.csv',sorted(pairs,key=lambda r:r['percent_of_true_class'],reverse=True))

    ranked=sorted(per_image,key=lambda r:(r['mean_iou'],r['index']))
    choices=[]
    used=set()
    for band,qs in [('weak',[.05,.10,.15]),('middle',[.45,.50,.55]),('strong',[.85,.90,.95])]:
        for q in qs:
            row=ranked[round(q*(len(ranked)-1))]
            choices.append((row,f'{band}: per-image mIoU quantile {q:.0%}'))
            used.add(row['index'])
    for target_class in [11,12,17]:
        candidates=[r for r in per_image if r['index'] not in used and image_cms[r['index']][target_class].sum()>=200]
        row=min(candidates,key=lambda r:scores(image_cms[r['index']])[2][target_class])
        choices.append((row,f'Diagnostic: weakest {NAMES[target_class]} IoU with >=200 true pixels'))
        used.add(row['index'])
    selection=[]
    for number,(row,reason) in enumerate(choices,1):
        i=row['index']
        image,truth,weight=load_pair(image_paths[i],mask_paths[i])
        pred=np.array(Image.open(predictions/f'{i:03d}.png'))
        true=truth.numpy()
        valid=weight.numpy()>0
        gt=COLORS[true].copy()
        gt[~valid]=0
        # Red = wrong valid pixel; gray = correct valid pixel; black = void.
        error=np.zeros((*true.shape,3),dtype=np.uint8)
        error[valid]=[65,65,65]
        error[valid&(pred!=true)]=[240,45,45]
        rgb=np.array(Image.open(image_paths[i]).convert('RGB'))
        fig,axes=plt.subplots(2,2,figsize=(16,12))
        for ax,pixels,title in zip(axes.flat,[rgb,gt,COLORS[pred],error],['Original RGB','Ground truth (black = void)','Epoch-19 prediction','Errors: red wrong, gray correct, black ignored']):
            ax.imshow(pixels)
            ax.set_title(title)
            ax.axis('off')
        fig.suptitle(f"{number:02d} | {Path(image_paths[i]).stem}\n{reason}; image mIoU={row['mean_iou']:.3f}",fontsize=11)
        fig.legend(handles=[Patch(color=color/255,label=name) for color,name in zip(COLORS,NAMES)],loc='lower center',ncol=10,fontsize=8)
        fig.subplots_adjust(left=.02,right=.98,bottom=.10,top=.90,wspace=.04,hspace=.22)
        filename=f'{number:02d}_{Path(image_paths[i]).stem}.png'
        fig.savefig(qualitative/filename,dpi=120)
        plt.close(fig)
        selection.append({'example':number,'index':i,'selection_reason':reason,'image_miou':row['mean_iou'],'file':'qualitative/'+filename,'source_image':image_paths[i]})
    write_csv(OUT/'qualitative_selection.csv',selection)
    for old,new in zip(before_weights,model.get_weights()):
        np.testing.assert_array_equal(old,new)
    for old,new in zip(before_optimizer,model.optimizer.variables):
        np.testing.assert_array_equal(old,new.numpy())
    after_hash=digest(CHECKPOINT)
    assert after_hash==before_hash
    with (CHECKPOINT.parent/'history.csv').open() as f:
        history=list(csv.DictReader(f))
    recorded=history[18]
    deltas={key:overall[key]-float(recorded['val_'+('dice' if key=='macro_dice' else key)]) for key in overall}
    assert max(abs(v) for v in deltas.values())<0.002, f'Unexpected history discrepancy: {deltas}'
    result={'checkpoint':str(CHECKPOINT),'epoch':19,'sha256_before':before_hash,'sha256_after':after_hash,
        'model_weights_unchanged':True,'optimizer_state_unchanged':True,'optimizer_iterations':56525,
        'input_shape':list(model.input_shape),'output_shape':list(model.output_shape),'parameters':model.count_params(),
        'images':500,'batch_size':1,'resolution':[512,1024],'valid_pixels':int(cm.sum()),'void_pixels':ignored,
        'metrics':overall,'difference_from_epoch19_history':deltas,'device':first_device,
        'evaluation_seconds':eval_seconds,'total_seconds':time.perf_counter()-start,
        'loss_reduction':'Same compiled SparseCategoricalCrossentropy(from_logits=True), sum_over_batch_size; void contributes zero but remains in denominator.',
        'macro_averaging':'Global valid-pixel confusion counts, unweighted average of classes with nonzero union; all 19 present.',
        'qualitative_selection':'9 fixed per-image mIoU quantiles and 3 targeted class failures; see manifest, not an unbiased visual sample.'}
    (OUT/'metrics.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps(result,indent=2),flush=True)

if __name__=='__main__':
    main()

