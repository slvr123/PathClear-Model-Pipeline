"""Frozen-baseline inference on five supplied, unlabeled internet photographs."""
import json
import time
from pathlib import Path
import numpy as np
import tensorflow as tf
import keras
from PIL import Image
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from evaluate_baseline import NAMES, COLORS, digest

ROOT = Path(__file__).resolve().parent
OUT = ROOT / 'checkpoints/baseline-5epochs/ph_samples_2026-09-21'
CHECKPOINT = ROOT / 'checkpoints/baseline-5epochs/best.keras'

def main():
    start = time.perf_counter()
    for gpu in tf.config.list_physical_devices('GPU'):
        tf.config.experimental.set_memory_growth(gpu, True)
    before_hash = digest(CHECKPOINT)
    model = keras.models.load_model(CHECKPOINT)
    assert model.input_shape == (None,512,1024,3)
    assert model.output_shape == (None,512,1024,19)
    weights = model.get_weights()
    optimizer = [v.numpy().copy() for v in model.optimizer.variables]
    rows = []
    for i in range(1,6):
        path = OUT / 'originals' / f'sample_{i:02d}.png'
        rgb = np.array(Image.open(path).convert('RGB'))
        image = tf.image.resize(rgb, [512,1024], method='bilinear') / 255.0
        t = time.perf_counter()
        logits = model(image[None], training=False)
        mask = tf.argmax(logits[0], axis=-1).numpy().astype('uint8')
        elapsed = time.perf_counter()-t
        Image.fromarray(mask).save(OUT / f'sample_{i:02d}_train_ids.png')
        restored = np.array(Image.fromarray(mask).resize((rgb.shape[1],rgb.shape[0]), Image.Resampling.NEAREST))
        colored = COLORS[restored]
        overlay = (0.55*rgb+0.45*colored).astype('uint8')
        Image.fromarray(colored).save(OUT / f'sample_{i:02d}_segmentation.png')
        Image.fromarray(overlay).save(OUT / f'sample_{i:02d}_overlay.png')
        fig,axes = plt.subplots(1,3,figsize=(18,8))
        for ax,arr,title in zip(axes,[rgb,colored,overlay],['Original','Predicted classes','Overlay']):
            ax.imshow(arr); ax.set_title(title); ax.axis('off')
        fig.suptitle(f'Sample {i} | Frozen epoch-19 Fast-SCNN | Unlabeled: accuracy unavailable',fontsize=15)
        fig.legend(handles=[Patch(color=COLORS[c]/255,label=NAMES[c]) for c in range(19)],loc='lower center',ncol=7,fontsize=10)
        fig.subplots_adjust(left=.01,right=.99,top=.90,bottom=.17,wspace=.03)
        fig.savefig(OUT / f'sample_{i:02d}_comparison.png',dpi=120)
        plt.close(fig)
        counts = np.bincount(mask.ravel(),minlength=19)
        rows.append({'sample':i,'original_shape':list(rgb.shape),'inference_seconds':elapsed,
                     'predicted_area_percent':{name:float(counts[c]/mask.size*100) for c,name in enumerate(NAMES)}})
        print(f'Sample {i} complete ({elapsed:.2f}s)',flush=True)
    assert all(np.array_equal(a,b) for a,b in zip(weights,model.get_weights()))
    assert all(np.array_equal(a,b.numpy()) for a,b in zip(optimizer,model.optimizer.variables))
    assert digest(CHECKPOINT)==before_hash
    data={'checkpoint_sha256':before_hash,'weights_unchanged':True,'optimizer_unchanged':True,
          'optimizer_iterations':int(model.optimizer.iterations.numpy()),'samples':rows,
          'preprocessing':'Existing pipeline: bilinear resize to 512x1024, divide by 255. Direct resize changes aspect ratio; masks restored by nearest neighbor for display.',
          'accuracy':None,'reason':'No ground-truth masks. Class area percentages are not accuracy or confidence.',
          'total_seconds':time.perf_counter()-start}
    (OUT/'inference_manifest.json').write_text(json.dumps(data,indent=2))
    lines=['# Philippine internet sample inspection','',
           'I ran the unchanged epoch-19 checkpoint on five supplied internet photographs.','No training occurred.','',
           'Numerical local accuracy is unavailable because these images have no ground-truth masks.',
           'I used the existing direct resize to 512 x 1024, which distorts non-2:1 images, particularly the portrait photograph.',
           'The displayed predictions are restored to each original image size with nearest-neighbor interpolation.','']
    for i in range(1,6):
        lines += [f'## Sample {i}','',f'![Sample {i}](sample_{i:02d}_comparison.png)','']
    (OUT/'gallery.md').write_text('\n'.join(lines))
    print(json.dumps({k:v for k,v in data.items() if k!='samples'},indent=2),flush=True)

if __name__=='__main__':
    main()
