"""Render saved evaluation predictions; no model loading or inference."""
import csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from PIL import Image
from scripts.evaluate_baseline import OUT,COLORS,NAMES

with (OUT/'qualitative_selection.csv').open() as f: rows=list(csv.DictReader(f))
with (OUT/'per_image_metrics.csv').open() as f: metrics=list(csv.DictReader(f))
for row in rows:
    i=int(row['index'])
    rgb=np.array(Image.open(row['source_image']).convert('RGB'))
    true=np.array(Image.open(metrics[i]['mask']).resize((1024,512),Image.Resampling.NEAREST))
    valid=true!=255
    gt=np.zeros((*true.shape,3),dtype=np.uint8)
    gt[valid]=COLORS[true[valid]]
    pred=np.array(Image.open(OUT/'predictions'/f'{i:03d}.png'))
    error=np.zeros((*true.shape,3),dtype=np.uint8)
    error[valid]=[65,65,65]
    error[valid&(pred!=true)]=[240,45,45]
    fig,axes=plt.subplots(2,2,figsize=(16,12))
    for ax,pixels,title in zip(axes.flat,[rgb,gt,COLORS[pred],error],['Original RGB','Ground truth (black = void)','Epoch-19 prediction','Errors: red wrong, gray correct, black ignored']):
        ax.imshow(pixels)
        ax.set_title(title,pad=10)
        ax.axis('off')
    fig.suptitle(f"{int(row['example']):02d} | {Path(row['source_image']).stem}\n{row['selection_reason']}; image mIoU={float(row['image_miou']):.3f}",fontsize=11,y=.97)
    fig.legend(handles=[Patch(color=c/255,label=n) for c,n in zip(COLORS,NAMES)],loc='lower center',ncol=10,fontsize=8,bbox_to_anchor=(.5,.02))
    fig.subplots_adjust(left=.02,right=.98,bottom=.10,top=.90,wspace=.04,hspace=.22)
    fig.savefig(OUT/row['file'],dpi=120)
    plt.close(fig)
print('Re-rendered 12 cached examples; no model inference or training.')
