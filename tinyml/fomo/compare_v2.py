"""Compare immutable models under the two predeclared decoding protocols."""
import argparse
import csv
import json
import numpy as np
from analyze_baseline_v1 import ART, EVALUATION, EXPORT, MANIFEST, TARGETS, TRAINING, coarse_sizes, dump, measure, sha
from config import output_directory
from dataset import validated_records


def main():
    parser=argparse.ArgumentParser(description=__doc__); parser.add_argument('--output',required=True)
    args=parser.parse_args(); out=output_directory(args.output,'v2-comparison')
    records,_=validated_records(MANIFEST); records=[r for r in records if r['split']=='validation']
    runs=[('baseline_v1',EVALUATION,EXPORT)]
    runs += [(f'weight{w}',ART/f'v2-weight{w}-int8-evaluation-20261004',ART/f'v2-weight{w}-export-20261004') for w in (10,40)]
    results=[]; rows=[]
    calibration=json.loads((EXPORT/'report.json').read_text())['calibration_ids']
    baseline_metadata=json.loads((TRAINING/'best.json').read_text())
    assert baseline_metadata['manifest_sha256']==sha(TARGETS)
    for weight in (10,40):
        meta=json.loads((ART/f'v2-weight{weight}-training-20261004/best.json').read_text())
        for key in ('manifest_sha256','contract','pretrained_weight_sha256','batch_size','initialization','requested_epochs'):
            assert meta[key]==baseline_metadata[key],key
        for key,value in baseline_metadata['settings'].items():
            assert meta['settings'][key]==(weight if key=='object_weight' else value),key
    for name,evaluation,export in runs:
        saved=json.loads((evaluation/'detections.json').read_text())
        assert [r['id'] for r in records]==[r['id'] for r in saved]
        assert json.loads((export/'report.json').read_text())['calibration_ids']==calibration
        grids=np.load(evaluation/'probabilities.npy',allow_pickle=False)
        assert grids.shape==(500,20,20,3)
        for decoder,threshold in (('cell_nms',.6),('cells',.8)):
            result,_=measure(records,grids,threshold,decoder=decoder)
            result.update(experiment=name,decoder=decoder,model=str(export/'model.int8.tflite'),
                          model_sha256=sha(export/'model.int8.tflite'),int8_size=(export/'model.int8.tflite').stat().st_size)
            result['coarse_size_recall']=coarse_sizes(result)
            if decoder=='cell_nms':
                official=json.loads((evaluation/'report.json').read_text())
                assert result['micro']==official['uncapped_micro'] and result['per_class']==official['per_class']
            results.append(result)
            row=dict(experiment=name,decoder=decoder,threshold=threshold,**result['micro'],
                     person_recall=result['per_class']['person']['recall'],vehicle_recall=result['per_class']['vehicle']['recall'],
                     ge8_recall=result['coarse_size_recall']['ge8']['recall'],int8_size=result['int8_size'])
            rows.append(row)
            print(json.dumps(row),flush=True)
    dump(out/'comparison.json',results)
    with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    header='| Experiment | Decoder / threshold | Precision | Recall | F1 | Person recall | Vehicle recall | >=8 px recall | INT8 bytes |\n|---|---|---:|---:|---:|---:|---:|---:|---:|\n'
    table=header+''.join(f"| {r['experiment']} | {r['decoder']} / {r['threshold']:.2f} | "+' | '.join(f"{r[k]*100:.2f}%" for k in ('precision','recall','f1','person_recall','vehicle_recall','ge8_recall'))+f" | {r['int8_size']:,} |\n" for r in rows)
    (out/'comparison.md').write_text(table,encoding='utf-8')
    dump(out/'protocol.json',dict(primary=dict(decoder='cell_nms',threshold=.6),secondary=dict(decoder='cells',threshold=.8),
         matching_radius_input_pixels=8,validation_images=500,objects=sum(len(r['annotations']) for r in records),
         annotation_exclusions='none',unmatched_unknown_region_predictions='reported separately, excluded from scored FP',
         calibration_ids=calibration,selection_caveat='Single-seed development validation, reused for checkpoint selection and policy tuning; no independent test claim.'))


if __name__=='__main__':main()
