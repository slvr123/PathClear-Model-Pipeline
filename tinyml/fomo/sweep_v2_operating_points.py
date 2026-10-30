"""Apply one identical operating-point selection rule to all saved INT8 candidates."""
import argparse
import csv
import json
import numpy as np
from analyze_baseline_v1 import ART, EVALUATION, MANIFEST, coarse_sizes, dump, measure
from config import output_directory
from dataset import validated_records


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',required=True)
    args=parser.parse_args();out=output_directory(args.output,'v2-operating-points')
    records,_=validated_records(MANIFEST);records=[r for r in records if r['split']=='validation']
    results=[];selected=[];rows=[]
    for name,evaluation in [('baseline_v1',EVALUATION)]+[(f'weight{w}',ART/f'v2-weight{w}-int8-evaluation-20261004') for w in (10,40)]:
        saved=json.loads((evaluation/'detections.json').read_text())
        assert [r['id'] for r in records]==[r['id'] for r in saved]
        grids=np.load(evaluation/'probabilities.npy',allow_pickle=False)
        candidates=[]
        for percent in range(20,96,5):
            r,_=measure(records,grids,percent/100,decoder='cells')
            r.update(experiment=name,decoder='cells',coarse_size_recall=coarse_sizes(r))
            candidates.append(r);results.append(r)
            rows.append(dict(experiment=name,decoder='cells',threshold=r['threshold'],**r['micro'],
                        person_recall=r['per_class']['person']['recall'],vehicle_recall=r['per_class']['vehicle']['recall'],ge8_recall=r['coarse_size_recall']['ge8']['recall']))
        eligible=[r for r in candidates if r['micro']['precision'] is not None and r['micro']['precision']>=.7]
        best=max(eligible,key=lambda r:(r['micro']['recall'],r['micro']['precision']))
        selected.append(best)
        print(json.dumps(dict(experiment=name,threshold=best['threshold'],**best['micro'])),flush=True)
    dump(out/'sweep.json',results);dump(out/'selected.json',selected)
    dump(out/'rule.json',dict(decoder='cells',thresholds=[i/100 for i in range(20,96,5)],precision_floor=.7,
         selection='Highest recall at or above precision floor; tie-break highest precision',scope='Additional development-validation tuning; fixed protocol comparison retained separately.'))
    with (out/'sweep.csv').open('w',newline='',encoding='utf-8') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)


if __name__=='__main__':main()
