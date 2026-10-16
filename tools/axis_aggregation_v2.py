"""Fixed equal datasets, then equal subsets, within each capability axis."""
from collections import defaultdict
import numpy as np


def stratum_weights(strata):
    datasets=defaultdict(lambda:defaultdict(list))
    for index,(axis,stratum) in enumerate(strata):
        dataset=stratum.split('/',1)[0]
        datasets[axis][dataset].append(index)
    weights=np.zeros(len(strata))
    for groups in datasets.values():
        for indices in groups.values():
            weights[indices]=1/len(groups)/len(indices)
    return weights


def summarize_units(records):
    strata=sorted({(r['axis'],r['stratum']) for r in records})
    weights=stratum_weights(strata)
    result={}
    for axis in sorted({r['axis'] for r in records}):
        result[axis]=sum(weights[index]*np.mean([r['correct_bit'] for r in records if (r['axis'],r['stratum'])==key])
                         for index,key in enumerate(strata) if key[0]==axis)
    return {key:float(value) for key,value in result.items()}
