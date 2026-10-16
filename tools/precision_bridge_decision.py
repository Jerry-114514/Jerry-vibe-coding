"""Strict per-model/per-axis 0.5 percentage-point bridge decision."""
from decimal import Decimal
import math

AXES=('pope','grounding','consistency','mmstar')
MODELS=('qwen3b','qwen7b')


def assess(local,api,threshold=.005):
    if set(local)!=set(MODELS) or set(api)!=set(MODELS):
        raise ValueError('Both exact checkpoints are required')
    if not math.isfinite(threshold) or not 0<threshold<1:
        raise ValueError('Invalid absolute-score threshold')
    rows=[]
    for model in MODELS:
        if set(local[model])!=set(AXES) or set(api[model])!=set(AXES):
            raise ValueError('All four axes are required for each checkpoint')
        for axis in AXES:
            left,right=local[model][axis],api[model][axis]
            if any(isinstance(x,bool) or not math.isfinite(x) or not 0<=x<=1 for x in (left,right)):
                raise ValueError('Scores must be finite proportions')
            difference=Decimal(str(right))-Decimal(str(left))
            rows.append({'model':model,'axis':axis,'local_score':left,'api_score':right,
                         'signed_difference':float(difference),'absolute_difference':float(abs(difference)),
                         'below_threshold':abs(difference)<Decimal(str(threshold))})
    return {'comparisons':rows,'score_threshold_passed':all(r['below_threshold'] for r in rows),
            'threshold':threshold,'additional_required_checks':['Exact same checkpoint identity','Matched full item coverage and scoring','Paired source-image cluster intervals reported']}
