"""Retry transient transport only, never training, predictions or failed checks."""
from datetime import datetime, timezone
import json
import time

import requests

from experiment_common import read_json
from route_b_9b_loop import OUT


def network_retry(operation, label):
    fallback_end=time.time()+12*3600
    attempt=0
    while True:
        try:return operation()
        except requests.RequestException as error:
            response=getattr(error,'response',None)
            status=response.status_code if response is not None else None
            if status is not None and status not in (408,429) and status<500:raise
            attempt+=1
            budget=OUT/'formal-evaluation-wall-budget.json'
            end=datetime.fromisoformat(read_json(budget)['end_utc']).timestamp() if budget.exists() else fallback_end
            record={'timestamp':datetime.now(timezone.utc).isoformat(),'operation':label,
                    'attempt':attempt,'type':type(error).__name__,'http_status':status,
                    'no_training_or_generation_retry':True,'registered_budget_not_reset':True}
            folder=OUT/'network-retry';folder.mkdir(exist_ok=True)
            with (folder/'events.jsonl').open('a',encoding='utf-8') as stream:stream.write(json.dumps(record)+'\n')
            remaining=end-time.time()
            if remaining<=0:raise RuntimeError('Network retry reached existing absolute time budget') from None
            time.sleep(min(60,2**min(attempt,6),remaining))
