"""Bounded independent follow-up; old16 must finish and be delivered first."""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import ROOT,read_json,write_json,sha
from prepare_9b_repair import OUT,OLD,OLD_EVAL,SEEDS,SOURCE_FILES,freeze_protocol,now

ENV=ROOT.parent/'arp-qwen35-venv/bin/python'
TAG='arp-9b-122b-20261017'


def verify_protocol():
    plan=freeze_protocol()
    assert read_json(OUT/'registration/delivery-proof.json')['status']=='passed'
    return plan


def phase_budget(kind,create=False):
    path=OUT/f'{kind}-wall-budget.json'
    if not path.exists():
        assert create, 'GPU budget must already exist'
        start=time.time();plan=read_json(OUT/'protocol.json')
        seconds=plan['training_wall_seconds_cap' if kind=='training' else 'evaluation_wall_seconds_cap']
        write_json(path,{'start_utc':datetime.fromtimestamp(start,timezone.utc).isoformat(),
            'end_utc':datetime.fromtimestamp(start+seconds,timezone.utc).isoformat(),
            'seconds_cap':seconds,'never_reset_on_restart':True})
    end=datetime.fromisoformat(read_json(path)['end_utc']).timestamp()
    assert time.time()<end,kind+' absolute budget exhausted'
    return end


def admission():
    plan=verify_protocol()
    proof=read_json(OUT/'old-queue-verification.json')
    assert proof['status']=='passed' and proof['complete_sha256']==sha(OLD_EVAL/'complete.json')
    assert read_json(OLD_EVAL/'complete.json')['rows']==270400
    import importlib.metadata
    environment=read_json(OUT/'environment-lock.json')
    for name,version in environment['packages'].items():
        assert importlib.metadata.version(name)==version,(name,'environment changed')
    model=ROOT/'.model_cache/qwen35_9b'
    for row in environment['model_files']:
        stat=(model/row['file']).stat()
        assert stat.st_size==row['bytes'] and stat.st_mtime_ns==row['mtime_ns'],'Weight file changed since fresh SHA verification'
    for name,digest in environment['small_files_sha256'].items():
        assert sha(model/name)==digest,('model/tokenizer metadata changed',name)
    assert shutil.disk_usage(OUT).free>=plan['minimum_free_bytes']
    return plan


def adapter_for(condition,seed):
    if condition=='N':return None
    if condition=='P-D':
        proof=read_json(OUT/'historical-reuse.json')[f'P-D-seed{seed}']
        path=ROOT/proof['source']/'adapter'
        assert sha(path/'adapter_model.safetensors')==proof['adapter_sha256']
        return path
    path=OUT/'training'/f'{condition}-seed{seed}'/'adapter'
    marker=read_json(path.parent/'complete.json')
    assert marker['actual_updates']==1000 and sha(path/'adapter_model.safetensors')==marker['adapter_sha256']
    return path


def deliver(folder,token,kind=None):
    import requests
    from publish_artifacts import publish
    from run_9b_formal_when_ready import verify_receipt
    deadline=phase_budget(kind) if kind else time.time()+1200
    attempt=0
    while True:
        try:
            receipt=publish(folder,TAG,token=token)
            proof=verify_receipt(receipt,token)
            write_json(folder/'delivery-proof.json',{'status':'passed','receipt':receipt,'fresh_server_check':proof})
            # Drop only this verified delivery's redundant export tar; keep source/checkpoints.
            for asset in receipt['assets']:
                path=ROOT/'.exports'/asset['name']
                if path.suffix=='.gz' and path.exists():
                    assert path.resolve().parent==(ROOT/'.exports').resolve()
                    assert sha(path)==asset['sha256'] and path.stat().st_size==asset['bytes']
                    write_json(folder/'export-archive-restore-pointer.json',{'assets':receipt['assets'],
                        'release_url':receipt['release_url'],'local_export_only':True,'server_verification':proof})
                    path.unlink()
            return receipt
        except requests.RequestException as error:
            response=getattr(error,'response',None);status=response.status_code if response is not None else None
            if status is not None and status<500 and status not in (408,429):raise
            attempt+=1
            record={'utc':now(),'operation':'GitHub delivery','folder':str(folder.relative_to(ROOT)),
                    'attempt':attempt,'error_type':type(error).__name__,'http_status':status,
                    'no_generation_or_training_retry':True}
            path=OUT/'network-retries.jsonl'
            with path.open('a') as f:f.write(json.dumps(record)+'\n')
            remaining=deadline-time.time()
            if remaining<=0:raise RuntimeError('Delivery absolute deadline exhausted') from None
            time.sleep(min(60,2**min(attempt,6),remaining))


def old_queue_alive():
    for path in Path('/proc').glob('[0-9]*/cmdline'):
        try:
            argv=path.read_bytes().split(bytes([0]))
            if any(v.endswith(b'run_9b_native_frontend_v41.py') for v in argv):return True
        except (OSError,ProcessLookupError):pass
    return False


def wait_old(token):
    while not (OLD_EVAL/'complete.json').exists():
        if (OLD_EVAL/'stopped.json').exists():raise RuntimeError('Old16 stopped; do not steal priority or restart it')
        assert old_queue_alive(),'Old16 has no live process or completion marker'
        write_json(OUT/'status.json',{'phase':'waiting_for_old16_verified_delivery','timestamp_utc':now(),
                                    'old_queue_live_verified':True,'new_GPU_budget_not_started':True})
        time.sleep(30)
    if (OUT/'old-queue-verification.json').exists():return
    import run_9b_native_frontend_v41 as old
    result=old.verify_formal()
    assert result['models']==16 and result['rows']==270400
    assert read_json(OLD_EVAL/'terminal-proof/delivery-proof.json')['status']=='passed'
    from run_9b_formal_when_ready import verify_receipt
    verify_receipt(read_json(OLD_EVAL/'complete.json')['receipt'],token)
    write_json(OUT/'old-queue-verification.json',{'status':'passed','timestamp_utc':now(),
        'complete_sha256':sha(OLD_EVAL/'complete.json'),'models':16,'rows':270400})


def child(script,arguments,kind):
    command=[str(ENV),'-u',str(ROOT/'tools'/script),*arguments]
    with subprocess.Popen(command,cwd=ROOT) as process:
        while True:
            try:
                code=process.wait(timeout=5)
                assert code==0,('Repair child failed',script,arguments,code)
                return
            except subprocess.TimeoutExpired:
                path=OUT/f'{kind}-wall-budget.json'
                if path.exists() and time.time()>=datetime.fromisoformat(read_json(path)['end_utc']).timestamp():
                    process.terminate()
                    try:process.wait(timeout=10)
                    except subprocess.TimeoutExpired:process.kill();process.wait()
                    raise RuntimeError(kind+' absolute budget exhausted; never reset')


def run(token):
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'manager.lock').open('a') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'complete.json').exists():return
        assert not (OUT/'stopped.json').exists(),'Prior failure retained; engineering recovery needs explicit audit'
        freeze_protocol()
        registration=OUT/'registration';registration.mkdir(exist_ok=True)
        for source in [OUT/'protocol.json',OUT/'data-preparation.json',OUT/'heldout-selection.json',
                OUT/'environment-lock.json',OUT/'heldout-novelty-verification.json',OUT/'historical-reuse.json',OUT/'training-order.json',OUT/'development-manifest.jsonl',OUT/'heldout-manifest.jsonl']:
            shutil.copy2(source,registration/source.name)
        for name in SOURCE_FILES:shutil.copy2(ROOT/'tools'/name,registration/name)
        shutil.copytree(OUT/'training-inputs',registration/'training-inputs',dirs_exist_ok=True)
        if not (registration/'delivery-proof.json').exists():deliver(registration,token)
        wait_old(token)
        from train_9b_repair import verify_training
        for job in read_json(OUT/'training-order.json')['jobs']:
            condition,seed=job['condition'],job['seed'];folder=OUT/'training'/job['id']
            args=['--condition',condition,'--seed',str(seed)]
            if not (folder/'complete.json').exists():
                if condition=='P-G' and seed==42 and not (folder/'interruption-at50.json').exists():
                    child('train_9b_repair.py',args+['--until','50'],'training')
                child('train_9b_repair.py',args,'training')
            verify_training(condition,seed)
            if not (folder/'delivery-proof.json').exists():deliver(folder,token,'training')
        from evaluate_9b_repair import verify_evaluation
        for job in read_json(OUT/'training-order.json')['jobs']:
            folder=OUT/'evaluations/development'/job['id']
            if not (folder/'scores.json').exists():
                child('evaluate_9b_repair.py',['--condition',job['condition'],'--seed',str(job['seed']),'--dataset','development'],'evaluation')
            verify_evaluation(job['condition'],job['seed'],'development')
            if not (folder/'delivery-proof.json').exists():deliver(folder,token,'evaluation')
        from repair_9b_codec import CONDITIONS
        endpoints=[('N',42)]+[(c,s) for s in SEEDS for c in CONDITIONS]
        for condition,seed in endpoints:
            folder=OUT/'evaluations/heldout'/f'{condition}-seed{seed}'
            if not (folder/'scores.json').exists():
                child('evaluate_9b_repair.py',['--condition',condition,'--seed',str(seed),'--dataset','heldout'],'evaluation')
            verify_evaluation(condition,seed,'heldout')
            if not (folder/'delivery-proof.json').exists():deliver(folder,token,'evaluation')
        from report_9b_repair import report
        report()
        deliver(OUT/'report',token,'evaluation')
        receipt=deliver(OUT,token,'evaluation')
        completion={'status':'all_fixed_factorial_endpoints_verified_and_delivered','new_runs':6,
            'new_updates':6000,'timestamp_utc':now(),'protocol_sha256':sha(OUT/'protocol.json'),'receipt':receipt}
        write_json(OUT/'terminal-proof/complete.json',completion)
        deliver(OUT/'terminal-proof',token,'evaluation')
        write_json(OUT/'complete.json',completion)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('action',choices=['start','run']);args=parser.parse_args()
    token=os.environ.get('GH_TOKEN') or sys.stdin.readline().strip();assert token
    if args.action=='start':
        command='cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/run_9b_repair.py run >> runs/9b-u137-factorial-console.log 2>&1'
        subprocess.run(['screen','-dmS','9b-u137-factorial','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Independent factorial queue launched; registration and old16 verification precede GPU work')
    else:
        try:run(token)
        except BlockingIOError:print('Existing repair manager remains active; no duplicate queue')
        except Exception as error:
            write_json(OUT/'stopped.json',{'type':type(error).__name__,'message':str(error),
                'timestamp_utc':now(),'old16_unchanged':True,'no_budget_reset':True})
            failure=OUT/'failure-proof';failure.mkdir(exist_ok=True)
            shutil.copy2(OUT/'stopped.json',failure/'stopped.json')
            try:deliver(failure,token)
            except Exception as delivery_error:write_json(OUT/'failure-delivery-error.json',{'type':type(delivery_error).__name__})
            raise
