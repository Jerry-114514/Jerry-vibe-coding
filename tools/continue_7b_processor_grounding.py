"""Unblock independent7B audit after9B calibration failure; numerics unchanged."""
from datetime import datetime,timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from experiment_common import ROOT,read_json,write_json,sha
from run_7b_processor_grounding import OUT,MODELS,ENV,prepare,publish_checked,analyze
from route_b_9b_loop import OUT as NINE
from finalize_9b_delivery import script_live


def run(token):
    import fcntl
    prepare()
    with (OUT/'manager.lock').open('w') as handle:
        fcntl.flock(handle,fcntl.LOCK_EX|fcntl.LOCK_NB)
        done=read_json(NINE/'formal-training-complete.json')
        assert done['runs']==15 and done['updates']==15000
        assert not script_live('continue_9b_training_only.py') and not script_live('route_b_9b_loop.py')
        assert (NINE/'processor-pixel-evaluation-v3/stopped.json').exists()
        calibration=read_json(NINE/'processor-pixel-evaluation-v3/minimal/N-seed42/scores.json')
        assert calibration['axis_scores']['grounding']<.8
        amendment=OUT/'independent-queue-amendment.json'
        if not amendment.exists():
            assert not list((OUT/'models').glob('*/eval.jsonl'))
            write_json(amendment,{'status':'user_authorized_independent_measurement_continues',
                'timestamp':datetime.now(timezone.utc).isoformat(),'original_protocol_sha256':sha(OUT/'protocol.json'),
                'controller_source_sha256':sha(Path(__file__)),'reason':'9B failed common-contract calibration;7B has its own fixed80% gate and must not wait for9B',
                'supersedes_queue_dependency_only':True,'registered_generations_unchanged':38400,
                'sampling_scoring_budget_and_stop_rules_unchanged':True,'new9b_generations':0,'new7b_predictions_before_amendment':0})
        registration=OUT/('independent-queue-registration-'+sha(OUT/'protocol.json')[:12]);registration.mkdir(exist_ok=True)
        for source in (amendment,OUT/'protocol.json',Path(__file__)):shutil.copyfile(source,registration/source.name)
        if not (registration/'delivery-proof.json').exists():publish_checked(registration,token)
        budget=OUT/'gpu-wall-budget.json'
        if not budget.exists():
            start=time.time();write_json(budget,{'start_utc':datetime.fromtimestamp(start,timezone.utc).isoformat(),
                'end_utc':datetime.fromtimestamp(start+12*3600,timezone.utc).isoformat(),'never_reset_on_restart':True,
                'queue_amendment_sha256':sha(amendment)})
        def child(arm,seed,limit=None):
            remaining=datetime.fromisoformat(read_json(budget)['end_utc']).timestamp()-time.time();assert remaining>0
            folder=OUT/'models'/f'{arm}-seed{seed}';path=folder/'eval.jsonl'
            if path.exists():
                raw=path.read_bytes()
                if raw and not raw.endswith(b'\n'):
                    boundary=raw.rfind(b'\n')+1
                    path.with_name('eval.incomplete-tail-'+str(time.time_ns())).write_bytes(raw[boundary:])
                    with path.open('r+b') as stream:stream.truncate(boundary)
            command=[str(ENV),'-u',str(ROOT/'tools/run_7b_processor_grounding.py'),'eval','--arm',arm,'--seed',str(seed)]
            if limit is not None:command+=['--limit',str(limit)]
            write_json(OUT/'queue-status.json',{'phase':'independent7b_calibration' if limit else 'independent7b_full_grounding',
                'arm':arm,'seed':seed,'pid':os.getpid(),'timestamp':datetime.now(timezone.utc).isoformat()})
            subprocess.run(command,cwd=ROOT,check=True,timeout=remaining)
        calibration=OUT/'models/N-seed42/calibration.json'
        if not calibration.exists():child('N',42,100)
        proof=read_json(calibration)
        assert proof['accuracy']>=.80 and proof['parse_failures']<=5,'7B common-contract calibration failed; no prompt tuning or threshold changes'
        # Preserve the original100-row calibration ledger before append-only
        # continuation, making the initial checkpoint directly reproducible.
        first=OUT/'calibration-proof';first.mkdir(exist_ok=True)
        source=OUT/'models/N-seed42/eval.jsonl'
        prefix=b''.join(source.read_bytes().splitlines(keepends=True)[:100])
        (first/'eval-first100.jsonl').write_bytes(prefix)
        assert sha(first/'eval-first100.jsonl')==proof['ledger_sha256']
        shutil.copyfile(calibration,first/'calibration.json')
        publish_checked(first,token)
        for arm,seed in MODELS:
            folder=OUT/'models'/f'{arm}-seed{seed}'
            if not (folder/'scores.json').exists():child(arm,seed)
            if not (folder/'delivery-proof.json').exists():publish_checked(folder,token)
        analyze();receipt=publish_checked(OUT,token)
        complete={'status':'all16_processor_pixel_grounding_verified_and_delivered','protocol_sha256':sha(OUT/'protocol.json'),
            'queue_amendment_sha256':sha(amendment),'models':16,'rows_per_model':2400,
            'timestamp':datetime.now(timezone.utc).isoformat(),'receipt':receipt}
        terminal=OUT/'terminal-proof';terminal.mkdir(exist_ok=True);write_json(terminal/'complete.json',complete)
        publish_checked(terminal,token);write_json(OUT/'complete.json',complete)


if __name__=='__main__':
    token=os.environ.get('GH_TOKEN') or sys.stdin.readline().strip();assert token
    if '--start' in sys.argv:
        command='cd /root/autodl-tmp/arp; exec /root/autodl-tmp/arp-venv/bin/python -u tools/continue_7b_processor_grounding.py >> runs/7b-processor-pixel-independent-console.log 2>&1'
        subprocess.run(['screen','-dmS','7b-independent-processor-audit','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Independent7B calibration queued; frozen numerical worker and38400-generation budget unchanged')
    else:
        try:run(token)
        except Exception as error:write_json(OUT/'stopped.json',{'type':type(error).__name__,'message':str(error),'timestamp':datetime.now(timezone.utc).isoformat()});raise
