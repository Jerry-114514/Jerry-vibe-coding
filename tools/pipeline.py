"""Formal B -> C -> conditional D -> E, with restartable remote backups."""
import argparse
from pathlib import Path
import subprocess
import sys
from experiment_common import MODELS, ROOT, read_json, read_rows, sha, write_json


def pipeline(run, publish=False):
    run=Path(run).resolve()
    if not run.is_relative_to((ROOT/'runs').resolve()):
        raise ValueError('Run must reside under runs/')
    # Re-audit all current evidence with the current verifier before launching.
    # A long-running preflight parent may have imported an older verifier.
    from preflight import readiness
    readiness()
    ready=read_json(ROOT/'runs/preflight/ready.json')
    if ready['status']!='ready':
        raise RuntimeError('End-to-end preflight has not passed')
    if ready['protocol_sha256']!=sha(ROOT/'configs/experiment.json'):
        raise RuntimeError('Protocol changed after preflight')
    if ready['manifest_sha256']!=sha(ROOT/'data/manifest.jsonl'):
        raise RuntimeError('Evaluation manifest changed after preflight')
    for relative,digest in ready['checks'].items():
        if sha(ROOT/relative)!=digest:
            raise RuntimeError('Preflight evidence changed: '+relative)
    run.mkdir(parents=True,exist_ok=True)
    def command(args,stage):
        write_json(run/'status.json',{'stage':stage,'status':'running','command':args})
        subprocess.run([sys.executable,*args],cwd=ROOT,check=True)
        write_json(run/'status.json',{'stage':stage,'status':'complete'})
    def backup(stage,teacher=False):
        if publish:
            from publish_artifacts import publish as save
            save(run,'arp-'+run.name,include_training=stage=='D2',include_teacher=teacher)
    try:
        for key in MODELS:
            command(['tools/run_inference.py','--checkpoint',key,'--output',str(run/'ledger')],'B3-'+key)
            backup('B3-'+key)
        command(['tools/analyze_rulers.py','--run',str(run)],'C')
        command(['tools/analyze_rulers.py','--run',str(run),'--cluster'],'C-image-cluster-sensitivity')
        command(['tools/export_results.py','--run',str(run)],'E-analysis')
        backup('C')
        gate=read_json(run/'analysis/gate.json')
        if gate['decision']!='go':
            write_json(run/'status.json',{'stage':'C3','status':'stopped-at-gate','decision':gate['decision'],
                                        'reason':gate['reason'],'training_started':False})
            backup('gate-stop')
            return
        command(['tools/prepare_training.py','arms','--run',str(run)],'D1')
        pool=ROOT/'data/training/pool.jsonl'
        command(['tools/train_distill.py','cache','--manifest',str(pool),'--cache',str(run/'teacher-cache')],'D2')
        backup('D2',teacher=True)
        for arm in ['A1','A2','U','N']:
            command(['tools/train_distill.py','train','--arm',arm,'--manifest',str(run/'training'/f'{arm}.jsonl'),
                     '--cache',str(run/'teacher-cache'),'--output',str(run/'arms'/arm),'--run',str(run)],'D3-'+arm)
            if arm!='N':
                command(['tools/run_inference.py','--checkpoint','qwen3b','--adapter',str(run/'arms'/arm/'adapter'),
                         '--output',str(run/'post'/arm)],'D4-'+arm)
            backup('D4-'+arm)
        command(['tools/export_results.py','--run',str(run)],'E-final')
        backup('E')
        write_json(run/'status.json',{'stage':'E','status':'complete','training_started':True})
    except Exception as error:
        last=read_json(run/'status.json') if (run/'status.json').exists() else {}
        write_json(run/'status.json',{**last,'status':'failed','error_type':type(error).__name__,
                                     'message':str(error)[:500]})
        if publish:
            try:
                backup('failure')
            except Exception:
                pass
        raise


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('--run',default=str(ROOT/'runs/formal-20261014'))
    p.add_argument('--publish',action='store_true')
    a=p.parse_args()
    pipeline(a.run,a.publish)
