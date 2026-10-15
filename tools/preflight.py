"""CPU integration and real GPU smoke checks; synthetic data cannot open D."""
import argparse
import hashlib
from pathlib import Path
import subprocess
import sys
import time
from experiment_common import AXES, MODELS, ROOT, read_json, read_rows, sha, write_json, write_rows

PREFLIGHT=ROOT/'runs/preflight'


def cpu():
    from analyze_rulers import analyze
    from export_results import export
    PREFLIGHT.mkdir(parents=True,exist_ok=True)
    result=subprocess.run([sys.executable,'-m','unittest','discover','-s','tests','-v'],cwd=ROOT,capture_output=True,text=True)
    (PREFLIGHT/'unit-tests.txt').write_text(result.stdout+result.stderr,encoding='utf-8')
    if result.returncode:
        raise RuntimeError('Scoring and guard tests failed')
    manifest=read_rows(ROOT/'data/manifest.jsonl')
    path=PREFLIGHT/'synthetic-integration'
    levels={'qwen3b':[.68,.12,.55,.49],'qwen7b':[.85,.23,.73,.68],
            'intern1b':[.61,.08,.45,.42],'intern2b':[.72,.15,.62,.55],'intern8b':[.86,.25,.77,.72],
            'llava05b':[.55,.05,.35,.33],'llava7b':[.79,.20,.69,.63]}
    for key in MODELS:
        rows=[]
        for item in manifest:
            uniform=int(hashlib.sha256(('fixture:'+item['item_id']).encode()).hexdigest()[:12],16)/16**12
            p=levels[key][AXES.index(item['axis'])]
            if item['axis']=='consistency':
                p=p**.2
            failure=int(hashlib.sha256((key+item['item_id']).encode()).hexdigest()[:6],16)%100==0
            rows.append({'checkpoint':key,'axis':item['axis'],'item_id':item['item_id'],
                         'correct_bit':int(uniform<p and not failure),'parse_fail_flag':int(failure),
                         'raw_generation':'SYNTHETIC TEST FIXTURE, NOT MODEL INFERENCE','parsed_answer':None})
        ledger=path/'ledger'/f'{key}-000.jsonl'
        write_rows(ledger,rows)
        write_json(ledger.with_suffix('.meta.json'),{'smoke':False,'synthetic':True,'manifest_sha256':sha(ROOT/'data/manifest.jsonl')})
    gate=analyze(path,n_boot=32,synthetic_test=True)
    if gate['formal']:
        raise AssertionError('Synthetic fixtures must not produce a formal training gate')
    export(path)
    write_json(PREFLIGHT/'cpu-integration.json',{'status':'passed','unit_tests':17,
                                              'analysis_fixture':'synthetic only','plots_exported':True,'formal_gate_opened':False})
    print('CPU integration passed',flush=True)


def gpu():
    PREFLIGHT.mkdir(parents=True,exist_ok=True)
    pool=ROOT/'data/training/pool.jsonl'
    if not pool.exists():
        raise RuntimeError('Independent training pool is not ready')
    rows=read_rows(pool)
    chosen=[]
    for axis in AXES:
        group=[x for x in rows if x['axis']==axis]
        chosen.extend(group[:2])
    manifest=PREFLIGHT/'training-smoke.jsonl'
    write_rows(manifest,chosen)
    subprocess.run([sys.executable,'tools/train_distill.py','cache','--manifest',str(manifest),
                    '--cache',str(PREFLIGHT/'teacher-cache')],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'tools/train_distill.py','train','--arm','A1','--manifest',str(manifest),
                    '--cache',str(PREFLIGHT/'teacher-cache'),'--output',str(PREFLIGHT/'lora-smoke'),
                    '--steps','4','--smoke'],cwd=ROOT,check=True)
    subprocess.run([sys.executable,'tools/run_inference.py','--checkpoint','qwen3b','--smoke',
                    '--adapter',str(PREFLIGHT/'lora-smoke/adapter'),'--output',str(PREFLIGHT/'adapted-ledger')],cwd=ROOT,check=True)
    for key in MODELS:
        print('GPU smoke:',key,flush=True)
        subprocess.run([sys.executable,'tools/run_inference.py','--checkpoint',key,'--smoke',
                        '--output',str(PREFLIGHT/'ledger')],cwd=ROOT,check=True)
    readiness()


def readiness():
    requirements=[PREFLIGHT/'assets.json',PREFLIGHT/'environment.json',PREFLIGHT/'cpu-integration.json',
                  ROOT/'data/training/pool.lock.json',PREFLIGHT/'lora-smoke/complete.json',
                  PREFLIGHT/'adapted-ledger/qwen3b-000.complete.json']
    for key in MODELS:
        requirements.append(PREFLIGHT/f'ledger/{key}-000.complete.json')
    for path in requirements:
        if not path.exists():
            raise RuntimeError('Missing preflight check: '+str(path))
    for key in MODELS:
        path=PREFLIGHT/f'ledger/{key}-000.jsonl'
        rows=read_rows(path)
        complete=read_json(path.with_suffix('.complete.json'))
        if len(rows)!=8 or complete['ledger_sha256']!=sha(path) or not all(x.get('numerics_checked') for x in rows):
            raise RuntimeError('Real numerical-health/coverage verification missing: '+key)
    if not read_json(PREFLIGHT/'lora-smoke/complete.json')['parameters_changed']:
        raise AssertionError('LoRA optimizer did not update parameters')
    write_json(PREFLIGHT/'ready.json',{'status':'ready','models':list(MODELS),'actual_model_smoke_rows':8*len(MODELS),
               'teacher_cache_items':8,'lora_smoke_steps':4,'protocol_sha256':sha(ROOT/'configs/experiment.json'),
               'manifest_sha256':sha(ROOT/'data/manifest.jsonl'),'checks':{str(p.relative_to(ROOT)):sha(p) for p in requirements},
               'formal_experiment_completed':False})
    print('All end-to-end preflight checks passed: ready for formal B -> C -> conditional D -> E.',flush=True)


if __name__=='__main__':
    p=argparse.ArgumentParser()
    p.add_argument('action',choices=['cpu','gpu','ready'])
    a=p.parse_args()
    {'cpu':cpu,'gpu':gpu,'ready':readiness}[a.action]()
