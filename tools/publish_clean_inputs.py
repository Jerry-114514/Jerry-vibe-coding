"""Archive exact frozen input images from AutoDL directly to GitHub Release."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import subprocess
import sys

from experiment_common import ROOT, read_json, read_rows, sha, write_json

SOURCE = ROOT/'runs/protocol-v2-data/clean-role-manifests-v4'
OUT = ROOT/'runs/protocol-v2-data/frozen-inputs-v4'


def worker():
    import fcntl
    OUT.mkdir(parents=True,exist_ok=True)
    with (OUT/'worker.lock').open('w') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if (OUT/'github-storage.json').exists():
            print('Frozen-input backup already has verified upload receipt',flush=True)
            return
        state={'status':'running','worker_script':Path(__file__).name,'pid':os.getpid(),
               'started_utc':datetime.now(timezone.utc).isoformat(),'stage':'verify_and_link_input_images'}
        write_json(OUT/'status.json',state)
        try:
            complete=read_json(SOURCE/'complete.json')
            assets=read_rows(SOURCE/'selected-image-assets.jsonl')
            if sha(SOURCE/'selected-image-assets.jsonl')!=complete['selected_image_assets_sha256']:
                raise ValueError('Frozen selected image inventory changed')
            required={path for role in ('calibration','audit','confirm') for row in read_rows(SOURCE/(role+'.jsonl')) for path in row['image_paths']}
            if required != {row['path'] for row in assets}:
                raise ValueError('Image inventory does not exactly cover frozen input paths')
            total=sum(row['bytes'] for row in assets)
            if shutil.disk_usage(ROOT).free < 2*total+4*1024**3:
                raise RuntimeError('Insufficient reserve for archive and possible GitHub split assets')
            linked=0
            for row in assets:
                source=(ROOT/row['path']).resolve()
                if not source.is_relative_to((ROOT/'data').resolve()) or source.is_symlink():
                    raise ValueError('Frozen image outside project data')
                if source.stat().st_size!=row['bytes'] or sha(source)!=row['source_sha256']:
                    raise ValueError('Frozen image bytes differ from selected inventory')
                target=OUT/'input-tree'/row['path']
                target.parent.mkdir(parents=True,exist_ok=True)
                if target.exists():
                    if sha(target)!=row['source_sha256']: raise ValueError('Existing export image differs')
                else: os.link(source,target)
                linked+=1
                if linked%200==0:
                    state.update(completed_images=linked,expected_images=len(assets),updated_utc=datetime.now(timezone.utc).isoformat())
                    write_json(OUT/'status.json',state)
            target=OUT/'manifests';target.mkdir(exist_ok=True)
            for path in SOURCE.glob('*.json*'):
                if path.name.startswith('github-'): continue
                shutil.copy2(path,target/path.name)
            (OUT/'README.md').write_text(
                '# Frozen clean inputs v4\n\n'
                'Copy the contents of input-tree/ into the repository root to restore the exact image paths.\n'
                'manifests/ contains calibration, audit, confirmation and training manifests and source fingerprints.\n'
                'Historical training image/teacher-cache/adapter assets remain in the arp-formal-20261014 Release.\n'
                'Raw public source Parquet files and base-model weights are reproducible from pinned source locks.\n'
                'This input backup is not proof of passed human review or permission to train.\n',encoding='utf-8')
            write_json(OUT/'verification.json',{'status':'passed','input_images':len(assets),'input_bytes':total,
                       'exact_frozen_input_coverage':True,'all_source_file_sizes_and_sha256_verified':True,
                       'selected_image_inventory_sha256':sha(SOURCE/'selected-image-assets.jsonl')})
            state.update(stage='uploading',completed_images=len(assets),expected_images=len(assets),
                         input_bytes=total,updated_utc=datetime.now(timezone.utc).isoformat())
            write_json(OUT/'status.json',state)
            from publish_artifacts import publish
            publish(OUT,'arp-protocol-v2-20261016',token=os.environ['GH_TOKEN'])
            state.update(status='complete',stage='upload_verified',completed_utc=datetime.now(timezone.utc).isoformat())
            write_json(OUT/'complete.json',state)
        except Exception as error:
            state.update(status='failed',error_type=type(error).__name__)
            raise
        finally: write_json(OUT/'status.json',state)


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('--worker',action='store_true');args=parser.parse_args()
    if args.worker: worker()
    else:
        token=sys.stdin.readline().strip()
        if not token: raise SystemExit('GitHub credential required')
        if '.arp-v4-input-backup' in subprocess.run(['screen','-ls'],capture_output=True,text=True).stdout:
            raise SystemExit('Input backup already running')
        OUT.mkdir(parents=True,exist_ok=True)
        command=('source /etc/network_turbo >/dev/null 2>&1; cd /root/autodl-tmp/arp; '
                 'exec /root/autodl-tmp/arp-venv/bin/python -u tools/publish_clean_inputs.py --worker >> '+str(OUT/'console.log')+' 2>&1')
        subprocess.run(['screen','-dmS','arp-v4-input-backup','bash','-c',command],env={**os.environ,'GH_TOKEN':token},check=True)
        print('Started exact frozen-image backup directly from AutoDL to GitHub',flush=True)
