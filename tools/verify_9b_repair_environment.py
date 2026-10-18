"""Fresh low-priority CPU weight/environment lock; no model or CUDA context."""
import importlib.metadata
import platform
import sys

from experiment_common import ROOT,read_json,write_json,sha
from prepare_9b_repair import OUT,now


def verify():
    model=ROOT/'.model_cache/qwen35_9b'
    reference=read_json(model/'weights.verified.json')
    records=[]
    for item in reference['files']:
        path=model/item['file']
        assert path.stat().st_size==item['bytes'] and sha(path)==item['sha256']
        records.append({**item,'mtime_ns':path.stat().st_mtime_ns})
        print('Fresh model SHA passed',path.name,flush=True)
    result={'status':'passed','timestamp_utc':now(),'model_revision':reference['revision'],
        'python':platform.python_version(),'executable':sys.executable,
        'packages':{name:importlib.metadata.version(name) for name in ('torch','transformers','peft','safetensors')},
        'model_files':records,
        'small_files_sha256':{p.name:sha(p) for p in model.iterdir() if p.is_file() and p.suffix in ('.json','.jinja','.txt') and not p.name.endswith('.ranges.json')},
        'no_model_loaded':True,'no_GPU_work':True}
    assert result['packages']['transformers']=='5.20.1'
    write_json(OUT/'environment-lock.json',result)
    return result


if __name__=='__main__':verify()
