"""Remote preflight: verify frozen A assets, network, package and GPU health."""
import argparse
import importlib.metadata as metadata
import platform
import shutil
import subprocess
import sys
from experiment_common import ROOT, read_json, read_rows, sha, write_json


def validate_assets():
    from PIL import Image
    lock = read_json(ROOT / 'data/experiment.lock.json')
    for relative, expected in lock['files'].items():
        if sha(ROOT / relative) != expected:
            raise ValueError('Frozen input changed: ' + relative)
    assets = read_rows(ROOT / 'data/assets.sha256.jsonl')
    for item in assets:
        path = ROOT / item['path']
        if sha(path) != item['sha256']:
            raise ValueError('Image digest differs: ' + item['path'])
        with Image.open(path) as image:
            image.load()
    return {'status': 'passed', 'images': len(assets), 'manifest_sha256': sha(ROOT / 'data/manifest.jsonl'),
            'prompt_rows': len(read_rows(ROOT / 'data/manifest.jsonl'))}


def network():
    import requests
    output = {}
    for base in ['https://huggingface.co', 'https://hf-mirror.com', 'https://api.github.com']:
        url = base + ('/api/models/Qwen/Qwen2.5-VL-3B-Instruct' if 'github' not in base else '/repos/Jerry-114514/Jerry-vibe-coding')
        try:
            response = requests.get(url, timeout=20)
            output[base] = {'status': response.status_code}
        except requests.RequestException as e:
            output[base] = {'error': type(e).__name__}
    write_json(ROOT / 'runs/preflight/network.json', output)
    print(output, flush=True)


def environment():
    import torch
    import torchvision
    import transformers
    import peft
    import timm
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is unavailable')
    tensor = torch.randn(128, 128, device='cuda', dtype=torch.float16)
    (tensor @ tensor).sum().item()
    result = {'python': platform.python_version(), 'gpu': torch.cuda.get_device_name(0),
              'vram_bytes': torch.cuda.get_device_properties(0).total_memory,
              'cuda': torch.version.cuda, 'torch': torch.__version__, 'torchvision': torchvision.__version__,
              'transformers': transformers.__version__, 'peft': peft.__version__,
              'free_disk_bytes': shutil.disk_usage(ROOT).free,
              'pip_check': subprocess.run([sys.executable, '-m', 'pip', 'check'], capture_output=True, text=True).stdout}
    write_json(ROOT / 'runs/preflight/environment.json', result)
    freeze = subprocess.check_output([sys.executable, '-m', 'pip', 'freeze'], text=True)
    (ROOT / 'runs/preflight/pip-freeze.txt').write_text(freeze, encoding='utf-8')
    print(result, flush=True)


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.add_argument('action', choices=['assets', 'network', 'environment', 'freeze-models'])
    a = p.parse_args()
    if a.action == 'assets':
        report = validate_assets()
        write_json(ROOT / 'runs/preflight/assets.json', report)
        print(report, flush=True)
    elif a.action == 'network':
        network()
    elif a.action == 'environment':
        environment()
    else:
        from model_backends import freeze_models
        print(freeze_models(), flush=True)
