"""Restore only the COCO images in the private localization review queue."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
from pathlib import Path
from urllib.request import urlopen

from PIL import Image

from experiment_common import ROOT, read_rows, write_json


def fetch(queue_path):
    run = queue_path.parent.parent
    queue = read_rows(queue_path)
    output = run / 'review' / 'images'
    output.mkdir(parents=True, exist_ok=True)

    def restore(entry):
        key, asset, row = entry['review_id'], entry['asset'], entry['source']
        dest = output / f'{key}.jpg'
        source = ROOT / asset['path']
        if dest.is_file():
            raw = dest.read_bytes()
        elif source.is_file():
            raw = source.read_bytes()
        else:
            # Public COCO mirror. The pinned archive byte digest is required
            # before any downloaded content is decoded or written.
            url = f"http://images.cocodataset.org/train2014/{source.name}"
            with urlopen(url, timeout=18) as response:
                raw = response.read(asset['bytes'] + 1)
        if len(raw) != asset['bytes'] or hashlib.sha256(raw).hexdigest() != asset['sha256']:
            raise ValueError(f'{key}: source file hash or size mismatch')
        with Image.open(io.BytesIO(raw)) as image:
            rgb = image.convert('RGB')
            pixels = hashlib.sha256(str(rgb.size).encode() + rgb.tobytes()).hexdigest()
            if rgb.size != (row['width'], row['height']) or [pixels] != row['image_pixel_sha256']:
                raise ValueError(f'{key}: decoded pixel hash or dimensions mismatch')
        if not dest.is_file():
            dest.write_bytes(raw)
        return {'review_id': key, 'sha256': asset['sha256'], 'pixel_sha256': pixels}

    passed, errors = [], []
    with ThreadPoolExecutor(max_workers=8) as executor:
        tasks = [(row['review_id'], executor.submit(restore, row)) for row in queue]
        for key, task in tasks:
            try:
                passed.append(task.result())
            except (OSError, ValueError) as exc:
                errors.append({'review_id': key, 'error': str(exc)})
    write_json(run / 'private' / 'review-image-verification.json',
               {'verified': passed, 'errors': errors})
    print(f'{len(passed)} verified, {len(errors)} failed')
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('queue', type=Path)
    fetch(parser.parse_args().queue)
