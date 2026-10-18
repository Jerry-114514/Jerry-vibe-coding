"""Archive only frozen selected image bytes directly on AutoDL, never locally."""
import os
from pathlib import Path
import shutil

from experiment_common import ROOT,read_json,write_json,sha
from prepare_9b_repair import OUT,now


def bundle():
    prepare=read_json(OUT/'data-preparation.json')
    selected=read_json(OUT/'heldout-selection.json')['selected_images']
    expected=dict(prepare['source_image_hashes'])
    for path,meta in selected.items():
        assert path not in expected,'Heldout image overlaps training image path'
        expected[path]=meta['source_sha256']
    folder=OUT/'input-image-bundle';folder.mkdir(exist_ok=True)
    records=[]
    for relative,digest in sorted(expected.items()):
        source=ROOT/relative;destination=folder/'files'/relative
        assert source.resolve().is_relative_to((ROOT/'data').resolve())
        assert sha(source)==digest
        destination.parent.mkdir(parents=True,exist_ok=True)
        assert destination.parent.resolve().is_relative_to(folder.resolve())
        if destination.exists():assert not destination.is_symlink() and sha(destination)==digest
        else:
            try:os.link(source,destination)
            except OSError:shutil.copy2(source,destination)
        records.append({'restore_to':relative,'archive_member':str(destination.relative_to(ROOT)),
                        'sha256':digest,'bytes':source.stat().st_size})
    write_json(folder/'restore-index.json',{'timestamp_utc':now(),'protocol_sha256':sha(OUT/'protocol.json'),
        'source_sha256':sha(Path(__file__)),'images':len(records),'bytes':sum(r['bytes'] for r in records),
        'files':records,'storage':'AutoDL directly to GitHub Release; no local image downloads',
        'restore':'Extract listed archive members on AutoDL, place at restore_to, verify each SHA256'})
    print('Frozen image bundle ready',len(records),'images',sum(r['bytes'] for r in records),'bytes',flush=True)
    return folder


if __name__=='__main__':bundle()
