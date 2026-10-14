"""Python 3 subset of REFER's read-only box API, backed by the original archives.

The official Python 2 API is vendored in data/raw/refer/refer.original.py.
Masks and visualization are deliberately outside this CPU box preparation API.
"""
import json
import io
import pickle
from pathlib import Path
import zipfile
import pyarrow.parquet as pq


class _DataOnlyUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        raise pickle.UnpicklingError(f"Unexpected executable pickle global: {module}.{name}")


def load_original(archive_path, dataset, split_by):
    with zipfile.ZipFile(archive_path) as archive:
        instances = json.loads(archive.read(f"{dataset}/instances.json"))
        refs = _DataOnlyUnpickler(io.BytesIO(archive.read(f"{dataset}/refs({split_by}).p")), encoding="latin1").load()
    return refs, instances


class REFER:
    def __init__(self, data_root, dataset="refcoco", splitBy="unc"):
        expected = "umd" if dataset == "refcocog" else "unc"
        if splitBy != expected:
            raise ValueError(f"Frozen experiment only supports {dataset}/{expected}")
        root = Path(data_root)
        archive_path = root.parent / "refer" / f"{dataset}.zip"
        directory = root / dataset.replace("+", "plus")
        self.Refs, self.Anns, self.Imgs = {}, {}, {}
        if archive_path.exists():
            refs, instances = load_original(archive_path, dataset, splitBy)
            self.Refs = {ref["ref_id"]:ref for ref in refs}
            self.Anns = {ann["id"]:ann for ann in instances["annotations"]}
            self.Imgs = {img["id"]:img for img in instances["images"]}
            self.source = "original_REFER_archive"
        else:
            for path in sorted(directory.glob("*.parquet")):
                for row in pq.read_table(path).to_pylist():
                    ref = {key:row[key] for key in ("ref_id", "ann_id", "image_id", "category_id", "sent_ids", "sentences", "split", "file_name")}
                    self.Refs[ref["ref_id"]] = ref
                    self.Anns[ref["ann_id"]] = json.loads(row["raw_anns"])
                    self.Imgs[ref["image_id"]] = json.loads(row["raw_image_info"])
            self.source = "HF_preserved_annotations_mirror"
        if not self.Refs:
            raise ValueError(f"No records in {directory}")
        self.refToAnn = {rid:self.Anns[ref["ann_id"]] for rid,ref in self.Refs.items()}

    def getRefIds(self, image_ids=None, cat_ids=None, ref_ids=None, split=""):
        def ids(value):
            return set(value if isinstance(value, list) else [value]) if value is not None else set()
        images, cats, refs = ids(image_ids), ids(cat_ids), ids(ref_ids)
        return [rid for rid,ref in self.Refs.items()
                if (not images or ref["image_id"] in images)
                and (not cats or ref["category_id"] in cats)
                and (not refs or rid in refs)
                and (not split or ref["split"] == split)]

    def loadRefs(self, ref_ids):
        return [self.Refs[rid] for rid in (ref_ids if isinstance(ref_ids, list) else [ref_ids])]

    def getRefBox(self, ref_id):
        return self.refToAnn[ref_id]["bbox"]
