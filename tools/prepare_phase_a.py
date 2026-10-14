"""Reproducible ARP phase-A preparation. See README.md for the frozen protocol."""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import io
import json
import math
from pathlib import Path
import random
import re
import statistics
import sys
import time
import zipfile

import requests
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
RAW = DATA / "raw"
SEED = 20261014
N_MIN = 800
TEMPLATES = [
    "Using the image, answer this question: {question}",
    "Look at the image and select the answer to the following: {question}",
    "Which option best answers this question about the image? {question}",
    "Consider the image carefully. For the question below, choose the correct option: {question}",
    "Based on the visual evidence, determine the answer to this question: {question}",
]

def log(message):
    print(message, flush=True)

def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

def sha256(path):
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def inspect_sources():
    urls = {
        "mmstar": "https://huggingface.co/api/datasets/Lin-Chen/MMStar",
        "mmbench": "https://huggingface.co/api/datasets/lmms-lab/MMBench",
        "refer_readme": "https://raw.githubusercontent.com/lichengunc/refer/master/README.md",
        "pope_sample": "https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_random.json",
    }
    def one(pair):
        name, url = pair
        try:
            path = RAW / "inspection" / (name + ".txt")
            if path.exists():
                log(f"Using pinned source metadata: {name}")
                return
            r = requests.get(url, timeout=40)
            r.raise_for_status()
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(r.text, encoding="utf-8")
            if name in ("mmstar", "mmbench"):
                value = r.json()
                log(json.dumps({"name": name, "sha": value.get("sha"), "files": value.get("siblings")}, ensure_ascii=True))
            else:
                log(json.dumps({"name": name, "status": r.status_code, "sample": r.text[:9000]}, ensure_ascii=True))
        except Exception as e:
            log(f"{name}: {e}")
    with cf.ThreadPoolExecutor(4) as pool:
        list(pool.map(one, urls.items()))

def probe_sources():
    urls = {
        "ref_archive": "https://web.archive.org/web/20220413011718id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/refcoco.zip",
        "ref_hf": "https://huggingface.co/api/datasets?search=refcoco&limit=15",
        "ref_git": "https://raw.githubusercontent.com/lichengunc/refer/master/data/refcoco/refs(unc).p",
        "refer_api": "https://raw.githubusercontent.com/lichengunc/refer/master/refer.py",
        "coco_https": "https://images.cocodataset.org/val2014/COCO_val2014_000000310196.jpg",
        "coco_s3": "https://coco-dataset.s3.amazonaws.com/val2014/COCO_val2014_000000310196.jpg",
        "coco_s3_path": "https://s3.amazonaws.com/images.cocodataset.org/val2014/COCO_val2014_000000310196.jpg",
        "coco_http": "http://images.cocodataset.org/val2014/COCO_val2014_000000310196.jpg",
        "mmbench_official": "https://opencompass.openxlab.space/utils/VLMEval/MMBench_DEV_EN.tsv",
    }
    def one(pair):
        name, url = pair
        try:
            r = requests.get(url, timeout=25, stream=True)
            log(json.dumps({"name":name,"status":r.status_code,"length":r.headers.get("Content-Length"),"type":r.headers.get("Content-Type")}, ensure_ascii=True))
            if name in ("ref_hf", "refer_api") and r.ok:
                path = RAW / "inspection" / (name + ".txt")
                path.write_text(r.text, encoding="utf-8")
                log(r.text[:20000])
            r.close()
        except Exception as e:
            log(f"{name}: {e}")
    with cf.ThreadPoolExecutor(7) as pool:
        list(pool.map(one, urls.items()))

def inspect_refer_mirrors():
    urls = {name: f"https://huggingface.co/api/datasets/{name}" for name in ("jxu124/refcoco", "jxu124/refcocog", "jxu124/refcocoplus")}
    def one(pair):
        name,url = pair
        path = RAW / "inspection" / (name.replace("/", "_") + ".json")
        if path.exists():
            log(f"Using pinned source metadata: {name}")
            return
        r = requests.get(url, timeout=40)
        r.raise_for_status()
        value = r.json()
        dump(path, value)
        log(json.dumps({"name":name,"sha":value.get("sha"),"files":value.get("siblings")}, ensure_ascii=True))
    with cf.ThreadPoolExecutor(4) as pool:
        list(pool.map(one,urls.items()))

def download(url, path, retries=4):
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and path.stat().st_size:
        return path
    partial = path.with_suffix(path.suffix + ".part")
    for attempt in range(retries):
        try:
            with requests.get(url, timeout=(15, 90), stream=True) as response:
                response.raise_for_status()
                with partial.open("wb") as out:
                    for chunk in response.iter_content(1024 * 1024):
                        out.write(chunk)
                expected_size = response.headers.get("Content-Length")
                if expected_size and not response.headers.get("Content-Encoding"):
                    assert partial.stat().st_size == int(expected_size), f"Incomplete download: {url}"
            partial.replace(path)
            log(f"Downloaded {path.relative_to(ROOT)} ({path.stat().st_size:,} bytes)")
            return path
        except Exception:
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

def fetch_refer_mirror():
    jobs = []
    for name in ("refcoco", "refcocoplus", "refcocog"):
        metadata = json.loads((RAW / "inspection" / f"jxu124_{name}.json").read_text(encoding="utf-8"))
        for entry in metadata["siblings"]:
            filename = entry["rfilename"]
            if filename.startswith(("data/train-", "data/validation-")):
                jobs.append((f"https://huggingface.co/datasets/{metadata['id']}/resolve/{metadata['sha']}/{filename}", RAW / "refer_mirror" / name / Path(filename).name))
    with cf.ThreadPoolExecutor(6) as pool:
        list(pool.map(lambda pair: download(*pair), jobs))
    dump(DATA / "refer_mirror_sources.json", [{"url":url,"path":path.relative_to(ROOT).as_posix(),"sha256":sha256(path)} for url,path in jobs])
    import pyarrow.parquet as pq
    for name in ("refcoco", "refcocoplus", "refcocog"):
        for path in sorted((RAW / "refer_mirror" / name).glob("*.parquet")):
            table = pq.read_table(path)
            row = table.slice(0,1).to_pylist()[0]
            log(json.dumps({"name":path.name,"rows":len(table),"schema":str(table.schema),"sample":{k:str(v)[:1000] for k,v in row.items()}}, ensure_ascii=True))

def fetch_metadata():
    jobs = []
    for mode in ("random", "popular", "adversarial"):
        jobs.append((f"https://raw.githubusercontent.com/RUCAIBox/POPE/main/output/coco/coco_pope_{mode}.json", RAW / "pope" / f"coco_pope_{mode}.json"))
    for name, stamp in (("refcoco", "20220413011718"), ("refcoco+", "20220413011656"), ("refcocog", "20220413012904")):
        jobs.append((f"https://web.archive.org/web/{stamp}id_/https://bvisionweb1.cs.unc.edu/licheng/referit/data/{name}.zip", RAW / "refer" / f"{name}.zip"))
    for name, filename in (("mmstar", "mmstar.parquet"), ("mmbench", "en/dev-00000-of-00001.parquet")):
        metadata = json.loads((RAW / "inspection" / f"{name}.txt").read_text(encoding="utf-8"))
        repo, revision = metadata["id"], metadata["sha"]
        jobs.append((f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{filename}", RAW / name / Path(filename).name))
    jobs.append(("https://raw.githubusercontent.com/lichengunc/refer/master/refer.py", RAW / "refer" / "refer.original.py"))
    with cf.ThreadPoolExecutor(8) as pool:
        futures = {pool.submit(download, url, path):(url,path) for url,path in jobs}
        errors = []
        for future in cf.as_completed(futures):
            url,path = futures[future]
            try:
                future.result()
            except Exception as e:
                errors.append({"url":url,"error":str(e)})
                log(f"FAILED {path}: {e}")
    dump(DATA / "download_sources.json", [{"url":url,"path":path.relative_to(ROOT).as_posix(),"sha256":sha256(path) if path.exists() else None} for url,path in jobs])
    if errors:
        dump(DATA / "download_errors.json", errors)
        raise RuntimeError(f"{len(errors)} downloads failed")
    import pyarrow.parquet as pq
    for name in ("mmstar", "mmbench"):
        path = next((RAW / name).glob("*.parquet"))
        table = pq.read_table(path)
        sample = table.slice(0,1).to_pylist()[0]
        sample = {key:(str(value)[:1500] if key != "image" else str(value)[:100]) for key,value in sample.items()}
        log(json.dumps({"name":name,"rows":len(table),"schema":str(table.schema),"sample":sample}, ensure_ascii=True))
    for path in (RAW / "refer").glob("*.zip"):
        with zipfile.ZipFile(path) as archive:
            log(f"{path.name}: {archive.namelist()}")

def write_jsonl(path, records):
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".tmp")
    with partial.open("w", encoding="utf-8", newline="\n") as out:
        for row in records:
            out.write(json.dumps(row, ensure_ascii=False, separators=(",", ":"), allow_nan=False) + "\n")
    partial.replace(path)

def read_jsonl(path):
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]

def rank(tag, identity):
    return hashlib.sha256(f"{SEED}:{tag}:{identity}".encode()).hexdigest()

def image_bytes(row):
    return row["image"]["bytes"] if isinstance(row["image"], dict) else row["image"]

def save_embedded_image(path, content):
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        path.write_bytes(content)
    with Image.open(path) as img:
        img.verify()
    with Image.open(path) as img:
        return {"width":img.width,"height":img.height}

def build_manifest():
    import pyarrow.parquet as pq
    from refer_api import REFER
    records, stats, remote_images = [], {}, {}
    for mode in ("random", "popular", "adversarial"):
        source = RAW / "pope" / f"coco_pope_{mode}.json"
        rows = read_jsonl(source)
        assert len(rows) >= N_MIN
        for row in rows:
            path = DATA / "images" / "coco" / "val2014" / row["image"]
            remote_images[path.relative_to(ROOT).as_posix()] = f"http://images.cocodataset.org/val2014/{row['image']}"
            records.append({"axis":"pope", "item_id":f"pope:{mode}:{row['question_id']}",
                            "image_path":path.relative_to(ROOT).as_posix(),
                            "question":row["text"] + " Answer with yes or no only.",
                            "answer_key":row["label"].lower(), "dataset":"pope", "subset":mode,
                            "source_item_id":str(row["question_id"]), "split":"val2014",
                            "bootstrap_unit":f"pope:{mode}:{row['question_id']}",
                            "image_group":row["image"], "answer_type":"yes_no"})
        stats[f"pope/{mode}"] = {"available_items":len(rows), "selected_items":len(rows)}
    training_stats = {}
    for dataset, split_by in (("refcoco", "unc"), ("refcoco+", "unc"), ("refcocog", "umd")):
        api = REFER(RAW / "refer_mirror", dataset=dataset, splitBy=split_by)
        train_ids, val_ids = api.getRefIds(split="train"), api.getRefIds(split="val")
        train_ann_ids = {api.Refs[rid]["ann_id"] for rid in train_ids}
        val_ann_ids = {api.Refs[rid]["ann_id"] for rid in val_ids}
        assert not train_ann_ids & val_ann_ids, f"Object leakage in {dataset}"
        areas = [api.Anns[aid]["bbox"][2] * api.Anns[aid]["bbox"][3] for aid in sorted(train_ann_ids)]
        assert areas and all(area > 0 for area in areas)
        training_stats[dataset] = {"split_by":split_by, "source":api.source, "train_refs":len(train_ids),
                                   "train_unique_annotations":len(train_ann_ids),
                                   "median_bbox_area_px2":statistics.median(areas),
                                   "train_val_annotation_overlap":0,
                                   "area_definition":"bbox_width * bbox_height (not segmentation area)",
                                   "train_annotation_ids_sha256":hashlib.sha256(json.dumps(sorted(train_ann_ids)).encode()).hexdigest()}
        # One reference and one sentence per image within each dataset.
        selected, used_images = [], set()
        for rid in sorted(val_ids, key=lambda rid:rank(dataset, rid)):
            ref = api.Refs[rid]
            if ref["image_id"] in used_images:
                continue
            used_images.add(ref["image_id"])
            selected.append(rid)
            if len(selected) == N_MIN:
                break
        assert len(selected) >= N_MIN, f"Not enough unique validation images in {dataset}"
        for rid in selected:
            ref = api.Refs[rid]
            sent = min(ref["sentences"], key=lambda sent:sent["sent_id"])
            img = api.Imgs[ref["image_id"]]
            x,y,w,h = api.getRefBox(rid)
            bbox = [x,y,x+w,y+h]
            item_id = f"grounding:{dataset}:{split_by}:{rid}:{sent['sent_id']}"
            path = DATA / "images" / "coco" / "train2014" / img["file_name"]
            remote_images[path.relative_to(ROOT).as_posix()] = f"http://images.cocodataset.org/train2014/{img['file_name']}"
            records.append({"axis":"grounding", "item_id":item_id,
                            "image_path":path.relative_to(ROOT).as_posix(),
                            "question":f"Locate the object referred to by: {sent['sent']}. Return only its bounding box as [x_min, y_min, x_max, y_max] in original-image pixel coordinates. The image is {img['width']} pixels wide and {img['height']} pixels high.",
                            "answer_key":bbox, "answer_type":"bbox_xyxy_pixels",
                            "dataset":dataset,"subset":split_by,"split":"val", "ref_id":rid,
                            "ann_id":ref["ann_id"],"sent_id":sent["sent_id"],"image_id":ref["image_id"],
                            "width":img["width"],"height":img["height"],"bootstrap_unit":item_id,
                            "image_group":img["file_name"],"iou_threshold":0.5})
        stats[f"grounding/{dataset}"] = {"available_refs":len(val_ids),
                                           "available_sentences":sum(len(api.Refs[rid]["sentences"]) for rid in val_ids),
                                           "available_images":len({api.Refs[rid]["image_id"] for rid in val_ids}),
                                           "selected_items":len(selected),"selected_unique_images":len(used_images)}
    dump(DATA / "baseline_train_statistics.json", training_stats)
    star_rows = pq.read_table(RAW / "mmstar" / "mmstar.parquet").to_pylist()
    star_hashes = set()
    for row in sorted(star_rows, key=lambda row:row["index"]):
        item_id = f"mmstar:{row['index']}"
        content = image_bytes(row)
        star_hashes.add(hashlib.sha256(content).hexdigest())
        path = DATA / "images" / "mmstar" / f"{row['index']}.jpg"
        size = save_embedded_image(path, content)
        question = row["question"]
        matches = re.findall(r"(?:^|[\s,])(?:\(([A-D])\)|([A-D])\s*[:.)])", question)
        choices = sorted({first or second for first,second in matches})
        assert 2 <= len(choices) <= 4 and row["answer"] in choices, (row["index"], choices)
        records.append({"axis":"mmstar","item_id":item_id,"image_path":path.relative_to(ROOT).as_posix(),
                        "question":question + "\nReturn only the option letter.","answer_key":row["answer"],
                        "answer_type":"option_letter", "options":choices,"dataset":"mmstar", "split":"val",
                        "source_item_id":str(row["index"]), "category":row["category"],"l2_category":row["l2_category"],
                        "bootstrap_unit":item_id,"image_group":item_id, **size})
    stats["mmstar"] = {"available_items":len(star_rows),"selected_items":len(star_rows)}
    bench_rows = pq.read_table(RAW / "mmbench" / "dev-00000-of-00001.parquet").to_pylist()
    candidates, seen, excluded_star_overlap = [], set(), 0
    for row in sorted(bench_rows, key=lambda row:int(row["index"])):
        content = image_bytes(row)
        content_hash = hashlib.sha256(content).hexdigest()
        if content_hash in star_hashes:
            excluded_star_overlap += 1
            continue
        options = {letter:row[letter] for letter in "ABCD" if row.get(letter) and row[letter].strip().lower() not in ("nan", "none", "null")}
        if len(options) < 2 or row["answer"] not in options:
            continue
        # Circular option permutations and duplicate entries are one original question.
        identity = (content_hash, row["question"].strip(), tuple(sorted(options.values())))
        if identity in seen:
            continue
        seen.add(identity)
        candidates.append((row,options,content_hash))
    selected = sorted(candidates, key=lambda entry:rank("consistency", entry[0]["index"]))[:N_MIN]
    assert len(selected) >= N_MIN
    base_items, answer_keys = [], []
    for row,options,content_hash in selected:
        base_id = f"consistency:mmbench_en_dev:{row['index']}"
        path = DATA / "images" / "mmbench" / f"{row['index']}.jpg"
        size = save_embedded_image(path, image_bytes(row))
        hint = row.get("hint")
        hint = hint.strip() if hint and hint.strip().lower() not in ("nan", "none", "null") else ""
        context = ("Context: " + hint + "\n") if hint else ""
        choices = "\n".join(f"{letter}. {text}" for letter,text in options.items())
        base_items.append({"base_item_id":base_id,"image_path":path.relative_to(ROOT).as_posix(),
                           "source_item_id":str(row["index"]),"question":row["question"], "hint":hint,
                           "options":options,"answer_key":row["answer"],"category":row.get("category"), **size})
        for number,template in enumerate(TEMPLATES, 1):
            item_id = f"{base_id}:p{number}"
            records.append({"axis":"consistency","item_id":item_id,
                            "image_path":path.relative_to(ROOT).as_posix(),
                            "question":context + template.format(question=row["question"]) + "\n" + choices + "\nReturn only the option letter.",
                            "answer_key":row["answer"],"answer_type":"option_letter", "options":list(options),
                            "dataset":"mmbench", "split":"en/dev", "source_item_id":str(row["index"]),
                            "base_item_id":base_id,"bootstrap_unit":base_id,"paraphrase_id":number,
                            "image_group":content_hash,"category":row.get("category"), **size})
            answer_keys.append({"item_id":item_id,"base_item_id":base_id,"answer_key":row["answer"]})
    stats["consistency"] = {"source_rows":len(bench_rows), "deduplicated_candidates":len(candidates),
                                "excluded_mmstar_image_overlap_rows":excluded_star_overlap,
                                "selected_base_items":len(selected),"paraphrases_per_item":5,
                                "selected_prompt_rows":len(selected)*5}
    records.sort(key=lambda row:(row["axis"],row["item_id"]))
    write_jsonl(DATA / "manifest.pending.jsonl", records)
    write_jsonl(DATA / "consistency" / "base_items.jsonl", base_items)
    write_jsonl(DATA / "consistency" / "answer_keys.jsonl", answer_keys)
    dump(DATA / "consistency" / "templates.json", {"p_para":5,"templates":TEMPLATES,"score":"all_five_correct","resample_unit":"base_item_id"})
    dump(DATA / "selection_report.json", stats)
    dump(DATA / "image_download_plan.json", remote_images)
    log(json.dumps({"prompt_rows":len(records),"remote_images":len(remote_images),"selection":stats},ensure_ascii=False,indent=2))

def audit_refer_mirror():
    import pyarrow.parquet as pq
    from refer_api import load_original
    report = {}
    for dataset,split_by in (("refcoco","unc"),("refcoco+","unc"),("refcocog","umd")):
        refs, instances = load_original(RAW / "refer" / f"{dataset}.zip",dataset,split_by)
        original = {ref["ref_id"]:ref for ref in refs if ref["split"] in ("train","val")}
        anns = {ann["id"]:ann for ann in instances["annotations"]}
        imgs = {img["id"]:img for img in instances["images"]}
        mirror = {}
        for path in sorted((RAW / "refer_mirror" / dataset.replace("+","plus")).glob("*.parquet")):
            for row in pq.read_table(path).to_pylist():
                mirror[row["ref_id"]] = row
        assert set(original) == set(mirror), f"Reference ID mismatch: {dataset}"
        for rid,ref in original.items():
            row = mirror[rid]
            assert all(ref[key] == row[key] for key in ("ann_id","image_id","split","sentences","category_id")), (dataset,rid)
            assert anns[ref["ann_id"]] == json.loads(row["raw_anns"]), (dataset,rid,"annotation")
            assert imgs[ref["image_id"]] == json.loads(row["raw_image_info"]), (dataset,rid,"image metadata")
        report[dataset] = {"split_by":split_by,"matched_train_val_references":len(original),
                           "matched_original_annotations":True,"matched_original_sentences":True,
                           "matched_original_image_metadata":True,"status":"passed"}
    dump(DATA / "refer_mirror_audit.json",report)
    log(json.dumps(report,ensure_ascii=False,indent=2))

def fetch_images():
    plan = json.loads((DATA / "image_download_plan.json").read_text(encoding="utf-8"))
    def one(pair):
        relative,url = pair
        path = ROOT / relative
        if not path.exists():
            path.parent.mkdir(parents=True,exist_ok=True)
            partial = path.with_suffix(".jpg.part")
            for attempt in range(6):
                try:
                    with requests.get(url,timeout=(15,45)) as response:
                        response.raise_for_status()
                        content = response.content
                    with Image.open(io.BytesIO(content)) as img:
                        img.verify()
                    partial.write_bytes(content)
                    partial.replace(path)
                    break
                except Exception:
                    if attempt == 5:
                        raise
                    time.sleep(min(2**attempt,16))
        with Image.open(path) as img:
            img.verify()
        return relative
    failures = []
    with cf.ThreadPoolExecutor(24) as pool:
        futures = {pool.submit(one,pair):pair for pair in sorted(plan.items())}
        for completed,future in enumerate(cf.as_completed(futures),1):
            relative,url = futures[future]
            try:
                future.result()
            except Exception as e:
                failures.append({"path":relative,"url":url,"error":str(e)})
                log(f"Image failure {relative}: {e}")
            if completed % 100 == 0 or completed == len(plan):
                log(f"COCO images checked: {completed}/{len(plan)}, failures: {len(failures)}")
    dump(DATA / "image_download_status.json", {"total":len(plan),"failed":failures})
    if failures:
        raise RuntimeError(f"{len(failures)} images failed; rerun images to resume")

def freeze_manifest():
    records = read_jsonl(DATA / "manifest.pending.jsonl")
    sizes = {}
    for row in records:
        path = ROOT / row["image_path"]
        if path not in sizes:
            with Image.open(path) as img:
                sizes[path] = (img.width,img.height)
                img.verify()
        size = sizes[path]
        if "width" in row:
            assert (row["width"],row["height"]) == size, row["item_id"]
        row.update(width=size[0],height=size[1])
    destination = DATA / "manifest.jsonl"
    if destination.exists():
        previous = read_jsonl(destination)
        if previous != records:
            raise RuntimeError("Frozen manifest differs: use a new experiment directory")
    write_jsonl(destination, records)
    dump(DATA / "manifest.meta.json", {
        "schema_version":"arp.phase_a.v1","prepared_date":"2026-10-14","seed":SEED,"n_min":N_MIN,
        "manifest_sha256":sha256(destination),"prompt_rows":len(records),
        "axis_order":["pope","grounding","consistency","mmstar"],
        "scoring":{"pope":"yes/no accuracy; equal-weight 3 subsets",
                   "grounding":"IoU >= 0.5; equal-weight 3 datasets",
                   "consistency":"one unit per base_item_id; all 5 variants correct",
                   "mmstar":"option-letter accuracy"},
        "path_base":"project root; image_path is portable POSIX relative path",
        "grounding_box_format":"xyxy in original-image pixels",
        "degenerate_answerers":{"pope":"always no","grounding":"image-centered rectangle with image aspect ratio; per-dataset training bbox-area median in pixels; clip area to image area only when necessary",
                                 "consistency":"always option A on all five variants","mmstar":"Python random.Random(20261014); uniform over actual options; sorted manifest order"},
        "consistency_bootstrap":"resample 800 base_item_id groups, preserving all 5 variants",
        "group_correlation_note":"POPE reuses 500 images; grounding datasets share some COCO images. image_group supports image-cluster sensitivity analyses.",
        "source_lock_files":["data/download_sources.json","data/refer_mirror_sources.json"],
        "selection_report":"data/selection_report.json",
    })
    log(f"Manifest frozen: {len(records)} rows, SHA256 {sha256(destination)}")

def run_baselines():
    from scoring import box_iou,center_box,consistency_bit
    records = read_jsonl(DATA / "manifest.jsonl")
    training_stats = json.loads((DATA / "baseline_train_statistics.json").read_text(encoding="utf-8"))
    rng = random.Random(SEED)
    ledger, groups, units = [], {}, []
    for row in records:
        axis = row["axis"]
        if axis == "pope":
            prediction, name = "no", "always_no"
            bit = int(prediction == row["answer_key"])
            details = {}
        elif axis == "grounding":
            area = training_stats[row["dataset"]]["median_bbox_area_px2"]
            prediction, name = center_box(row["width"],row["height"],area), "center_train_median_box"
            iou = box_iou(prediction,row["answer_key"])
            bit, details = int(iou >= 0.5), {"iou":iou,"target_area_px2":area,"effective_area_px2":min(area,row['width']*row['height'])}
        elif axis == "consistency":
            prediction, name = "A", "fixed_option_A"
            bit, details = int(prediction == row["answer_key"]), {}
        elif axis == "mmstar":
            prediction, name = rng.choice(row["options"]), "seeded_uniform_random_option"
            bit, details = int(prediction == row["answer_key"]), {"seed":SEED}
        else:
            raise ValueError(axis)
        result = {"checkpoint":f"degenerate:{name}","axis":axis,"item_id":row["item_id"],
                  "bootstrap_unit":row["bootstrap_unit"],"dataset":row["dataset"],"subset":row.get("subset"),
                  "raw_generation":json.dumps(prediction) if isinstance(prediction,list) else prediction,
                  "parsed_answer":prediction,"answer_key":row["answer_key"],"correct_bit":bit,
                  "parse_fail_flag":False, **details}
        ledger.append(result)
        if axis == "consistency":
            groups.setdefault(row["base_item_id"],[]).append(result)
        else:
            units.append(result)
    for base_id,variants in sorted(groups.items()):
        bit = consistency_bit([row["correct_bit"] for row in variants])
        units.append({"checkpoint":"degenerate:fixed_option_A","axis":"consistency",
                      "item_id":base_id,"bootstrap_unit":base_id,"dataset":"mmbench","subset":None,
                      "correct_bit":bit,"parse_fail_flag":False,"variant_item_ids":[row["item_id"] for row in variants]})
    summary = {}
    for axis in ("pope","grounding","consistency","mmstar"):
        selected = [row for row in units if row["axis"] == axis]
        n,correct = len(selected),sum(row["correct_bit"] for row in selected)
        assert n >= N_MIN
        subgroups = {}
        for row in selected:
            key = row.get("subset") if axis == "pope" else row["dataset"]
            if key:
                subgroups.setdefault(key,[]).append(row["correct_bit"])
        per_subset = {key:{"n":len(bits),"correct":sum(bits),"c_a":sum(bits)/len(bits)} for key,bits in subgroups.items()}
        summary[axis] = {"n_scoring_units":n,"n_prompt_rows":sum(row["axis"]==axis for row in ledger),
                         "correct":correct,"c_a":correct/n,"per_subset":per_subset}
    write_jsonl(DATA / "baselines" / "per_item.jsonl", ledger)
    write_jsonl(DATA / "baselines" / "per_unit.jsonl", sorted(units,key=lambda row:(row["axis"],row["item_id"])))
    dump(DATA / "baselines" / "c_a.json", {"seed":SEED,"manifest_sha256":sha256(DATA / "manifest.jsonl"),
                                            "consistency_fixed_answer":"A","axes":summary})
    log(json.dumps(summary,ensure_ascii=False,indent=2))

def validate():
    from scoring import consistency_bit
    manifest = read_jsonl(DATA / "manifest.jsonl")
    meta = json.loads((DATA / "manifest.meta.json").read_text(encoding="utf-8"))
    assert sha256(DATA / "manifest.jsonl") == meta["manifest_sha256"]
    ids = [row["item_id"] for row in manifest]
    assert len(ids) == len(set(ids))
    required = {"axis","item_id","image_path","question","answer_key","bootstrap_unit"}
    image_rows = {}
    consistency = {}
    axis_units = {}
    for row in manifest:
        assert required <= row.keys(), row["item_id"]
        assert row["question"].strip()
        path = (ROOT / row["image_path"]).resolve()
        assert path.is_relative_to(ROOT.resolve())
        image_rows.setdefault(row["image_path"],[]).append(row)
        axis_units.setdefault(row["axis"],set()).add(row["bootstrap_unit"])
        if row["answer_type"] == "bbox_xyxy_pixels":
            x1,y1,x2,y2 = row["answer_key"]
            assert all(math.isfinite(x) for x in (x1,y1,x2,y2))
            assert 0 <= x1 < x2 <= row["width"]+1 and 0 <= y1 < y2 <= row["height"]+1
        elif row["answer_type"] == "option_letter":
            assert row["answer_key"] in row["options"]
        else:
            assert row["answer_key"] in ("yes","no")
        if row["axis"] == "consistency":
            consistency.setdefault(row["base_item_id"],[]).append(row)
    assets = []
    for relative,rows in sorted(image_rows.items()):
        path = ROOT / relative
        with Image.open(path) as img:
            img.load()
            assert all((row["width"],row["height"]) == img.size for row in rows)
        assets.append({"path":relative,"bytes":path.stat().st_size,"sha256":sha256(path)})
    for rows in consistency.values():
        assert len(rows) == 5 and {row["paraphrase_id"] for row in rows} == set(range(1,6))
        assert len({row["answer_key"] for row in rows}) == 1
        assert len({row["question"] for row in rows}) == 5
        assert len({row["image_path"] for row in rows}) == 1
    assert set(axis_units) == {"pope","grounding","consistency","mmstar"}
    assert all(len(items) >= N_MIN for items in axis_units.values())
    ledger = read_jsonl(DATA / "baselines" / "per_item.jsonl")
    assert {row["item_id"] for row in ledger} == set(ids) and len(ledger) == len(ids)
    assert all(row["correct_bit"] in (0,1) and row["parse_fail_flag"] is False for row in ledger)
    units = read_jsonl(DATA / "baselines" / "per_unit.jsonl")
    floors = json.loads((DATA / "baselines" / "c_a.json").read_text(encoding="utf-8"))
    for axis in axis_units:
        axis_rows = [row for row in units if row["axis"]==axis]
        assert len(axis_rows) == len(axis_units[axis])
        assert sum(row["correct_bit"] for row in axis_rows)/len(axis_rows) == floors["axes"][axis]["c_a"]
    by_item = {row["item_id"]:row for row in ledger}
    by_unit = {row["item_id"]:row for row in units if row["axis"] == "consistency"}
    for base_id,rows in consistency.items():
        expected = consistency_bit([by_item[row["item_id"]]["correct_bit"] for row in rows])
        assert by_unit[base_id]["correct_bit"] == expected
    for path in (DATA / "download_sources.json",DATA / "refer_mirror_sources.json"):
        for source in json.loads(path.read_text(encoding="utf-8")):
            if source.get("sha256"):
                assert sha256(ROOT / source["path"]) == source["sha256"]
    write_jsonl(DATA / "assets.sha256.jsonl",assets)
    report = {"status":"passed","manifest_sha256":sha256(DATA / "manifest.jsonl"),
              "prompt_rows":len(manifest),"scoring_units":len(units),"unique_images":len(assets),
              "image_bytes":sum(row["bytes"] for row in assets),
              "axis_units":{axis:len(items) for axis,items in axis_units.items()},
              "checks":["source hashes","unique item IDs","all images fully decoded","image dimensions",
                        "answer keys and box coordinates","five variants grouped per base question",
                        ">=800 scoring units per axis","complete baseline coverage","floor summary agrees with records"]}
    dump(DATA / "validation_report.json",report)
    import importlib.metadata
    lock_paths = [DATA / "manifest.jsonl",DATA / "manifest.meta.json",DATA / "baseline_train_statistics.json",
                  DATA / "consistency" / "templates.json",DATA / "baselines" / "per_item.jsonl",
                  DATA / "baselines" / "per_unit.jsonl",DATA / "baselines" / "c_a.json",
                  DATA / "assets.sha256.jsonl",DATA / "download_sources.json",DATA / "refer_mirror_sources.json",
                  DATA / "refer_mirror_audit.json",ROOT / "requirements.txt",*sorted((ROOT / "tools").glob("*.py"))]
    dump(DATA / "experiment.lock.json", {
        "schema_version":"arp.phase_a.lock.v1","python":sys.version.split()[0],
        "packages":{package:importlib.metadata.version(package) for package in ("requests","Pillow","pyarrow")},
        "files":{path.relative_to(ROOT).as_posix():sha256(path) for path in lock_paths}})
    log(json.dumps(report,ensure_ascii=False,indent=2))

def prepare_all():
    inspect_sources()
    inspect_refer_mirrors()
    fetch_metadata()
    fetch_refer_mirror()
    audit_refer_mirror()
    build_manifest()
    fetch_images()
    freeze_manifest()
    from make_paraphrases import generate
    write_jsonl(DATA / "consistency" / "items.jsonl", generate(read_jsonl(DATA / "consistency" / "base_items.jsonl")))
    run_baselines()
    validate()

def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    commands = {"inspect":inspect_sources,"probe":probe_sources,"mirrors":inspect_refer_mirrors,
                "fetch":fetch_metadata,"fetch-refer-mirror":fetch_refer_mirror,
                "build":build_manifest,"images":fetch_images,"freeze":freeze_manifest,
                "baselines":run_baselines,"validate":validate,"audit-refer":audit_refer_mirror,"all":prepare_all}
    parser.add_argument("command", choices=list(commands))
    args = parser.parse_args()
    commands[args.command]()

if __name__ == "__main__":
    main()
