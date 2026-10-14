"""Emit the frozen five-template consistency items and answer keys independently."""
import argparse
from pathlib import Path

from prepare_phase_a import DATA, ROOT, TEMPLATES, dump, read_jsonl, write_jsonl


def generate(base_items):
    for base in base_items:
        base_id = base["base_item_id"]
        context = "Context: " + base["hint"] + "\n" if base.get("hint") else ""
        options = "\n".join(f"{letter}. {text}" for letter,text in base["options"].items())
        for number,template in enumerate(TEMPLATES,1):
            yield {"axis":"consistency","item_id":f"{base_id}:p{number}",
                   "base_item_id":base_id,"bootstrap_unit":base_id,"paraphrase_id":number,
                   "image_path":base["image_path"],
                   "question":context + template.format(question=base["question"]) + "\n" + options + "\nReturn only the option letter.",
                   "answer_key":base["answer_key"],"options":list(base["options"]),
                   "answer_type":"option_letter","dataset":"mmbench","split":"en/dev",
                   "source_item_id":base["source_item_id"],"width":base["width"],"height":base["height"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DATA / "consistency" / "base_items.jsonl")
    parser.add_argument("--output", type=Path, default=DATA / "consistency" / "items.jsonl")
    args = parser.parse_args()
    bases = read_jsonl(args.input)
    records = list(generate(bases))
    write_jsonl(args.output,records)
    # Compare all inference-relevant fields to the already prepared manifest.
    prepared = {row["item_id"]:row for row in read_jsonl(DATA / "manifest.pending.jsonl") if row["axis"]=="consistency"}
    for row in records:
        assert all(prepared[row["item_id"]][key] == value for key,value in row.items())
    print(f"Generated {len(records)} prompts for {len(bases)} base questions; matches manifest.")


if __name__ == "__main__":
    main()
