import os
import re
import json
import glob
import random
from pathlib import Path

SPEC_KEY_ORDER = ["inits", "targets", "connectivity", "floorplan_dim", "blockages"]

def _stable_spec(spec):
    out = {}
    for k in SPEC_KEY_ORDER:
        if k in spec:
            out[k] = spec[k]
    for k in spec.keys():
        if k not in out:
            out[k] = spec[k]
    return out

def _parse_first_json(text: str):
    start = text.find("{")
    if start == -1:
        raise ValueError("No JSON start '{' found")

    for end in range(len(text), start, -1):
        if text[end-1] != "}":
            continue
        chunk = text[start:end]
        try:
            return json.loads(chunk)
        except Exception:
            continue
    raise ValueError("Could not parse JSON object")

def extract_spec_and_switches(text: str):
    spec = _parse_first_json(text)
    spec = _stable_spec(spec)

    m = re.search(r"--\s*Synthesized\s*Network\s*:?\s*--", text)
    if not m:
        raise ValueError("No '-- Synthesized Network --' marker found")

    tail = text[m.end():]
    syn = _parse_first_json(tail)

    if "switches" not in syn:
        if isinstance(syn, dict):
            for v in syn.values():
                if isinstance(v, dict) and "switches" in v:
                    syn = v
                    break

    if "switches" not in syn:
        raise ValueError("Synthesized JSON has no 'switches'")

    return spec, syn["switches"]

def build_samples(raw_dir: str):
    samples = []
    for fp in sorted(glob.glob(os.path.join(raw_dir, "*.txt"))):
        name = Path(fp).stem
        text = Path(fp).read_text(encoding="utf-8", errors="ignore")

        try:
            spec, switches = extract_spec_and_switches(text)
        except Exception as e:
            raise ValueError(f"{fp}: {e}")

        samples.append({
            "id": name,
            "spec": spec,
            "switches": switches,
        })
    return samples

def write_jsonl(path: str, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def main():
    raw_dir = "/content/drive/MyDrive/DSAI/CaseStudy/data/raw/"  
    out_dir = "/content/drive/MyDrive/DSAI/CaseStudy/data/processed/"
    train_path = os.path.join(out_dir, "train.jsonl")
    valid_path = os.path.join(out_dir, "valid.jsonl")

    samples = build_samples(raw_dir)
    print(f"Loaded {len(samples)} samples")

    seed = 42
    rng = random.Random(seed)
    rng.shuffle(samples)

    n = len(samples)
    n_valid = max(1, int(0.1 * n))
    valid = samples[:n_valid]
    train = samples[n_valid:]

    write_jsonl(train_path, train)
    write_jsonl(valid_path, valid)

    split_info = {
        "seed": seed,
        "train_ids": [x["id"] for x in train],
        "valid_ids": [x["id"] for x in valid],
    }
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    with open(os.path.join(out_dir, "split.json"), "w", encoding="utf-8") as f:
        json.dump(split_info, f, ensure_ascii=False, indent=2)

    print(f"Train: {len(train)} / Valid: {len(valid)}")
    print(f"Wrote:\n - {train_path}\n - {valid_path}\n - {os.path.join(out_dir, 'split.json')}")

if __name__ == "__main__":
    main()
