# fewshot_validate_stage2_text.py
import os
import json
import yaml
import torch
from typing import Any, Dict, Optional, List, Tuple

from tqdm.notebook import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel

from fewshot_noc_text import build_fewshot_stage2_text_prompt


# CONFIG 
TEST_FP = "/kaggle/input/datasets/haehyunlee/noc-singleshot/kaggle/working/step2_full/test.jsonl"
CFG_PATH = "/kaggle/working/CaseStudy/configs/llama7b.yaml"
OUT_DIR = "/kaggle/working/CaseStudy/outputs/fewshot_stage2_text/"
CKPT_DIR = "/kaggle/input/datasets/chetana092004/llama7b-stage1-ckpt/v2/checkpoint-3200"

N_SAMPLES = 5
BATCH_SIZE = 1
MAX_NEW_TOKENS = 1024
N_SHOTS = 3

PRED_PATH = os.path.join(OUT_DIR, "predictions.jsonl")
STATS_PATH = os.path.join(OUT_DIR, "stats.json")
os.makedirs(OUT_DIR, exist_ok=True)


# =====================================================
# PARSER (TEXT FORMAT)
# =====================================================
def parse_stage2_text(pred_text: str) -> Tuple[Optional[Dict[str, Any]], Dict[str, Any]]:
    """
    Parse strict text output into dict:
      {"switches": {s_k:{x:int,y:int}}, "routing_paths": {r_m:[...]}}

    Returns (parsed_or_none, debug_info)
    """
    dbg = {"error": None, "lines": 0}
    txt = pred_text.strip().replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip() for ln in txt.split("\n") if ln.strip()]
    dbg["lines"] = len(lines)

    # Find section indices
    try:
        i_sw = lines.index("SWITCHES")
        i_rt = lines.index("ROUTES")
        i_end = lines.index("END")
    except ValueError:
        dbg["error"] = "Missing one of SWITCHES/ROUTES/END markers"
        return None, dbg

    if not (i_sw < i_rt < i_end):
        dbg["error"] = "Section order invalid"
        return None, dbg

    sw_lines = lines[i_sw+1:i_rt]
    rt_lines = lines[i_rt+1:i_end]

    switches: Dict[str, Dict[str, int]] = {}
    routes: Dict[str, List[str]] = {}

    # Parse switches
    for ln in sw_lines:
        parts = ln.split()
        if len(parts) != 3:
            dbg["error"] = f"Bad switch line: {ln}"
            return None, dbg
        sid, xs, ys = parts
        if not sid.startswith("s_"):
            dbg["error"] = f"Bad switch id: {sid}"
            return None, dbg
        try:
            x = int(xs); y = int(ys)
        except Exception:
            dbg["error"] = f"Non-integer switch coords: {ln}"
            return None, dbg
        switches[sid] = {"x": x, "y": y}

    # Parse routes
    for ln in rt_lines:
        parts = ln.split()
        if len(parts) < 3:
            dbg["error"] = f"Bad route line: {ln}"
            return None, dbg
        rid = parts[0]
        path = parts[1:]
        if not rid.startswith("r_"):
            dbg["error"] = f"Bad route id: {rid}"
            return None, dbg
        routes[rid] = path

    parsed = {"switches": switches, "routing_paths": routes}
    return parsed, dbg



def point_in_blockage(x: int, y: int, b: Dict[str, Any]) -> bool:
    bx, by = int(b["x"]), int(b["y"])
    bw, bh = int(b["width"]), int(b["height"])
    return (bx <= x < bx + bw) and (by <= y < by + bh)

def validate_switches(pred: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, bool]:
    ok = {
        "has_switches": False,
        "switches_int": False,
        "switches_in_bounds": False,
        "switches_outside_blockages": False,
    }

    switches = pred.get("switches")
    if not isinstance(switches, dict) or len(switches) == 0:
        return ok
    ok["has_switches"] = True

    W, H = spec.get("floorplan_dim", [None, None])
    W = int(W) if W is not None else 0
    H = int(H) if H is not None else 0

    blockages = spec.get("blockages", {}) or {}

    all_int = True
    all_in_bounds = True
    all_outside = True

    for _, coord in switches.items():
        if not isinstance(coord, dict) or "x" not in coord or "y" not in coord:
            all_int = False; all_in_bounds = False; all_outside = False
            break

        x, y = coord["x"], coord["y"]
        if not isinstance(x, int) or not isinstance(y, int):
            all_int = False
            try:
                x = int(x); y = int(y)
            except Exception:
                all_in_bounds = False; all_outside = False
                continue

        if not (0 <= x < W and 0 <= y < H):
            all_in_bounds = False

        for b in blockages.values():
            if point_in_blockage(x, y, b):
                all_outside = False
                break

    ok["switches_int"] = all_int
    ok["switches_in_bounds"] = all_in_bounds
    ok["switches_outside_blockages"] = all_outside
    return ok

def validate_routing(pred: Dict[str, Any], spec: Dict[str, Any]) -> Dict[str, bool]:
    ok = {
        "has_routing_paths": False,
        "routes_match_connectivity": False,
        "route_nodes_exist": False,
    }

    paths = pred.get("routing_paths")
    if not isinstance(paths, dict) or len(paths) == 0:
        return ok
    ok["has_routing_paths"] = True

    conn = spec.get("connectivity", {}) or {}
    inits = spec.get("inits", {}) or {}
    targets = spec.get("targets", {}) or {}
    switches = pred.get("switches", {}) or {}

    # 1) endpoints match exactly for all conn keys
    endpoints_ok = True
    for r_id, pair in conn.items():
        if r_id not in paths:
            endpoints_ok = False; break
        if not isinstance(pair, list) or len(pair) != 2:
            endpoints_ok = False; break
        src, dst = pair[0], pair[1]
        p = paths[r_id]
        if not isinstance(p, list) or len(p) < 2:
            endpoints_ok = False; break
        if p[0] != src or p[-1] != dst:
            endpoints_ok = False; break
    ok["routes_match_connectivity"] = endpoints_ok

    # 2) all node ids exist
    nodes_ok = True
    for _, p in paths.items():
        if not isinstance(p, list) or len(p) < 2:
            nodes_ok = False; break
        for node in p:
            if not isinstance(node, str):
                nodes_ok = False; break
            if node.startswith("i_"):
                if node not in inits: nodes_ok = False; break
            elif node.startswith("t_"):
                if node not in targets: nodes_ok = False; break
            elif node.startswith("s_"):
                if node not in switches: nodes_ok = False; break
            else:
                nodes_ok = False; break
        if not nodes_ok:
            break
    ok["route_nodes_exist"] = nodes_ok
    return ok


# =====================================================
# LOAD CONFIG / MODEL
# =====================================================
with open(CFG_PATH) as f:
    cfg = yaml.safe_load(f)

tok = AutoTokenizer.from_pretrained(cfg["model_name"], use_fast=True)
if tok.pad_token is None:
    tok.pad_token = tok.eos_token
tok.padding_side = "left"

quant_cfg = BitsAndBytesConfig(
    load_in_4bit=cfg["load_in_4bit"],
    bnb_4bit_quant_type=cfg["bnb_4bit_quant_type"],
    bnb_4bit_compute_dtype=getattr(torch, cfg["bnb_4bit_compute_dtype"]),
    bnb_4bit_use_double_quant=cfg["bnb_4bit_use_double_quant"],
)

print("Loading base model...")
base = AutoModelForCausalLM.from_pretrained(
    cfg["model_name"],
    quantization_config=quant_cfg,
    device_map="auto",
)
base.config.use_cache = True

print("Loading LoRA adapter...")
model = PeftModel.from_pretrained(base, CKPT_DIR)
model.eval()


print("Loading dataset...")
raw = load_dataset("text", data_files={"test": TEST_FP})["test"]
raw = raw.select(range(min(N_SAMPLES, len(raw))))
data = [json.loads(x["text"]) for x in raw]

def get_spec(row: Dict[str, Any]) -> Dict[str, Any]:
    if "spec" in row and isinstance(row["spec"], dict):
        return row["spec"]
    if "Arch Specification" in row and isinstance(row["Arch Specification"], dict):
        return row["Arch Specification"]
    raise KeyError("Cannot find spec in row. Expected row['spec'].")

def get_gt_network(row: Dict[str, Any]) -> Tuple[Optional[Dict[str, Any]], Optional[Dict[str, Any]]]:
    gt_sw = row.get("switches")
    gt_rt = row.get("routing_paths")
    net = row.get("network")
    if isinstance(net, dict):
        gt_sw = gt_sw or net.get("switches")
        gt_rt = gt_rt or net.get("routing_paths")
    return gt_sw, gt_rt


# =====================================================
# STATS
# =====================================================
stats = {
    "n_total": 0,
    "parsed_ok": 0,
    "has_switches": 0,
    "has_routing_paths": 0,
    "switches_in_bounds": 0,
    "switches_outside_blockages": 0,
    "routes_match_connectivity": 0,
    "route_nodes_exist": 0,
}

print("Starting inference...")
total_batches = (len(data) + BATCH_SIZE - 1) // BATCH_SIZE

with open(PRED_PATH, "w", encoding="utf-8") as fout:
    for batch in tqdm((data[i:i+BATCH_SIZE] for i in range(0, len(data), BATCH_SIZE)),
                      total=total_batches, desc="Batches"):

        specs = [get_spec(r) for r in batch]

        prompts = [
            build_fewshot_stage2_text_prompt(s, n_shots=N_SHOTS)
            for s in specs
        ]

        inputs = tok(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=cfg["max_seq_length"],
        ).to(model.device)

        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
                eos_token_id=tok.eos_token_id,
            )

        # token-slice (correct)
        input_len = inputs["input_ids"].shape[1]
        gen_only = outputs[:, input_len:]
        pred_texts = tok.batch_decode(gen_only, skip_special_tokens=True)

        for row, spec, pred_text in zip(batch, specs, pred_texts):
            pred_text = pred_text.strip()
            parsed, dbg = parse_stage2_text(pred_text)
            gt_sw, gt_rt = get_gt_network(row)

            stats["n_total"] += 1
            ok_parsed = parsed is not None
            stats["parsed_ok"] += int(ok_parsed)

            if not ok_parsed:
                fout.write(json.dumps({
                    "pred_text": pred_text,
                    "parsed": None,
                    "parse_debug": dbg,
                    "gt_switches": gt_sw,
                    "gt_routing_paths": gt_rt,
                    "spec": spec,
                }, ensure_ascii=False) + "\n")
                continue

            sw_ok = validate_switches(parsed, spec)
            rt_ok = validate_routing(parsed, spec)

            stats["has_switches"] += int(sw_ok["has_switches"])
            stats["switches_in_bounds"] += int(sw_ok["switches_in_bounds"])
            stats["switches_outside_blockages"] += int(sw_ok["switches_outside_blockages"])

            stats["has_routing_paths"] += int(rt_ok["has_routing_paths"])
            stats["routes_match_connectivity"] += int(rt_ok["routes_match_connectivity"])
            stats["route_nodes_exist"] += int(rt_ok["route_nodes_exist"])

            fout.write(json.dumps({
                "pred_text": pred_text,          # raw model output (text)
                "parsed": parsed,                # parsed dict
                "parse_debug": dbg,
                "gt_switches": gt_sw,
                "gt_routing_paths": gt_rt,
                "spec": spec,
                "checks": {**sw_ok, **rt_ok},
            }, ensure_ascii=False) + "\n")

def rate(x: int, n: int) -> float:
    return round(x / n, 4) if n else 0.0

n = stats["n_total"]
report = {
    "n_total": n,
    "parsed_ok_rate": rate(stats["parsed_ok"], n),
    "has_switches_rate": rate(stats["has_switches"], n),
    "has_routing_paths_rate": rate(stats["has_routing_paths"], n),
    "switches_in_bounds_rate": rate(stats["switches_in_bounds"], n),
    "switches_outside_blockages_rate": rate(stats["switches_outside_blockages"], n),
    "routes_match_connectivity_rate": rate(stats["routes_match_connectivity"], n),
    "route_nodes_exist_rate": rate(stats["route_nodes_exist"], n),
    "pred_file": PRED_PATH,
}

with open(STATS_PATH, "w") as f:
    json.dump(report, f, indent=2)

print("\n===== FEW-SHOT STAGE2 TEXT VALIDATION =====")
print(json.dumps(report, indent=2))
print("\nSaved predictions to:", PRED_PATH)
print("Saved stats to:", STATS_PATH)
