import json
from typing import Any, Dict, List, Tuple, Optional

def _dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def _pack_points(d: Dict[str, Dict[str, Any]]) -> List[List[Any]]:
    out = [[k, int(v["x"]), int(v["y"])] for k, v in (d or {}).items()]
    out.sort(key=lambda x: x[0])
    return out

def _pack_blockages(blockages: Dict[str, Dict[str, Any]]) -> List[List[int]]:
    out = []
    for b in (blockages or {}).values():
        out.append([int(b["x"]), int(b["y"]), int(b["width"]), int(b["height"])])
    out.sort()
    return out

def _pack_connectivity(conn: Dict[str, List[str]]) -> List[List[str]]:
    out = []
    for r, pair in (conn or {}).items():
        out.append([r, pair[0], pair[1]])
    out.sort(key=lambda x: x[0])
    return out

def _routes_template(conn_list: List[List[str]]) -> str:
    # conn_list: [[r_id, src, dst], ...]
    lines = ["ROUTES"]
    for r_id, src, dst in conn_list:
        lines.append(f"{r_id} {src} ... {dst}")
    return "\n".join(lines)

EX1_OUT_TEXT = """BEGIN_OUTPUT
SWITCHES
s_0 916 349
s_1 677 985
s_2 25 766
s_3 40 511
s_4 921 819
s_5 413 985
ROUTES
r_0 i_0 s_4 s_0 t_3
r_1 i_0 s_4 s_1 s_2 t_0
r_2 i_1 s_0 t_3
r_3 i_1 s_0 s_4 s_1 t_1
r_4 i_1 s_0 s_4 s_1 s_2 t_0
r_5 i_2 s_3 s_2 s_5 t_4
r_6 i_2 s_3 s_2 t_0
r_7 i_3 s_3 s_2 s_5 s_1 t_1
r_8 i_3 s_3 s_2 t_0
r_9 i_3 s_3 t_2
r_10 i_4 s_2 s_5 t_4
END
""".strip()

def build_fewshot_stage2_text_prompt(spec: Dict[str, Any], n_shots: int = 1) -> str:
    # pack
    W, H = spec["floorplan_dim"]
    inits = [[k, int(v["x"]), int(v["y"])] for k,v in spec["inits"].items()]
    tgs   = [[k, int(v["x"]), int(v["y"])] for k,v in spec["targets"].items()]
    inits.sort(key=lambda x: x[0]); tgs.sort(key=lambda x: x[0])

    co = [[r, pair[0], pair[1]] for r, pair in spec["connectivity"].items()]
    co.sort(key=lambda x: x[0])

    bl = [[int(b["x"]), int(b["y"]), int(b["width"]), int(b["height"])] for b in (spec.get("blockages") or {}).values()]
    bl.sort()

    # free_points: keep small to fit context
    fp = spec.get("free_points", [])
    if isinstance(fp, list) and len(fp) > 96:
        fp = fp[:96]  # hard cap

    packed = {"dim":[int(W),int(H)], "in":inits, "tg":tgs, "co":co, "bl":bl, "fp":fp}

    header = (
        "Return ONLY between BEGIN_OUTPUT and END.\n"
        "SWITCHES: output s_0..s_{K-1} with integer x y.\n"
        "Each switch must use a coordinate from fp (free points).\n"
        "ROUTES: output one line per co entry, and each route must start/end exactly as in co.\n"
        "No extra text.\n"
    )

    blocks = [header]
    if n_shots >= 1:
        blocks.append("=== EXAMPLE OUTPUT ===\n")
        blocks.append(EX1_OUT_TEXT.strip() + "\n\n")

    # NOW SOLVE
    blocks.append("=== NOW SOLVE ===\n")
    blocks.append(_dumps_compact(packed) + "\n")
    blocks.append("BEGIN_OUTPUT\nSWITCHES\n")

    # very short route skeleton: r_id src dst only
    blocks.append("ROUTES\n")
    for r, src, dst in co:
        blocks.append(f"{r} {src} {dst}\n")
    blocks.append("END\n")
    return "".join(blocks)
