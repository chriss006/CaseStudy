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

def build_fewshot_stage2_text_prompt(
    test_spec: Dict[str, Any],
    n_shots: int = 1,
) -> str:
    W, H = test_spec["floorplan_dim"]

    conn_list = _pack_connectivity(test_spec.get("connectivity", {}) or {})

    packed = {
        "dim": [int(W), int(H)],
        "in": _pack_points(test_spec.get("inits", {}) or {}),
        "tg": _pack_points(test_spec.get("targets", {}) or {}),
        "co": conn_list,
        "bl": _pack_blockages(test_spec.get("blockages", {}) or {}),
    }

    # Optional: if you add these in spec generation
    if "n_switches" in test_spec:
        packed["n_switches"] = int(test_spec["n_switches"])
    if "free_points" in test_spec:
        # expects [[x,y],...]
        packed["free_points"] = test_spec["free_points"]

    header = (
        "Return ONLY between BEGIN_OUTPUT and END.\n"
        "SWITCHES lines: s_k x y (int).\n"
        "If n_switches is provided: output exactly n_switches switches: s_0..s_{n_switches-1}.\n"
        "If free_points is provided: every switch coordinate MUST be chosen from free_points.\n"
        "All switches inside dim and outside all bl rectangles.\n"
        "ROUTES: one line per co entry, keep r_id/src/dst exactly. Only replace '...'.\n"
        "Each hop uses Manhattan-L (2 segments) and must not cross any blockage.\n"
    )

    blocks: List[str] = [header]

    if n_shots >= 1:
        blocks.append("=== EXAMPLE OUTPUT ===\n")
        blocks.append(EX1_OUT_TEXT + "\n\n")

    blocks.append("=== NOW SOLVE ===\n")
    blocks.append(_dumps_compact(packed) + "\n")
    blocks.append("BEGIN_OUTPUT\nSWITCHES\n")

    # IMPORTANT: give route template so routes_match_connectivity doesn't fail
    blocks.append(_routes_template(conn_list) + "\n")
    blocks.append("END\n")

    return "".join(blocks).strip()
