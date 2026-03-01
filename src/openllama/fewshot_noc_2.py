import json
from typing import Any, Dict, List, Tuple, Optional

def _dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def _pack_points(d: Dict[str, Dict[str, Any]]) -> List[List[Any]]:
    out: List[List[Any]] = []
    for k, v in d.items():
        out.append([k, int(v["x"]), int(v["y"])])
    out.sort(key=lambda x: x[0])
    return out

def _pack_blockages(blockages: Dict[str, Dict[str, Any]]) -> List[List[int]]:
    out: List[List[int]] = []
    for b in (blockages or {}).values():
        out.append([int(b["x"]), int(b["y"]), int(b["width"]), int(b["height"])])
    out.sort()
    return out

def _pack_connectivity(conn: Dict[str, List[str]]) -> List[List[str]]:
    out: List[List[str]] = []
    for r, pair in conn.items():
        out.append([r, pair[0], pair[1]])
    out.sort(key=lambda x: x[0])
    return out

# --- Keep your original EX1_OUT_TEXT (this is the "shot") ---
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
    n_shots: int = 1,  # keep 1-shot
) -> str:
    """
    1-shot prompt but fits 1024 tokens by:
      - using ONLY output example (no example JSON spec)
      - compressing test_spec into short arrays
      - very short header
    """

    W, H = test_spec["floorplan_dim"]
    packed = {
        "dim": [int(W), int(H)],
        "in": _pack_points(test_spec["inits"]),
        "tg": _pack_points(test_spec["targets"]),
        "co": _pack_connectivity(test_spec["connectivity"]),
        "bl": _pack_blockages(test_spec.get("blockages", {}) or {}),
    }

    header = (
        "Return ONLY between BEGIN_OUTPUT and END.\n"
        "Format: SWITCHES then ROUTES.\n"
        "All switches inside dim and outside all bl rectangles.\n"
        "For every hop A->B, use Manhattan-L (2 segments) avoiding bl.\n"
        "Include ALL routes in co with same ids.\n"
    )

    blocks: List[str] = [header]

    if n_shots >= 1:
        blocks.append("=== EXAMPLE OUTPUT ===\n")
        blocks.append(EX1_OUT_TEXT + "\n\n")

    blocks.append("=== NOW SOLVE ===\n")
    blocks.append(_dumps_compact(packed) + "\n")
    blocks.append("BEGIN_OUTPUT\nSWITCHES\n")
    blocks.append("ROUTES\n")
    blocks.append("END\n")

    return "".join(blocks).strip()
