import json
from typing import Any, Dict, List, Tuple, Optional

SPEC_KEY_ORDER = ["inits", "targets", "connectivity", "floorplan_dim", "blockages"]

def _stable_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    out = {}
    for k in SPEC_KEY_ORDER:
        if k in spec:
            out[k] = spec[k]
    for k in spec.keys():
        if k not in out:
            out[k] = spec[k]
    return out

def _dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def _strip_blockages_for_prompt(spec: Dict[str, Any]) -> Dict[str, Any]:
    spec = dict(spec)
    spec["blockages"] = {} 
    return spec

EX1_SPEC = {
    "inits": {
        "i_0": {"x": 841, "y": 819},
        "i_1": {"x": 894, "y": 289},
        "i_2": {"x": 168, "y": 579},
        "i_3": {"x": 40, "y": 511},
        "i_4": {"x": 162, "y": 641}
    },
    "targets": {
        "t_0": {"x": 23, "y": 766},
        "t_1": {"x": 760, "y": 988},
        "t_2": {"x": 87, "y": 4},
        "t_3": {"x": 829, "y": 29},
        "t_4": {"x": 413, "y": 988}
    },
    "connectivity": {
        "r_0": ["i_0", "t_3"],
        "r_1": ["i_0", "t_0"],
        "r_2": ["i_1", "t_3"],
        "r_3": ["i_1", "t_1"],
        "r_4": ["i_1", "t_0"],
        "r_5": ["i_2", "t_4"],
        "r_6": ["i_2", "t_0"],
        "r_7": ["i_3", "t_1"],
        "r_8": ["i_3", "t_0"],
        "r_9": ["i_3", "t_2"],
        "r_10": ["i_4", "t_4"]
    },
    "floorplan_dim": [1000, 1000],
    "blockages": {
        "b_0": {"x": 172, "y": 33, "width": 666, "height": 951},
        "b_1": {"x": 2, "y": 0, "width": 82, "height": 238},
        "b_2": {"x": 27, "y": 512, "width": 132, "height": 45},
        "b_3": {"x": 26, "y": 644, "width": 247, "height": 277},
        "b_4": {"x": 131, "y": 967, "width": 22, "height": 26},
        "b_5": {"x": 785, "y": 172, "width": 105, "height": 87},
        "b_6": {"x": 898, "y": 205, "width": 61, "height": 143},
        "b_7": {"x": 826, "y": 537, "width": 101, "height": 192}
    }
}

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
"""

DEFAULT_EXAMPLES: List[Tuple[Dict[str, Any], str]] = [
    (EX1_SPEC, EX1_OUT_TEXT),
]

def build_fewshot_stage2_text_prompt(
    test_spec: Dict[str, Any],
    examples: Optional[List[Tuple[Dict[str, Any], str]]] = None,
    n_shots: int = 1,
) -> str:
    if examples is None:
        examples = DEFAULT_EXAMPLES
    n_shots = max(0, min(n_shots, len(examples)))

    test_spec = _stable_spec(test_spec)
    test_spec_prompt = _strip_blockages_for_prompt(test_spec)

    header = (
        "You are an expert NoC physical designer.\n"
        "Given an architecture specification, output switch placement and routing paths.\n\n"
        "IMPORTANT FORMAT RULES (MUST FOLLOW EXACTLY):\n"
        "1) Output TEXT ONLY (not JSON).\n"
        "2) The output MUST start with 'BEGIN_OUTPUT' on its own line.\n"
        "3) Then exactly these sections in order:\n"
        "   SWITCHES\n"
        "   <one per line: s_k x y>\n"
        "   ROUTES\n"
        "   <one per line: r_m node0 node1 ... nodeN>\n"
        "   END\n"
        "4) x and y must be integers.\n"
        "5) Every route r_m MUST start with its init i_* and end with its target t_* exactly as in connectivity.\n"
        "6) Route nodes can only be i_*, s_*, t_*.\n\n"
        "Spec JSON (blockages may be omitted here to save space; still avoid obstacles implicitly).\n"
    )

    blocks = [header]

    for i in range(n_shots):
        ex_spec, ex_out_text = examples[i]
        ex_spec = _stable_spec(ex_spec)
        ex_spec_prompt = _strip_blockages_for_prompt(ex_spec)

        blocks.append("=== EXAMPLE ===\n")
        blocks.append(_dumps_compact(ex_spec_prompt) + "\n")
        blocks.append(ex_out_text.strip() + "\n\n")

    blocks.append("=== NOW SOLVE ===\n")
    blocks.append(_dumps_compact(test_spec_prompt) + "\n")
    blocks.append("BEGIN_OUTPUT\nSWITCHES\n")  # 여기부터 강제 시작!

    return "".join(blocks).strip()
