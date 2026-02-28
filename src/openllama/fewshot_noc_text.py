# fewshot_noc_stage2_text.py
import json
from typing import Dict, Any, List, Tuple

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

EX1_OUT = {
    "switches": {
        "s_0": {"x": 916, "y": 349},
        "s_1": {"x": 677, "y": 985},
        "s_2": {"x": 25, "y": 766},
        "s_3": {"x": 40, "y": 511},
        "s_4": {"x": 921, "y": 819},
        "s_5": {"x": 413, "y": 985}
    },
    "routing_paths": {
        "r_0": ["i_0", "s_4", "s_0", "t_3"],
        "r_1": ["i_0", "s_4", "s_1", "s_2", "t_0"],
        "r_2": ["i_1", "s_0", "t_3"],
        "r_3": ["i_1", "s_0", "s_4", "s_1", "t_1"],
        "r_4": ["i_1", "s_0", "s_4", "s_1", "s_2", "t_0"],
        "r_5": ["i_2", "s_3", "s_2", "s_5", "t_4"],
        "r_6": ["i_2", "s_3", "s_2", "t_0"],
        "r_7": ["i_3", "s_3", "s_2", "s_5", "s_1", "t_1"],
        "r_8": ["i_3", "s_3", "s_2", "t_0"],
        "r_9": ["i_3", "s_3", "t_2"],
        "r_10": ["i_4", "s_2", "s_5", "t_4"]
    }
}

EX2_SPEC = {
    "inits": {
        "i_0": {"x": 848, "y": 189},
        "i_1": {"x": 35, "y": 878},
        "i_2": {"x": 258, "y": 63},
        "i_3": {"x": 170, "y": 62}
    },
    "targets": {
        "t_0": {"x": 742, "y": 992},
        "t_1": {"x": 880, "y": 345},
        "t_2": {"x": 834, "y": 236},
        "t_3": {"x": 28, "y": 880}
    },
    "connectivity": {
        "r_0": ["i_0", "t_3"],
        "r_1": ["i_0", "t_1"],
        "r_2": ["i_1", "t_2"],
        "r_3": ["i_1", "t_3"],
        "r_4": ["i_2", "t_3"],
        "r_5": ["i_2", "t_0"],
        "r_6": ["i_3", "t_1"],
        "r_7": ["i_3", "t_0"],
        "r_8": ["i_3", "t_3"]
    },
    "floorplan_dim": [1000, 1000],
    "blockages": {
        "b_0": {"x": 15, "y": 65, "width": 817, "height": 619},
        "b_1": {"x": 12, "y": 265, "width": 26, "height": 31},
        "b_2": {"x": 280, "y": 674, "width": 315, "height": 212},
        "b_3": {"x": 12, "y": 833, "width": 61, "height": 43},
        "b_4": {"x": 424, "y": 285, "width": 452, "height": 367},
        "b_5": {"x": 518, "y": 882, "width": 221, "height": 110},
        "b_6": {"x": 901, "y": 22, "width": 39, "height": 43},
        "b_7": {"x": 706, "y": 24, "width": 165, "height": 43},
        "b_8": {"x": 852, "y": 183, "width": 121, "height": 16},
        "b_9": {"x": 651, "y": 269, "width": 341, "height": 43},
        "b_10": {"x": 845, "y": 318, "width": 96, "height": 19},
        "b_11": {"x": 648, "y": 571, "width": 348, "height": 327}
    }
}

EX2_OUT = {
    "switches": {
        "s_0": {"x": 28, "y": 878},
        "s_1": {"x": 170, "y": 63},
        "s_2": {"x": 872, "y": 182}
    },
    "routing_paths": {
        "r_0": ["i_0", "s_2", "s_1", "s_0", "t_3"],
        "r_1": ["i_0", "s_2", "t_1"],
        "r_2": ["i_1", "s_0", "s_1", "s_2", "t_2"],
        "r_3": ["i_1", "s_0", "t_3"],
        "r_4": ["i_2", "s_1", "s_0", "t_3"],
        "r_5": ["i_2", "s_1", "s_0", "t_0"],
        "r_6": ["i_3", "s_1", "s_2", "t_1"],
        "r_7": ["i_3", "s_1", "s_0", "t_0"],
        "r_8": ["i_3", "s_1", "s_0", "t_3"]
    }
}

EX3_SPEC = {
    "inits": {
        "i_0": {"x": 365, "y": 946},
        "i_1": {"x": 408, "y": 946},
        "i_2": {"x": 16, "y": 899},
        "i_3": {"x": 15, "y": 926},
        "i_4": {"x": 976, "y": 175}
    },
    "targets": {
        "t_0": {"x": 975, "y": 380},
        "t_1": {"x": 944, "y": 0},
        "t_2": {"x": 804, "y": 946},
        "t_3": {"x": 935, "y": 0},
        "t_4": {"x": 976, "y": 187},
        "t_5": {"x": 977, "y": 636}
    },
    "connectivity": {
        "r_0": ["i_0", "t_0"],
        "r_1": ["i_0", "t_3"],
        "r_2": ["i_0", "t_1"],
        "r_3": ["i_1", "t_1"],
        "r_4": ["i_1", "t_5"],
        "r_5": ["i_1", "t_2"],
        "r_6": ["i_2", "t_5"],
        "r_7": ["i_2", "t_1"],
        "r_8": ["i_2", "t_2"],
        "r_9": ["i_3", "t_3"],
        "r_10": ["i_3", "t_5"],
        "r_11": ["i_4", "t_4"],
        "r_12": ["i_4", "t_5"],
        "r_13": ["i_4", "t_3"]
    },
    "floorplan_dim": [1000, 1000],
    "blockages": {
        "b_0": {"x": 336, "y": 1, "width": 47, "height": 160},
        "b_1": {"x": 17, "y": 58, "width": 33, "height": 99},
        "b_2": {"x": 19, "y": 3, "width": 954, "height": 942}
    }
}

EX3_OUT = {
    "switches": {
        "s_0": {"x": 976, "y": 636},
        "s_1": {"x": 408, "y": 946},
        "s_2": {"x": 16, "y": 926},
        "s_3": {"x": 804, "y": 946},
        "s_4": {"x": 944, "y": 0},
        "s_5": {"x": 976, "y": 380},
        "s_6": {"x": 976, "y": 187}
    },
    "routing_paths": {
        "r_0": ["i_0", "s_1", "s_3", "s_0", "s_5", "t_0"],
        "r_1": ["i_0", "s_1", "s_3", "s_0", "s_5", "s_6", "s_4", "t_3"],
        "r_2": ["i_0", "s_1", "s_3", "s_0", "s_5", "s_6", "s_4", "t_1"],
        "r_3": ["i_1", "s_1", "s_3", "s_0", "s_5", "s_6", "s_4", "t_1"],
        "r_4": ["i_1", "s_1", "s_3", "s_0", "t_5"],
        "r_5": ["i_1", "s_1", "s_3", "t_2"],
        "r_6": ["i_2", "s_2", "s_1", "s_3", "s_0", "t_5"],
        "r_7": ["i_2", "s_2", "s_1", "s_3", "s_0", "s_5", "s_6", "s_4", "t_1"],
        "r_8": ["i_2", "s_2", "s_1", "s_3", "t_2"],
        "r_9": ["i_3", "s_2", "s_1", "s_3", "s_0", "s_5", "s_6", "s_4", "t_3"],
        "r_10": ["i_3", "s_2", "s_1", "s_3", "s_0", "t_5"],
        "r_11": ["i_4", "s_6", "t_4"],
        "r_12": ["i_4", "s_6", "s_0", "t_5"],
        "r_13": ["i_4", "s_6", "s_4", "t_3"]
    }
}


DEFAULT_EXAMPLES: List[Tuple[Dict[str, Any], Dict[str, Any]]] = [
    (EX1_SPEC, EX1_OUT),
    (EX2_SPEC, EX2_OUT),
    (EX3_SPEC, EX3_OUT),
]

def _render_output_as_text(out: Dict[str, Any]) -> str:
    """
    Convert {"switches":{...},"routing_paths":{...}} into strict text format.
    """
    switches = out.get("switches", {}) or {}
    routes   = out.get("routing_paths", {}) or {}

    lines = []
    lines.append("SWITCHES")
    # deterministic order by key
    for sid in sorted(switches.keys(), key=lambda x: (len(x), x)):
        xy = switches[sid]
        lines.append(f"{sid} {int(xy['x'])} {int(xy['y'])}")

    lines.append("ROUTES")
    for rid in sorted(routes.keys(), key=lambda x: (len(x), x)):
        path = routes[rid]
        # "r_0 i_0 s_4 s_0 t_3"
        lines.append(f"{rid} " + " ".join(path))

    lines.append("END")
    return "\n".join(lines)

def build_fewshot_stage2_text_prompt(
    test_spec: Dict[str, Any],
    examples: List[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    n_shots: int = 3,
) -> str:
    """
    Stage2 few-shot, but output is STRICT TEXT (not JSON).
    """
    if examples is None:
        examples = DEFAULT_EXAMPLES
    n_shots = max(0, min(n_shots, len(examples)))

    test_spec = _stable_spec(test_spec)

    header = (
        "You are an expert NoC physical designer.\n\n"
        "TASK\n"
        "Given an architecture specification JSON (inits, targets, connectivity, floorplan_dim, blockages),\n"
        "predict a valid network design: switch placements and routing paths.\n\n"
        "OUTPUT FORMAT (STRICT, TEXT ONLY)\n"
        "You MUST output exactly three sections in this order:\n"
        "1) SWITCHES\n"
        "   Each line: s_k X Y\n"
        "2) ROUTES\n"
        "   Each line: r_m node0 node1 ... nodeN\n"
        "   node0 must be the init i_* and nodeN must be the target t_*.\n"
        "3) END\n\n"
        "RULES\n"
        "- Output TEXT ONLY. No JSON, no markdown, no explanations.\n"
        "- X and Y must be integers.\n"
        "- Switches must be inside floorplan and outside all blockages.\n"
        "- For each r_id in connectivity, you MUST output exactly one ROUTES line with that r_id.\n"
        "- Route endpoints must match connectivity: start=i_*, end=t_*.\n"
        "- Route nodes can be only i_*, s_*, t_* and all used s_* must exist in SWITCHES.\n"
        "- Prefer fewer switches and shorter routes.\n\n"
    )

    blocks = [header]

    for i in range(n_shots):
        ex_spec, ex_out = examples[i]
        ex_spec = _stable_spec(ex_spec)

        blocks.append("======================\n")
        blocks.append(f"EXAMPLE {i+1}\n")
        blocks.append("======================\n")
        blocks.append("-- Arch Specification (JSON) --\n")
        blocks.append(_dumps_compact(ex_spec) + "\n")
        blocks.append("-- Output (TEXT ONLY) --\n")
        blocks.append(_render_output_as_text(ex_out) + "\n\n")

    blocks.append("======================\n")
    blocks.append("NOW SOLVE THIS (NEW INPUT)\n")
    blocks.append("======================\n")
    blocks.append("-- Arch Specification (JSON) --\n")
    blocks.append(_dumps_compact(test_spec) + "\n")
    blocks.append("-- Output (TEXT ONLY) --\n")

    return "".join(blocks).strip()
