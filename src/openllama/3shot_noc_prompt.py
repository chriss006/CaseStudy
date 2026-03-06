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


# ---------------------------------------------------------------------
# 3 REAL EXAMPLES (from your message)
# ---------------------------------------------------------------------

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


def build_fewshot_stage2_prompt(
    test_spec: Dict[str, Any],
    examples: List[Tuple[Dict[str, Any], Dict[str, Any]]] = None,
    n_shots: int = 3,
) -> str:
    """
    Few-shot prompt for Stage 2:
    Spec -> switches + routing_paths (single-stage generation)
    """
    if examples is None:
        examples = DEFAULT_EXAMPLES

    if n_shots < 0:
        n_shots = 0
    n_shots = min(n_shots, len(examples))

    test_spec = _stable_spec(test_spec)

    header = (
        "You are an expert NoC physical designer.\n\n"
        "TASK\n"
        "Given an architecture specification JSON (inits, targets, connectivity, floorplan_dim, blockages),\n"
        "output ONE valid network design as a SINGLE JSON object with exactly two keys:\n"
        "- \"switches\": dict of switch coordinates\n"
        "- \"routing_paths\": dict of routes, each route is a list of node IDs\n\n"
        "OUTPUT JSON SCHEMA (MUST MATCH)\n"
        "{"
        "\"switches\":{\"s_0\":{\"x\":int,\"y\":int}},"
        "\"routing_paths\":{\"r_0\":[\"i_0\",\"s_0\",\"t_0\"]}"
        "}\n\n"
        "VALIDITY RULES (MUST SATISFY ALL)\n"
        "1) Output JSON ONLY. No extra text, no markdown, no code fences.\n"
        "2) Use double quotes only. No trailing commas.\n"
        "3) All coordinates must be integers.\n"
        "4) Floorplan bounds: 0 <= x < W and 0 <= y < H, where floorplan_dim = [W, H].\n"
        "5) Blockage rectangles: blocked region is x <= X < x+width and y <= Y < y+height.\n"
        "   Every switch (X,Y) MUST be outside all blocked regions.\n"
        "6) Route endpoints must match connectivity exactly.\n"
        "   routing_paths[r_k] MUST start with i_u and end with t_v.\n"
        "7) Path nodes must be existing IDs: i_*, s_*, t_*.\n\n"
        "GOAL\n"
        "- Prefer fewer switches when possible.\n"
        "- Prefer shorter routing paths (fewer hops).\n"
        "- Reuse switches across routes if it reduces total switches.\n\n"
    )

    blocks = [header]

    for idx in range(n_shots):
        ex_spec, ex_out = examples[idx]
        ex_spec = _stable_spec(ex_spec)
        blocks.append("======================\n")
        blocks.append(f"EXAMPLE {idx+1}\n")
        blocks.append("======================\n")
        blocks.append("-- Arch Specification (JSON) --\n")
        blocks.append(_dumps_compact(ex_spec) + "\n\n")
        blocks.append("-- Output (JSON only) --\n")
        blocks.append(_dumps_compact(ex_out) + "\n\n")

    blocks.append("======================\n")
    blocks.append("NOW SOLVE THIS (NEW INPUT)\n")
    blocks.append("======================\n")
    blocks.append("-- Arch Specification (JSON) --\n")
    blocks.append(_dumps_compact(test_spec) + "\n\n")
    blocks.append("-- Output (JSON only) --\n")

    return "".join(blocks).strip()
