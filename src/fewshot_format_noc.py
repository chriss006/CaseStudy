import json


def build_fewshot_full_prompt(test_spec):
    """
    Few-shot prompt for:
    Spec -> Switch placement (single stage)
    Uses 3 full real examples
    """

    # =========================
    # Example 1
    # =========================

    EX1_SPEC = {
        "inits": {
            "i_0": {"x": 379, "y": 285},
            "i_1": {"x": 351, "y": 56},
            "i_2": {"x": 998, "y": 870},
            "i_3": {"x": 775, "y": 314},
            "i_4": {"x": 994, "y": 992}
        },
        "targets": {
            "t_0": {"x": 458, "y": 317},
            "t_1": {"x": 949, "y": 241},
            "t_2": {"x": 846, "y": 238},
            "t_3": {"x": 44, "y": 390},
            "t_4": {"x": 73, "y": 323}
        },
        "connectivity": {
            "r_0": ["i_0", "t_0"],
            "r_1": ["i_1", "t_0"],
            "r_2": ["i_1", "t_4"],
            "r_3": ["i_1", "t_1"],
            "r_4": ["i_1", "t_2"],
            "r_5": ["i_1", "t_3"],
            "r_6": ["i_2", "t_2"],
            "r_7": ["i_2", "t_0"],
            "r_8": ["i_3", "t_2"],
            "r_9": ["i_4", "t_2"],
            "r_10": ["i_4", "t_3"],
            "r_11": ["i_4", "t_4"],
            "r_12": ["i_4", "t_0"]
        },
        "floorplan_dim": [1000, 1000],
        "blockages": {
            "b_0": {"x": 215, "y": 57, "width": 163, "height": 89},
            "b_1": {"x": 381, "y": 84, "width": 461, "height": 229},
            "b_2": {"x": 47, "y": 385, "width": 199, "height": 222},
            "b_3": {"x": 51, "y": 327, "width": 944, "height": 662},
            "b_4": {"x": 848, "y": 243, "width": 137, "height": 47}
        }
    }

    EX1_OUT = {
        "switches": {
            "s_0": {"x": 847, "y": 317},
            "s_1": {"x": 458, "y": 317},
            "s_2": {"x": 73, "y": 323},
            "s_3": {"x": 996, "y": 870},
            "s_4": {"x": 847, "y": 286},
            "s_5": {"x": 775, "y": 314}
        }
    }

    # =========================
    # Example 2
    # =========================

    EX2_SPEC = {
        "inits": {
            "i_0": {"x": 889, "y": 824},
            "i_1": {"x": 887, "y": 816},
            "i_2": {"x": 577, "y": 983},
            "i_3": {"x": 7, "y": 150},
            "i_4": {"x": 889, "y": 773},
            "i_5": {"x": 806, "y": 369}
        },
        "targets": {
            "t_0": {"x": 96, "y": 983},
            "t_1": {"x": 449, "y": 983},
            "t_2": {"x": 563, "y": 48},
            "t_3": {"x": 5, "y": 230},
            "t_4": {"x": 286, "y": 984},
            "t_5": {"x": 888, "y": 521}
        },
        "connectivity": {
            "r_0": ["i_0", "t_0"],
            "r_1": ["i_0", "t_5"],
            "r_2": ["i_1", "t_2"],
            "r_3": ["i_1", "t_0"],
            "r_4": ["i_1", "t_5"],
            "r_5": ["i_1", "t_1"],
            "r_6": ["i_2", "t_3"],
            "r_7": ["i_2", "t_2"]
        },
        "floorplan_dim": [1000, 1000],
        "blockages": {
            "b_0": {"x": 9, "y": 50, "width": 795, "height": 930},
            "b_1": {"x": 518, "y": 400, "width": 368, "height": 547}
        }
    }

    EX2_OUT = {
        "switches": {
            "s_0": {"x": 888, "y": 816},
            "s_1": {"x": 888, "y": 473},
            "s_2": {"x": 563, "y": 49},
            "s_3": {"x": 449, "y": 983},
            "s_4": {"x": 8, "y": 150},
            "s_5": {"x": 888, "y": 369}
        }
    }

    # =========================
    # Example 3
    # =========================

    EX3_SPEC = {
        "inits": {
            "i_0": {"x": 688, "y": 940},
            "i_1": {"x": 20, "y": 8},
            "i_2": {"x": 10, "y": 208},
            "i_3": {"x": 18, "y": 7}
        },
        "targets": {
            "t_0": {"x": 13, "y": 59},
            "t_1": {"x": 232, "y": 860},
            "t_2": {"x": 984, "y": 597},
            "t_3": {"x": 592, "y": 329},
            "t_4": {"x": 968, "y": 941}
        },
        "connectivity": {
            "r_0": ["i_0", "t_0"],
            "r_1": ["i_0", "t_4"],
            "r_2": ["i_1", "t_3"],
            "r_3": ["i_1", "t_0"],
            "r_4": ["i_1", "t_2"],
            "r_5": ["i_2", "t_4"],
            "r_6": ["i_2", "t_0"],
            "r_7": ["i_3", "t_2"],
            "r_8": ["i_3", "t_1"],
            "r_9": ["i_3", "t_4"]
        },
        "floorplan_dim": [1000, 1000],
        "blockages": {
            "b_0": {"x": 14, "y": 11, "width": 577, "height": 848},
            "b_1": {"x": 130, "y": 904, "width": 61, "height": 56},
            "b_2": {"x": 388, "y": 447, "width": 592, "height": 492},
            "b_3": {"x": 771, "y": 178, "width": 150, "height": 129},
            "b_4": {"x": 739, "y": 946, "width": 202, "height": 22}
        }
    }

    EX3_OUT = {
        "switches": {
            "s_0": {"x": 981, "y": 597},
            "s_1": {"x": 968, "y": 940},
            "s_2": {"x": 13, "y": 33},
            "s_3": {"x": 688, "y": 940},
            "s_4": {"x": 13, "y": 208},
            "s_5": {"x": 592, "y": 329}
        }
    }

    # =========================
    # Build Prompt
    # =========================

    prompt = f"""
You are an expert NoC physical designer.

Given an architecture specification in JSON,
predict optimal switch placement.

Rules:
- Output ONLY valid JSON
- No explanation
- No markdown
- No text outside JSON
- All coordinates must be integers
- Do NOT place switches in blockages
- Keep inside floorplan

Output format:
{{
  "switches": {{
    "s_0": {{"x":int,"y":int}},
    "s_1": {{"x":int,"y":int}}
  }}
}}

### Example 1

Specification:
{json.dumps(EX1_SPEC, separators=(",", ":"))}

Output:
{json.dumps(EX1_OUT, separators=(",", ":"))}

### Example 2

Specification:
{json.dumps(EX2_SPEC, separators=(",", ":"))}

Output:
{json.dumps(EX2_OUT, separators=(",", ":"))}

### Example 3

Specification:
{json.dumps(EX3_SPEC, separators=(",", ":"))}

Output:
{json.dumps(EX3_OUT, separators=(",", ":"))}

### Now solve:

Specification:
{json.dumps(test_spec, separators=(",", ":"))}

Output:
"""

    return prompt.strip()
