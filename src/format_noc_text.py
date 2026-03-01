import json
from typing import Any, Dict, List, Tuple, Optional

SPEC_KEY_ORDER = ["inits", "targets", "connectivity", "floorplan_dim", "blockages"]

def _stable_spec(spec: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for k in SPEC_KEY_ORDER:
        if k in spec:
            out[k] = spec[k]
    for k in spec.keys():
        if k not in out:
            out[k] = spec[k]
    return out

def _dumps_compact(obj: Any) -> str:
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":"))

def _compress_blockages_for_prompt(spec: Dict[str, Any]) -> Dict[str, Any]:
    """
    Keep blockages but compress to reduce tokens:
      blockages_compact: [[x,y,w,h], ...]
    Also keep original keys inits/targets/connectivity/floorplan_dim.
    """
    spec = dict(spec)
    blockages = spec.get("blockages", {}) or {}
    compact: List[List[int]] = []
    for _, b in blockages.items():
        # tolerate int/float inputs, cast to int for prompt clarity
        compact.append([int(b["x"]), int(b["y"]), int(b["width"]), int(b["height"])])
    # Stable ordering for determinism (helps caching/debug)
    compact.sort(key=lambda r: (r[0], r[1], r[2], r[3]))

    # Replace heavy dict with compact list
    spec["blockages_compact"] = compact
    spec.pop("blockages", None)
    return spec

# ---------------- Example ----------------

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

# ---------------- Prompt builder ----------------

def build_fewshot_stage2_text_prompt(
    test_spec: Dict[str, Any],
    examples: Optional[List[Tuple[Dict[str, Any], str]]] = None,
    n_shots: int = 1,
    compress_blockages: bool = True,
) -> str:
    """
    Few-shot prompt for stage2: LLM predicts BOTH switch positions and routes.

    Key fixes vs your current code:
      1) Do NOT strip blockages. (You need them to place switches outside and route around.)
      2) Actually include the test_spec JSON in the NOW SOLVE section.
      3) Remove dummy outputs that bias the model (s_0 0 0 / r_0 i_0 t_0).
      4) Explicitly specify Manhattan-L (2-segment) connectivity rule.
      5) Both segments of the L must be completely outside blockages.
    """
    if examples is None:
        examples = DEFAULT_EXAMPLES
    n_shots = max(0, min(n_shots, len(examples)))

    def prep_spec(s: Dict[str, Any]) -> Dict[str, Any]:
        s = _stable_spec(s)
        return _compress_blockages_for_prompt(s) if compress_blockages else s

    test_spec_prompt = prep_spec(test_spec)

    header = (
    "You are a professional NoC physical design solver.\n"
    "You must generate a VALID, OPTIMIZED solution.\n\n"

    "If any rule is violated, the output is considered WRONG.\n"
    "Double-check all constraints before answering.\n\n"

    "================ OUTPUT FORMAT ================\n"
    "Return ONLY the block between BEGIN_OUTPUT and END.\n\n"

    "BEGIN_OUTPUT\n"
    "SWITCHES\n"
    "s_k x y\n"
    "...\n"
    "ROUTES\n"
    "r_m node0 node1 ... nodeN\n"
    "END\n\n"

    "================ OBJECTIVE ================\n"
    "- Minimize total Manhattan wirelength.\n"
    "- Minimize number of switches.\n"
    "- Prefer short routes with few hops.\n\n"

    "================ SWITCH RULES ================\n"
    "- All coordinates must be integers.\n"
    "- Must be inside floorplan_dim.\n"
    "- Must be strictly OUTSIDE every blockage.\n"
    "- Do NOT place redundant or unused switches.\n"
    "- Place switches near centroids of connected inits/targets.\n"
    "- Switch ids must be exactly: s_0, s_1, ..., s_K (no gaps, no extra ids).\n"
    "- Use at most 2 × (number of inits + targets) switches.\n\n"

    "================ ROUTING RULES ================\n"
    "- Must include ALL routes from connectivity.\n"
    "- Each route: init → switches → target.\n"
    "- Only use switches you created.\n"
    "- No skipped endpoints.\n"
    "- Do NOT invent new node ids.\n\n"

    "================ MANHATTAN-L RULE (HARD) ================\n"
    "- Each hop A→B must be connectable by exactly TWO axis-aligned segments:\n"
    "  option1: (x1,y1)->(x2,y1)->(x2,y2)\n"
    "  option2: (x1,y1)->(x1,y2)->(x2,y2)\n"
    "- BOTH segments must avoid blockages.\n"
    "- Choose option with NO blockage intersection.\n"
    "- If neither works, insert extra switches.\n\n"

    "================ BLOCKAGES ================\n"
    "- blockages_compact = [x,y,width,height]\n"
    "- Forbidden area: x∈[bx,bx+w], y∈[by,by+h]\n"
    "- Do NOT touch borders.\n\n"

    "================ INTERNAL CHECK (DO SILENTLY) ================\n"
    "Before finalizing:\n"
    "1. Verify every switch is legal.\n"
    "2. Verify every route exists.\n"
    "3. Verify every hop is Manhattan-L valid.\n"
    "4. Verify no blockage intersections.\n"
    "5. Verify no unused or extra switches.\n"
    "6. Verify no invented ids.\n\n"
)

    blocks: List[str] = [header]

    for i in range(n_shots):
        ex_spec, ex_out_text = examples[i]
        ex_spec_prompt = prep_spec(ex_spec)

        blocks.append("=== EXAMPLE ===\n")
        blocks.append(_dumps_compact(ex_spec_prompt) + "\n")
        blocks.append(ex_out_text.strip() + "\n\n")

    blocks.append("=== NOW SOLVE ===\n")
    blocks.append(_dumps_compact(test_spec_prompt) + "\n")
    blocks.append("BEGIN_OUTPUT\n")
    blocks.append("SWITCHES\n")
    blocks.append("ROUTES\n")
    blocks.append("END\n")

    return "".join(blocks).strip()
