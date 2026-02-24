import json
from typing import Dict, Any

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

def build_prompt(spec: Dict[str, Any]) -> str:
    spec = _stable_spec(spec)
    spec_str = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
    return (
        "You are an expert NoC physical designer.\n"
        "Given an architecture specification in JSON, output ONLY a JSON object:\n"
        "{\"switches\": {\"s_0\": {\"x\": int, \"y\": int}, ...}}\n"
        "Rules:\n"
        "- Output JSON only. No extra text.\n"
        "- Coordinates must be integers.\n"
        "- Keep switches inside floorplan and avoid blockages.\n"
        "\n"
        "-- Arch Specification --\n"
        f"{spec_str}\n"
        "\n"
        "-- Output (JSON only) --\n"
    )

def build_label(switches: Dict[str, Any]) -> str:
    return json.dumps({"switches": switches}, ensure_ascii=False, separators=(",", ":"))
