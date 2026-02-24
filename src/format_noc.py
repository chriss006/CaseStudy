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

def build_prompt(spec: dict) -> str:
    spec_str = json.dumps(spec, ensure_ascii=False, separators=(",", ":"))
    return (
        "Return ONLY valid JSON. No extra text.\n"
        "Schema:\n"
        "{\"switches\":{\"s_0\":{\"x\":0,\"y\":0},\"s_1\":{\"x\":0,\"y\":0}}}\n"
        "Use keys s_0, s_1, ... and integer x,y.\n"
        "Spec:\n"
        f"{spec_str}\n"
    )

def build_label(switches: Dict[str, Any]) -> str:
    return json.dumps({"switches": switches}, ensure_ascii=False, separators=(",", ":"))
