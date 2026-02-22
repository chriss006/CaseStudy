#!/usr/bin/env python3
"""NoC Architecture Validation Module"""
import json
from typing import Dict, List, Tuple

class ArchitectureValidator:
    """Validates NoC architecture constraints."""
    
    def __init__(self, spec: Dict, output: Dict):
        self.spec = spec
        self.output = output
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> Tuple[bool, Dict]:
        self.errors = []
        self.warnings = []
        
        self._validate_switch_placement()
        self._validate_path_elements()
        self._validate_route_connectivity()
        self._validate_no_cycles()
        
        is_valid = len(self.errors) == 0
        
        report = {
            "valid": is_valid,
            "errors": self.errors,
            "warnings": self.warnings,
            "checks": {
                "switch_placement": not self._has_error("switch_placement"),
                "path_elements": not self._has_error("path_elements"),
                "route_connectivity": not self._has_error("route_connectivity"),
                "no_cycles": not self._has_error("cycles")
            }
        }
        
        return is_valid, report
    
    def _has_error(self, error_type: str) -> bool:
        return any(error_type in e for e in self.errors)
    
    def _validate_switch_placement(self):
        floorplan = self.spec.get("floorplan_dim", [1000, 1000])
        max_x, max_y = floorplan[0], floorplan[1]
        blockages = self.spec.get("blockages", {})
        switches = self.output.get("switches", {})
        
        for switch_id, coords in switches.items():
            x, y = coords["x"], coords["y"]
            
            if not (0 <= x <= max_x and 0 <= y <= max_y):
                self.errors.append(
                    f"switch_placement: {switch_id} at ({x}, {y}) outside bounds ({max_x}, {max_y})"
                )
            
            for block_id, block in blockages.items():
                bx, by = block["x"], block["y"]
                bw, bh = block["width"], block["height"]
                
                if (bx <= x <= bx + bw) and (by <= y <= by + bh):
                    self.errors.append(
                        f"switch_placement: {switch_id} at ({x}, {y}) inside blockage {block_id}"
                    )
    
    def _validate_path_elements(self):
        inits = set(self.spec.get("inits", {}).keys())
        targets = set(self.spec.get("targets", {}).keys())
        switches = set(self.output.get("switches", {}).keys())
        
        all_valid_nodes = inits | targets | switches
        routing_paths = self.output.get("routing_paths", {})
        
        for route_id, path in routing_paths.items():
            if not isinstance(path, list):
                self.errors.append(f"path_elements: {route_id} path is not a list")
                continue
            
            for node in path:
                if node not in all_valid_nodes:
                    self.errors.append(
                        f"path_elements: {route_id} contains non-existent node '{node}'"
                    )
    
    def _validate_route_connectivity(self):
        connectivity = self.spec.get("connectivity", {})
        routing_paths = self.output.get("routing_paths", {})
        
        for route_id, (init, target) in connectivity.items():
            if route_id not in routing_paths:
                self.errors.append(
                    f"route_connectivity: Required route {route_id} ({init}->{target}) is missing"
                )
                continue
            
            path = routing_paths[route_id]
            
            if len(path) < 2:
                self.errors.append(
                    f"route_connectivity: {route_id} path too short"
                )
            elif path[0] != init:
                self.errors.append(
                    f"route_connectivity: {route_id} should start with {init}, got {path[0]}"
                )
            elif path[-1] != target:
                self.errors.append(
                    f"route_connectivity: {route_id} should end with {target}, got {path[-1]}"
                )
    
    def _validate_no_cycles(self):
        routing_paths = self.output.get("routing_paths", {})
        
        for route_id, path in routing_paths.items():
            if not isinstance(path, list) or len(path) < 2:
                continue
            
            seen_nodes = set()
            for node in path:
                if node in seen_nodes:
                    self.errors.append(
                        f"cycles: Route {route_id} contains loop - '{node}' appears multiple times"
                    )
                    break
                seen_nodes.add(node)

def validate_architecture(spec: Dict, output: Dict) -> Tuple[bool, Dict]:
    validator = ArchitectureValidator(spec, output)
    return validator.validate_all()
