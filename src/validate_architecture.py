#!/usr/bin/env python3
"""
NoC Architecture Validation Module
Validates that generated architectures meet hard constraints.
"""
import json
from typing import Dict, List, Tuple, Set


class ArchitectureValidator:
    """Validates NoC architecture constraints."""
    
    def __init__(self, spec: Dict, output: Dict):
        """
        Args:
            spec: Architecture specification (inits, targets, connectivity, floorplan, blockages)
            output: Generated output (switches, routing_paths)
        """
        self.spec = spec
        self.output = output
        self.errors = []
        self.warnings = []
    
    def validate_all(self) -> Tuple[bool, Dict]:
        """
        Run all validations.
        
        Returns:
            (is_valid, report_dict)
        """
        self.errors = []
        self.warnings = []
        
        # Run all validation checks
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
        """Check if specific error type exists."""
        return any(error_type in e for e in self.errors)
    
    def _validate_switch_placement(self):
        """Check switches are within bounds and not in blockages."""
        floorplan = self.spec.get("floorplan_dim", [1000, 1000])
        max_x, max_y = floorplan[0], floorplan[1]
        blockages = self.spec.get("blockages", {})
        switches = self.output.get("switches", {})
        
        for switch_id, coords in switches.items():
            x, y = coords["x"], coords["y"]
            
            # Check bounds
            if not (0 <= x <= max_x and 0 <= y <= max_y):
                self.errors.append(
                    f"switch_placement: {switch_id} at ({x}, {y}) outside floorplan bounds ({max_x}, {max_y})"
                )
            
            # Check blockages (simple rectangle collision)
            for block_id, block in blockages.items():
                bx, by = block["x"], block["y"]
                bw, bh = block["width"], block["height"]
                
                if (bx <= x <= bx + bw) and (by <= y <= by + bh):
                    self.errors.append(
                        f"switch_placement: {switch_id} at ({x}, {y}) inside blockage {block_id}"
                    )
    
    def _validate_path_elements(self):
        """Check all elements in routing paths exist."""
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
        """Check every required route has a valid path."""
        connectivity = self.spec.get("connectivity", {})
        routing_paths = self.output.get("routing_paths", {})
        
        for route_id, (init, target) in connectivity.items():
            if route_id not in routing_paths:
                self.errors.append(
                    f"route_connectivity: Required route {route_id} ({init}->{target}) is missing"
                )
                continue
            
            path = routing_paths[route_id]
            
            # Check path starts with initiator and ends with target
            if len(path) < 2:
                self.errors.append(
                    f"route_connectivity: {route_id} path too short (needs at least init and target)"
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
        """Check each individual routing path has no cycles (no loops within a single path)."""
        # For NoC architectures, we check that individual paths don't loop back on themselves
        # Multiple paths CAN share switches - that's normal and expected
        routing_paths = self.output.get("routing_paths", {})
        
        for route_id, path in routing_paths.items():
            if not isinstance(path, list) or len(path) < 2:
                continue
            
            # Check if the same node appears twice in a single path (self-loop)
            seen_nodes = set()
            for node in path:
                if node in seen_nodes:
                    self.errors.append(
                        f"cycles: Route {route_id} contains a loop - node '{node}' appears multiple times"
                    )
                    break
                seen_nodes.add(node)


def validate_architecture(spec: Dict, output: Dict) -> Tuple[bool, Dict]:
    """
    Convenience function to validate an architecture.
    
    Args:
        spec: Architecture specification
        output: Generated output
    
    Returns:
        (is_valid, report)
    """
    validator = ArchitectureValidator(spec, output)
    return validator.validate_all()


# Command-line usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 3:
        print("Usage: python validate_architecture.py <spec.json> <output.json>")
        sys.exit(1)
    
    with open(sys.argv[1]) as f:
        spec = json.load(f)
    
    with open(sys.argv[2]) as f:
        output = json.load(f)
    
    is_valid, report = validate_architecture(spec, output)
    
    print(json.dumps(report, indent=2))
    
    if is_valid:
        print("\n✅ Architecture is VALID")
        sys.exit(0)
    else:
        print(f"\n❌ Architecture is INVALID ({len(report['errors'])} errors)")
        sys.exit(1)
