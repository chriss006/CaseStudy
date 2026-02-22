#!/usr/bin/env python3
"""
NoC Architecture Generation using Fine-tuned LLM
Generates switch placements and routing paths from specifications.
"""
import json
import torch
import re
from pathlib import Path
from typing import Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from validate_architecture import validate_architecture


class NoCGenerator:
    """Generate NoC architectures using fine-tuned LLM."""
    
    def __init__(self, model_path: str, base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        """
        Initialize generator with trained model.
        
        Args:
            model_path: Path to fine-tuned model checkpoint (LoRA adapters)
            base_model: Base model name
        """
        print(f"Loading model from {model_path}...")
        
        # Quantization config (same as training)
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        # Load base model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Load LoRA adapters
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.eval()
        
        print("✅ Model loaded successfully")
    
    def create_prompt(self, spec: Dict) -> str:
        """
        Create prompt from specification.
        
        Args:
            spec: Architecture specification
        
        Returns:
            Formatted prompt string
        """
        # Use the same format as training (from format_noc.py)
        spec_str = json.dumps(spec, indent=2)
        
        prompt = f"""You are an expert NoC (Network-on-Chip) physical designer. Given a chip floorplan specification including initiators, targets, connectivity requirements, and blockage zones, generate optimal switch placements and routing paths.

Output ONLY valid JSON with this structure:
{{
  "switches": {{"s_0": {{"x": <int>, "y": <int>}}, ...}},
  "routing_paths": {{"r_0": ["<init>", "<switches>", ..., "<target>"], ...}}
}}

Constraints:
- Switches must be within floorplan bounds and avoid blockage zones
- All routes in connectivity must have valid paths
- Network must be deadlock-free (no cycles)
- Minimize total wirelength

Specification:
{spec_str}

Generated Architecture:"""
        
        return prompt
    
    def generate(
        self, 
        spec: Dict,
        max_new_tokens: int = 1024,
        temperature: float = 0.7,
        top_p: float = 0.9
    ) -> Tuple[Optional[Dict], str]:
        """
        Generate architecture from specification.
        
        Args:
            spec: Architecture specification
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature (lower = more deterministic)
            top_p: Nucleus sampling parameter
        
        Returns:
            (output_dict, raw_text) - Parsed output and raw generation
        """
        prompt = self.create_prompt(spec)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract just the generated part (after the prompt)
        generated_part = generated_text[len(prompt):].strip()
        
        # Parse JSON output
        output_dict = self._parse_output(generated_part)
        
        return output_dict, generated_part
    
    def _parse_output(self, text: str) -> Optional[Dict]:
        """
        Parse JSON from generated text.
        
        Args:
            text: Generated text
        
        Returns:
            Parsed dictionary or None if parsing fails
        """
        # Try to find JSON block
        # Look for pattern: { ... }
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parsing error: {e}")
                return None
        
        print("⚠️  No valid JSON found in output")
        return None
    
    def generate_and_validate(
        self,
        spec: Dict,
        max_new_tokens: int = 1024,
        temperature: float = 0.7
    ) -> Dict:
        """
        Generate and validate architecture.
        
        Args:
            spec: Architecture specification
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
        
        Returns:
            Results dictionary with output, validation, and metadata
        """
        output, raw_text = self.generate(spec, max_new_tokens, temperature)
        
        result = {
            "spec": spec,
            "output": output,
            "raw_generation": raw_text,
            "parsed_successfully": output is not None,
            "validation": None
        }
        
        # Validate if parsing succeeded
        if output is not None:
            is_valid, validation_report = validate_architecture(spec, output)
            result["validation"] = validation_report
            result["is_valid"] = is_valid
        else:
            result["is_valid"] = False
        
        return result


def generate_from_file(
    spec_file: str,
    model_path: str,
    output_file: Optional[str] = None,
    temperature: float = 0.7
) -> Dict:
    """
    Generate architecture from specification file.
    
    Args:
        spec_file: Path to JSON specification file
        model_path: Path to trained model
        output_file: Optional path to save results
        temperature: Sampling temperature
    
    Returns:
        Results dictionary
    """
    # Load specification
    with open(spec_file) as f:
        spec = json.load(f)
    
    # Create generator
    generator = NoCGenerator(model_path)
    
    # Generate and validate
    result = generator.generate_and_validate(spec, temperature=temperature)
    
    # Print summary
    print("\n" + "="*60)
    print("GENERATION SUMMARY")
    print("="*60)
    print(f"Parsed successfully: {result['parsed_successfully']}")
    
    if result['parsed_successfully']:
        print(f"Valid architecture: {result['is_valid']}")
        
        if result['validation']:
            print(f"Errors: {len(result['validation']['errors'])}")
            if result['validation']['errors']:
                print("\nFirst 3 errors:")
                for error in result['validation']['errors'][:3]:
                    print(f"  - {error}")
        
        if result['output']:
            print(f"\nGenerated switches: {len(result['output'].get('switches', {}))}")
            print(f"Generated routes: {len(result['output'].get('routing_paths', {}))}")
    else:
        print("❌ Failed to parse JSON output")
        print(f"\nRaw output (first 200 chars):\n{result['raw_generation'][:200]}")
    
    # Save if requested
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(result, f, indent=2)
        print(f"\n✅ Results saved to {output_file}")
    
    return result


# Command-line interface
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Generate NoC architecture from specification")
    parser.add_argument("spec_file", help="Path to specification JSON file")
    parser.add_argument("--model", required=True, help="Path to trained model checkpoint")
    parser.add_argument("--output", help="Path to save results JSON")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature (default: 0.7)")
    
    args = parser.parse_args()
    
    generate_from_file(
        args.spec_file,
        args.model,
        args.output,
        args.temperature
    )
