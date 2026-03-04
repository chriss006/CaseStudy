#!/usr/bin/env python3
"""NoC Architecture Generation using Fine-tuned LLM"""
import json
import torch
import re
from typing import Dict, Optional, Tuple
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from peft import PeftModel
from validate_architecture import validate_architecture

class NoCGenerator:
    def __init__(self, model_path: str, base_model: str = "mistralai/Mistral-7B-Instruct-v0.2"):
        print(f"Loading model from {model_path}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.float16
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        base = AutoModelForCausalLM.from_pretrained(
            base_model,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        self.model = PeftModel.from_pretrained(base, model_path)
        self.model.eval()
        
        print("✅ Model loaded successfully")
    
    def create_prompt(self, spec: Dict) -> str:
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
    
    def generate(self, spec: Dict, max_new_tokens: int = 1024, temperature: float = 0.7, top_p: float = 0.9) -> Tuple[Optional[Dict], str]:
        prompt = self.create_prompt(spec)
        
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.model.device)
        
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        generated_part = generated_text[len(prompt):].strip()
        output_dict = self._parse_output(generated_part)
        
        return output_dict, generated_part
    
    def _parse_output(self, text: str) -> Optional[Dict]:
        json_match = re.search(r'\{.*\}', text, re.DOTALL)
        
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except json.JSONDecodeError as e:
                print(f"⚠️  JSON parsing error: {e}")
                return None
        
        print("⚠️  No valid JSON found in output")
        return None
    
    def generate_and_validate(self, spec: Dict, max_new_tokens: int = 1024, temperature: float = 0.7) -> Dict:
        output, raw_text = self.generate(spec, max_new_tokens, temperature)
        
        result = {
            "spec": spec,
            "output": output,
            "raw_generation": raw_text,
            "parsed_successfully": output is not None,
            "validation": None
        }
        
        if output is not None:
            is_valid, validation_report = validate_architecture(spec, output)
            result["validation"] = validation_report
            result["is_valid"] = is_valid
        else:
            result["is_valid"] = False
        
        return result
