import torch
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
from typing import List, Dict, Optional
import time
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLLMBackend:
    """vLLM Backend Wrapper"""
    
    def __init__(
        self,
        model_path: str,
        max_loras: int = 8,
        gpu_memory_utilization: float = 0.85
    ):
        logger.info(f"Initializing vLLM with {model_path}...")
        
        self.engine = LLM(
            model=model_path,
            enable_lora=True,
            max_loras=max_loras,
            max_lora_rank=64,
            dtype="float16",
            gpu_memory_utilization=gpu_memory_utilization,
            trust_remote_code=True
        )
        
        logger.info("vLLM initialized")
        
        self.generation_count = 0
        self.total_tokens = 0
    
    def generate(
        self,
        prompts: List[str],
        lora_path: Optional[str] = None,
        lora_id: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 256,
        **kwargs
    ) -> List[str]:
        """Generate text"""
        
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            **kwargs
        )
        
        lora_request = None
        if lora_path:
            lora_request = LoRARequest(
                lora_name=f"lora_{lora_id}",
                lora_int_id=lora_id,
                lora_local_path=lora_path
            )
        
        outputs = self.engine.generate(
            prompts,
            sampling_params,
            lora_request=lora_request
        )
        
        self.generation_count += 1
        
        return [output.outputs[0].text for output in outputs]
    
    def health_check(self):
        """Quick health check"""
        try:
            self.generate(["Test"], max_tokens=5)
            return True
        except:
            return False
