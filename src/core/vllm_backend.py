
from vllm import LLM, SamplingParams
from vllm.lora.request import LoRARequest
import torch
from typing import List, Dict, Optional
import logging
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class VLLMBackend:
    """vLLM Backend Wrapper - Colab Compatible"""
    
    def __init__(
        self,
        model_path: str,
        max_loras: int = 8,
        max_lora_rank: int = 64,
        dtype: str = "float16",
        gpu_memory_utilization: float = 0.75,
        max_model_len: int = 2048
    ):
        """Initialize vLLM backend for Colab"""
        
        self.model_path = model_path
        self.max_loras = max_loras
        
        logger.info(f"Initializing vLLM backend...")
        logger.info(f"  Model: {model_path}")
        logger.info(f"  Max LoRAs: {max_loras}")
        logger.info(f"  Max Length: {max_model_len}")
        
        try:
            # Simple configuration that works
            self.engine = LLM(
                model=model_path,
                
                # LoRA settings
                enable_lora=True,
                max_loras=max_loras,
                max_lora_rank=max_lora_rank,
                
                # Model settings
                dtype=dtype,
                max_model_len=max_model_len,
                
                # Resource settings
                gpu_memory_utilization=gpu_memory_utilization,
                
                # Compatibility
                trust_remote_code=True,
                enforce_eager=True,  # Most stable for Colab
            )
            
            logger.info(" vLLM engine initialized successfully")
            
        except Exception as e:
            logger.error(f" Failed to initialize vLLM: {e}")
            raise
        
        # Track metrics
        self.generation_count = 0
        self.total_tokens_generated = 0
        
    def generate(
        self,
        prompts: List[str],
        lora_path: Optional[str] = None,
        lora_id: int = 1,
        temperature: float = 0.7,
        max_tokens: int = 256,
        top_p: float = 0.9,
        stop: Optional[List[str]] = None,
        **kwargs
    ) -> List[str]:
        """Generate text using vLLM"""
        
        # Create sampling parameters
        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            stop=stop if stop else [],
            **kwargs
        )
        
        # Create LoRA request if specified
        lora_request = None
        if lora_path:
            lora_request = LoRARequest(
                lora_name=f"lora_{lora_id}",
                lora_int_id=lora_id,
                lora_local_path=lora_path
            )
            logger.debug(f"Using LoRA: {lora_path}")
        
        # Generate
        start_time = time.time()
        
        outputs = self.engine.generate(
            prompts,
            sampling_params,
            lora_request=lora_request
        )
        
        # Extract text
        generated_texts = [output.outputs[0].text for output in outputs]
        
        # Update metrics
        self.generation_count += 1
        generation_time = time.time() - start_time
        
        logger.debug(f"Generated in {generation_time:.2f}s")
        
        return generated_texts
    
    def get_stats(self) -> Dict:
        """Get backend statistics"""
        return {
            "model": self.model_path,
            "generation_count": self.generation_count,
            "max_loras": self.max_loras
        }
    
    def health_check(self) -> bool:
        """Quick health check"""
        try:
            self.generate(["Test"], max_tokens=5, temperature=0)
            return True
        except:
            return False
