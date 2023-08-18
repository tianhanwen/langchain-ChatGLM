
import sys
from typing import Any
from configs.model_config import llm_model_key_dict, LLM_KEY_MODEL
from models.aliyun_qwen import AliyunQwen

def loaderLLM(llm_model: str = None) -> Any:
    llm_model_info = llm_model_key_dict[LLM_KEY_MODEL]
    if llm_model:
        llm_model_info = llm_model_key_dict[llm_model]
        
    llm_mode_name = llm_model_info['name']
    modelInsLLM = None
    if llm_mode_name == "qwen-v1":
        modelInsLLM = AliyunQwen("qwen-v1")
    return modelInsLLM