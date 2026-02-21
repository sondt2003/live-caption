import os
import torch
from ..base import BaseTranslator
from loguru import logger

class LocalLLMTranslator(BaseTranslator):
    def __init__(self, model_name=None):
        self.model_name = model_name or os.getenv('MODEL_NAME', 'qwen/Qwen1.5-4B-Chat')
        if 'Qwen' not in self.model_name:
            self.model_name = 'qwen/Qwen1.5-4B-Chat'
        self.model = None
        self.tokenizer = None

    def _init_model(self):
        if self.model is not None:
            return
        
        from transformers import AutoModelForCausalLM, AutoTokenizer
        model_path = os.path.join('models/LLM', os.path.basename(self.model_name))
        pretrained_path = self.model_name if not os.path.isdir(model_path) else model_path
        
        logger.info(f"Loading Local LLM model: {pretrained_path}...")
        self.model = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            torch_dtype="auto",
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_path)
        logger.info('Finish Load model')

    def translate(self, messages: list, json_mode: bool = True) -> str:
        self._init_model()
        
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )
        device = self.model.device
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)

        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]

        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response
