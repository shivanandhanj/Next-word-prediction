import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import LoraConfig, get_peft_model
from .config import Config

class LanguageModelLoRAWrapper:
    def __init__(self, model_name=Config.BASE_MODEL_NAME, r=4):
        self.model_name = model_name
        self.r = r
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load base model with memory-efficient settings
        self.base_model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float32,
            low_cpu_mem_usage=True
        )
       

        
        # Configure LoRA
        self.lora_config = LoraConfig(
            r=self.r,
            lora_alpha=Config.LORA_ALPHA,
            target_modules=Config.TARGET_MODULES,
            lora_dropout=Config.LORA_DROPOUT,
            bias="none",
            task_type="CAUSAL_LM",
        )
        
        # Create PEFT model
        self.model = get_peft_model(self.base_model, self.lora_config)
        self.model.enable_input_require_grads()
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
    
    def generate(self, prompt, max_length=50):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(
            input_ids=inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
            max_length=max_length,
            do_sample=True,
            temperature=0.7
        )
        return self.tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    def get_trainable_params(self):
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
    
    def predict_next_word(self, prompt, top_k=5):
        """Predict the next most likely words"""
        inputs = self.tokenizer(prompt, return_tensors="pt")
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits[0, -1]
            
        # Get top k predictions
        probs = torch.nn.functional.softmax(logits, dim=-1)
        top_probs, top_indices = torch.topk(probs, k=top_k)
        
        # Convert to human-readable words
        predictions = []
        for i in range(top_k):
            word = self.tokenizer.decode([top_indices[i]])
            predictions.append((word.strip(), top_probs[i].item()))
            
        return predictions