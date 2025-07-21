import torch
import numpy as np
from typing import List, Dict
from collections import defaultdict
from shared.models import LanguageModelLoRAWrapper
from shared.utils import pad_parameters_mem_efficient, safe_load_state_dict, log_memory_usage
from shared.config import Config

class FederatedServer:
    def __init__(self, global_model, client_weights=Config.CLIENT_WEIGHTS):
        self.global_model = global_model
        self.client_weights = client_weights or []
        self.device = torch.device("cpu")  # Force CPU usage
        self.global_model.to(self.device)
        self.global_model.eval()
    
    def aggregate(self, client_updates: List[Dict], client_ranks: List[int]):
        """
        HeLoRA-Pad aggregation with memory efficiency
        """
        log_memory_usage("start_aggregation")
        
        # Initialize aggregated state with zeros
        global_state = {
            k: torch.zeros_like(v, device=self.device)
            for k, v in self.global_model.state_dict().items()
            if 'lora' in k.lower()
        }
        
        total_weight = sum(
            (self.client_weights[i] if self.client_weights else 1) * client_ranks[i]
            for i in range(len(client_updates))
        )
        
        # Process one client at a time
        for i, update in enumerate(client_updates):
            log_memory_usage(f"client_{i}_start")
            
            client_weight = (self.client_weights[i] if self.client_weights else 1) * client_ranks[i]
            scale = client_weight / total_weight
            
            for key in global_state.keys():
                if key in update:
                    client_param = update[key].to(self.device)
                    
                    if client_param.shape != global_state[key].shape:
                        padded_param = pad_parameters_mem_efficient(
                            client_param, 
                            global_state[key].shape
                        )
                        global_state[key] += scale * padded_param
                    else:
                        global_state[key] += scale * client_param
                    
                    del client_param
                    if 'padded_param' in locals():
                        del padded_param
            
            log_memory_usage(f"client_{i}_end")
        
        # Update global model
        current_state = self.global_model.state_dict()
        for key in global_state:
            if key in current_state:
                current_state[key] = global_state[key]
        
        safe_load_state_dict(self.global_model, current_state)
        log_memory_usage("end_aggregation")
        return self.global_model

    def aggregate_with_knowledge_distillation(self, client_models: List[LanguageModelLoRAWrapper], public_dataset):
        """
        HeLoRA-KD implementation with memory efficiency
        """
        log_memory_usage("start_kd")
        
        # Small batch from public dataset
        sample_batch = next(iter(public_dataset))  
        inputs = sample_batch["input_ids"].to(self.device)
        attention_mask = sample_batch["attention_mask"].to(self.device)
        
        # Compute global logits once
        with torch.no_grad():
            global_logits = self.global_model(inputs, attention_mask=attention_mask).logits
        
        # Process each client
        for client_model in client_models:
            client_model.model.to(self.device)
            client_model.model.eval()
            
            with torch.no_grad():
                client_logits = client_model.model(inputs, attention_mask=attention_mask).logits
                
                # Rank-weighted consensus
                weight = client_model.rank / sum(m.rank for m in client_models)
                consensus = (1 - weight) * global_logits + weight * client_logits
            
            # Distill knowledge
            self._distill_to_global(consensus, sample_batch)
            
            # Clean up
            del client_logits
            client_model.model.to('cpu')
            log_memory_usage(f"after_client_{client_model.rank}")
        
        log_memory_usage("end_kd")
        return self.global_model

    def _distill_to_global(self, consensus, batch):
        """Memory-efficient distillation step"""
        self.global_model.train()
        
        inputs = batch["input_ids"].to(self.device)
        attention_mask = batch["attention_mask"].to(self.device)
        labels = batch["labels"].to(self.device)
        
        optimizer = torch.optim.AdamW(
            filter(lambda p: p.requires_grad, self.global_model.parameters()),
            lr=Config.LEARNING_RATE
        )
        
        # Forward pass
        outputs = self.global_model(
            input_ids=inputs,
            attention_mask=attention_mask,
            labels=labels
        )
        
        # Calculate losses
        ce_loss = outputs.loss
        
        # KL divergence loss
        soft_targets = torch.nn.functional.softmax(consensus / Config.KD_TEMPERATURE, dim=-1)
        soft_outputs = torch.nn.functional.log_softmax(outputs.logits / Config.KD_TEMPERATURE, dim=-1)
        kld_loss = torch.nn.KLDivLoss(reduction='batchmean')(soft_outputs, soft_targets)
        
        # Combined loss
        loss = Config.KD_BETA * ce_loss + (1 - Config.KD_BETA) * kld_loss
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Clean up
        del inputs, attention_mask, labels, outputs