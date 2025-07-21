import os
import numpy as np
import torch
from datasets import load_dataset
from shared.models import LanguageModelLoRAWrapper
from shared.config import Config
from server.server import FederatedServer
from shared.utils import log_memory_usage

def create_client_datasets():
    """Create client datasets from text files in clients/ directory"""
    client_datasets = []
    client_files = sorted([f for f in os.listdir("clients") if f.endswith(".txt")])
    
    for client_file in client_files:
        dataset = load_dataset("text", data_files={"train": [f"clients/{client_file}"]})["train"]
        client_datasets.append(dataset)
    
    return client_datasets

def tokenize_public_dataset():
    """Load and tokenize public dataset for knowledge distillation"""
    public_data = load_dataset("text", data_files={"train": ["public_data/public_text.txt"]})["train"]
    
    # Use the tokenizer from the first client model
    temp_model = LanguageModelLoRAWrapper()
    tokenizer = temp_model.tokenizer
    
    def tokenize_fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=Config.DEFAULT_SEQ_LENGTH,
            padding="max_length",
            return_tensors="pt"
        )
    
    tokenized = public_data.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"]
    )
    tokenized.set_format("torch")
    return tokenized

def main():
    # 1. Load datasets
    client_datasets = create_client_datasets()
    public_dataset = tokenize_public_dataset()
    
    # 2. Create clients with different ranks
    clients = []
    for i, (dataset, rank) in enumerate(zip(client_datasets, Config.DEFAULT_RANKS)):
        model_wrapper = LanguageModelLoRAWrapper(r=rank)
        client = {
            "model": model_wrapper,
            "dataset": dataset,
            "rank": rank,
            "id": i
        }
        clients.append(client)
        print(f"Created Client {i} with rank {rank} and {len(dataset)} samples")
    
    # 3. Initialize server with largest rank
    print("\nInitializing server...")
    server_model = LanguageModelLoRAWrapper(r=max(Config.DEFAULT_RANKS))
    server = FederatedServer(server_model.model)
    
    # 4. Training loop
    for round in range(Config.NUM_ROUNDS):
        print(f"\n=== Round {round + 1}/{Config.NUM_ROUNDS} ===")
        
        # Select clients for this round
        selected_indices = np.random.choice(
            len(clients), 
            size=min(Config.CLIENTS_PER_ROUND, len(clients)), 
            replace=False
        )
        selected_clients = [clients[i] for i in selected_indices]
        
        # Local training
        client_updates = []
        client_ranks = []
        for client in selected_clients:
            print(f"\nClient {client['id']} (Rank {client['rank']}) training...")
            
            # Tokenize client data
            def tokenize_fn(examples):
                return client["model"].tokenizer(
                    examples["text"],
                    truncation=True,
                    max_length=Config.DEFAULT_SEQ_LENGTH,
                    padding="max_length",
                    return_tensors="pt"
                )
            
            tokenized = client["dataset"].map(
                tokenize_fn,
                batched=True,
                remove_columns=["text"]
            )
            tokenized.set_format("torch")
            
            # Train for 1 epoch
            train_loader = torch.utils.data.DataLoader(
                tokenized,
                batch_size=Config.DEFAULT_BATCH_SIZE,
                shuffle=True
            )
            
            optimizer = torch.optim.AdamW(
                client["model"].model.parameters(),
                lr=Config.LEARNING_RATE
            )
            
            client["model"].model.train()
            for batch in train_loader:
                batch = {k: v.to(client["model"].model.device) for k, v in batch.items()}
                if "labels" not in batch:
                    batch["labels"] = batch["input_ids"]  # Simple fix for causal LM

                outputs = client["model"].model(**batch)
                loss = outputs.loss
                loss.backward()
                optimizer.step()
                optimizer.zero_grad()
            
            # Get LoRA parameters
            state_dict = client["model"].model.state_dict()
            lora_params = {k: v for k, v in state_dict.items() if 'lora' in k.lower()}
            client_updates.append(lora_params)
            client_ranks.append(client["rank"])
            
            # Show client-specific generation
            prompt = "Machine learning" if client["id"] == 0 else "The government"
            print(f"  Client {client['id']} generation:", 
                  client["model"].generate(prompt, max_length=20))
        
        # Server aggregation
        if Config.AGGREGATION_METHOD == "heLORA-pad":
            print("\nServer aggregating with HeLoRA-Pad...")
            server.aggregate(client_updates, client_ranks)
        else:
            print("\nServer aggregating with HeLoRA-KD...")
            client_models = [c["model"] for c in selected_clients]
            server.aggregate_with_knowledge_distillation(client_models, public_dataset)
        
        # Distribute updated model
        print("\nDistributing updated model...")
        global_state = server.global_model.state_dict()
        for client in selected_clients:
            client_state = client["model"].model.state_dict()
            
            for key in client_state:
                if 'lora' in key.lower():
                    if global_state[key].shape == client_state[key].shape:
                        client_state[key] = global_state[key]
                    else:
                        client_state[key] = global_state[key][:client_state[key].shape[0], 
                                              :client_state[key].shape[1]]
            
            client["model"].model.load_state_dict(client_state)
    
    # 5. Final evaluation
    print("\n=== Final Evaluation ===")
    test_prompts = [
        "Machine learning",
        "The government announced",
        "The weather forecast predicts",
        "In recent scientific research"
    ]
    
    for prompt in test_prompts:
        print(f"\nPrompt: '{prompt}'")
        print("Server response:", server_model.generate(prompt, max_length=30))
    
    # Show client-specific generations
    print("\n=== Client-Specific Generations ===")
    for client in clients:
        prompt = "Machine learning" if client["id"] == 0 else "The government"
        print(f"\nClient {client['id']} (Rank {client['rank']}) - Prompt: '{prompt}'")
        print("Response:", client["model"].generate(prompt, max_length=30))

if __name__ == "__main__":
    main()