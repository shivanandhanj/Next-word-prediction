import torch
from shared.models import LanguageModelLoRAWrapper
from shared.config import Config

def test_client_interactive(client_rank=4):
    # Initialize client model
    client = LanguageModelLoRAWrapper(r=client_rank)
    
    print(f"\nTesting Client with LoRA Rank {client_rank}")
    print("Type 'exit' to quit\n")
    
    while True:
        prompt = input("Enter your prompt: ")
        if prompt.lower() == 'exit':
            break
            
        # Get predictions
        predictions = client.predict_next_word(prompt)
        
        # Display results
        print("\nTop Predictions:")
        for word, prob in predictions:
            print(f"- {word} ({prob*100:.1f}%)")
        print()

if __name__ == "__main__":
    # Test different client configurations
    test_client_interactive(client_rank=2)  # Small client
    test_client_interactive(client_rank=8)  # Large client