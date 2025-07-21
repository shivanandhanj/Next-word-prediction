# Configuration constants
class Config:
    # Model settings
    BASE_MODEL_NAME = "distilgpt2"
    DEFAULT_SEQ_LENGTH = 64
    DEFAULT_BATCH_SIZE = 2
    
    # LoRA settings
    DEFAULT_RANKS = [2, 4, 8]  # Heterogeneous ranks for clients
    LORA_ALPHA = 8
    LORA_DROPOUT = 0.1
    TARGET_MODULES = ["c_attn", "c_proj"]  # Modules to apply LoRA to
    
    # Federated learning settings
    CLIENT_WEIGHTS = None  # Equal weights by default
    AGGREGATION_METHOD = "heLORA-pad"  # Options: "heLORA-pad", "heLORA-kd"
    
    # Training settings
    LEARNING_RATE = 5e-5
    NUM_ROUNDS = 3
    CLIENTS_PER_ROUND = 2
    
    # Knowledge distillation settings
    KD_TEMPERATURE = 2.0
    KD_BETA = 0.7  # Weight for CE loss vs KL loss