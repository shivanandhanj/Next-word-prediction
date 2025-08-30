# -----------------------------
# Install required packages  ///Main code////
!pip install -q transformers datasets peft accelerate bitsandbytes

import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM, Trainer, TrainingArguments, DataCollatorForLanguageModeling
from peft import LoraConfig, get_peft_model
import os

# -----------------------------
# Disable WandB logging
os.environ["WANDB_MODE"] = "offline"

# -----------------------------
# Paths
reddit_path = "/content/reddit_dataset/train.jsonl"   # Reddit dataset
client_paths = {
    "chat1": "/content/human1.txt",
    "chat2": "/content/human2.txt"
}

# -----------------------------
# Tokenizer
model_name = "gpt2"
tokenizer = AutoTokenizer.from_pretrained(model_name)
tokenizer.pad_token = tokenizer.eos_token

# -----------------------------
# Load & tokenize dataset
def load_and_tokenize(dataset_type="reddit", path=None):
    if dataset_type == "reddit":
        dataset = load_dataset("json", data_files={"train": path})
        def preprocess(example):
            text = ""
            for turn in example["dialog"]:
                text += f"{turn['role']}: {turn['content']}\n"
            encoding = tokenizer(text, truncation=True, padding="max_length", max_length=256)
            return {"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"]}
        tokenized = dataset["train"].map(preprocess)
    else:  # human chat
        dataset = load_dataset("text", data_files={"train": path})
        def preprocess(example):
            encoding = tokenizer(example["text"], truncation=True, padding="max_length", max_length=256)
            return {"input_ids": encoding["input_ids"], "attention_mask": encoding["attention_mask"]}
        tokenized = dataset["train"].map(preprocess)
    return tokenized

# -----------------------------
# Initialize student model with LoRA
student_model = AutoModelForCausalLM.from_pretrained(model_name)
lora_config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=["c_attn"],
    lora_dropout=0.05
)
student_model = get_peft_model(student_model, lora_config)
student_model.to("cuda" if torch.cuda.is_available() else "cpu")

# Teacher model for KD
teacher_model = AutoModelForCausalLM.from_pretrained("gpt2-medium").eval()
teacher_model.to("cuda" if torch.cuda.is_available() else "cpu")

# -----------------------------
# Custom KD Trainer
class KDTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, **kwargs):  # <-- fix TypeError
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        outputs = model(**inputs)
        student_logits = outputs.logits

        with torch.no_grad():
            teacher_logits = teacher_model(**inputs).logits

        # KL Divergence loss (Knowledge Distillation)
        kd_loss = torch.nn.functional.kl_div(
            torch.nn.functional.log_softmax(student_logits / 2, dim=-1),
            torch.nn.functional.softmax(teacher_logits / 2, dim=-1),
            reduction="batchmean"
        ) * (2**2)

        # Optional: combine with standard cross-entropy
        ce_loss = torch.nn.functional.cross_entropy(
            student_logits.view(-1, student_logits.size(-1)),
            inputs["input_ids"].view(-1)
        )
        loss = 0.5 * ce_loss + 0.5 * kd_loss

        return (loss, outputs) if return_outputs else loss

# -----------------------------
# Training arguments
args = TrainingArguments(
    output_dir="/content/gpt2_lora_kd",
    per_device_train_batch_size=2,
    learning_rate=5e-5,
    num_train_epochs=2,
    logging_steps=50,
    save_strategy="epoch",
    fp16=True
)

data_collator = DataCollatorForLanguageModeling(tokenizer, mlm=False)

# -----------------------------
# Step 1: Pre-train on Reddit dataset
print("Pre-training on Reddit dataset...")
reddit_dataset = load_and_tokenize("reddit", reddit_path)

trainer = KDTrainer(
    model=student_model,
    args=args,
    train_dataset=reddit_dataset,
    tokenizer=tokenizer,
    data_collator=data_collator
)
trainer.train()

student_model.save_pretrained("/content/gpt2_lora_kd_reddit")
tokenizer.save_pretrained("/content/gpt2_lora_kd_reddit")
print("Reddit pre-training completed.\n")

# -----------------------------
# Step 2: Fine-tune on selected client
def fine_tune_client(client_name):
    print(f"Fine-tuning for {client_name}...")
    client_dataset = load_and_tokenize("human", client_paths[client_name])

    trainer = KDTrainer(
        model=student_model,
        args=args,
        train_dataset=client_dataset,
        tokenizer=tokenizer,
        data_collator=data_collator
    )
    trainer.train()

    save_path = f"/content/gpt2_lora_{client_name}"
    student_model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)
    print(f"Client {client_name} model saved at {save_path}\n")

    # Test: next-word prediction
    prompt = "Hi, how are you"
    inputs = tokenizer(prompt, return_tensors="pt").to(student_model.device)
    output = student_model.generate(**inputs, max_new_tokens=20)
    print("Sample output:", tokenizer.decode(output[0], skip_special_tokens=True))

# -----------------------------
# Example usage
fine_tune_client("chat1")
fine_tune_client("chat2")


from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned client model (example: chat1)
model_path = "/content/gpt2_lora_chat1"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Function to predict the next word
def predict_next_word(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Encode prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate 1 token only
    output = model.generate(
        **inputs,
        max_new_tokens=1,
        do_sample=True,       # set True for randomness, False for greedy
        top_k=50,             # sampling top k tokens
        top_p=0.95            # nucleus sampling
    )

    # Decode the new token
    next_word = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return next_word

# Example usage
while True:
    user_input = input("Human1: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    next_word = predict_next_word(user_input)
    print("Predicted next word:", next_word)



from transformers import AutoModelForCausalLM, AutoTokenizer
import torch

# Load the fine-tuned client model (chat2)
model_path = "/content/gpt2_lora_chat2"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForCausalLM.from_pretrained(model_path)
model.to("cuda" if torch.cuda.is_available() else "cpu")
model.eval()

# Function to predict the next word
def predict_next_word(prompt):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Encode the prompt
    inputs = tokenizer(prompt, return_tensors="pt").to(device)

    # Generate 1 token only
    output = model.generate(
        **inputs,
        max_new_tokens=1,   # predict only next word
        do_sample=True,     # randomness for more natural prediction
        top_k=50,
        top_p=0.95
    )

    # Decode the predicted token
    next_word = tokenizer.decode(output[0, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
    return next_word

# -----------------------------
# Interactive loop for Human2
print("Type a word or phrase (type 'exit' to quit):")
while True:
    user_input = input("Human2: ")
    if user_input.lower() in ["exit", "quit"]:
        break
    next_word = predict_next_word(user_input)
    print("Predicted next word:", next_word)
