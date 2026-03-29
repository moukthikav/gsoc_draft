import json
import torch
import os
from datasets import Dataset
from transformers import MT5Tokenizer, MT5ForConditionalGeneration, TrainingArguments, Trainer

# 1. Load data
data_path = "data/training_data.json"
if not os.path.exists(data_path):
    print(f"Error: {data_path} not found. Run trainingdata.py first.")
    exit()

with open(data_path, "r", encoding="utf-8") as f:
    data = json.load(f)

dataset = Dataset.from_list(data)

# 2. Load model & tokenizer
model_name = "google/mt5-small"
tokenizer = MT5Tokenizer.from_pretrained(model_name)
model = MT5ForConditionalGeneration.from_pretrained(model_name)

# 3. Optimized Preprocessing
def preprocess(example):
    model_inputs = tokenizer(
        example["input"],
        max_length=64,       # Optimized for CPU speed
        padding="max_length",
        truncation=True
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            example["output"],
            max_length=48, 
            padding="max_length",
            truncation=True
        )

    # Replace pad_token_id with -100 so it's ignored in loss calculation
    model_inputs["labels"] = [(l if l != tokenizer.pad_token_id else -100) for l in labels["input_ids"]]
    return model_inputs

tokenized_dataset = dataset.map(preprocess, remove_columns=dataset.column_names)

# 4. Training Arguments
# Reduced to 10 epochs because your loss was already 0.01 at epoch 10
training_args = TrainingArguments(
    output_dir="models/triple_model",
    learning_rate=2e-4,            # Slightly higher LR for bigger batches
    per_device_train_batch_size=16, # Doubled for speed
    num_train_epochs=8,             # Cut from 15 to 8
    weight_decay=0.01,
    save_strategy="no",            
    logging_steps=10,
    dataloader_num_workers=0,      
    report_to="none"
)

# 5. Initialize Trainer
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_dataset,
)

print("Starting Optimized CPU Training (Final Run)...")
trainer.train()

# 6. THE FIX: Handle Non-Contiguous Tensors before saving
print("Finalizing model memory for saving...")
for param in model.parameters():
    if not param.is_contiguous():
        param.data = param.data.contiguous()

# 7. Save Model and Tokenizer
save_directory = "models/triple_model"
os.makedirs(save_directory, exist_ok=True)

model.save_pretrained(save_directory, safe_serialization=True)
tokenizer.save_pretrained(save_directory)

print(f"✅ SUCCESS! Model saved to {save_directory}")