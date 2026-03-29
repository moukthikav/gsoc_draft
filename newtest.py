import torch
from transformers import MT5Tokenizer, MT5ForConditionalGeneration

model_path = "models/triple_model"
tokenizer = MT5Tokenizer.from_pretrained(model_path)
model = MT5ForConditionalGeneration.from_pretrained(model_path)

def extract_triple(sentence):
    input_text = f"Extract triple: {sentence}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=64, truncation=True)

    # Use Beam Search for better quality
    outputs = model.generate(
        input_ids=inputs["input_ids"],
        max_length=48,
        num_beams=4, 
        early_stopping=True
    )

    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Test Example
test_sentence = "चारमीनार हैदराबाद में स्थित है।"
prediction = extract_triple(test_sentence)

print("-" * 30)
print(f"Input: {test_sentence}")
print(f"Output: {prediction}")
print("-" * 30)