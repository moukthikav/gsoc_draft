import json
from transformers import pipeline

# Load model
generator = pipeline(
    "text2text-generation",
    model="google/mt5-small",
    tokenizer="google/mt5-small"
)

# Load sentences
with open("data/wiki_sentences.json", "r", encoding="utf-8") as f:
    data = json.load(f)

results = []

for item in data:
    sentence = item["sentence"]

    prompt = f"Extract subject, relation, object from this Hindi sentence:\n{sentence}"

    output = generator(prompt, max_length=64, do_sample=False)

    triple = output[0]["generated_text"]

    results.append({
        "sentence": sentence,
        "prediction": triple
    })

# Save predictions
with open("results/baseline_predictions.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False, indent=2)

print("Baseline extraction complete!")
