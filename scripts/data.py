import pandas as pd
import json

# Load your 500 sentences
df = pd.read_csv("data/hindi_triples_500.csv")

training_data = []

for _, row in df.iterrows():
    # Make sure your CSV column names match here ('sentence', 'subject', etc.)
    input_text = f"extract triple: {row['sentence']}"
    target_text = f"({row['subject']}, {row['predicate']}, {row['object']})"
    
    training_data.append({
        "input": input_text,
        "output": target_text
    })

with open("data/training_data.json", "w", encoding="utf-8") as f:
    json.dump(training_data, f, ensure_ascii=False, indent=4)

print(f"Successfully processed {len(training_data)} sentences.")