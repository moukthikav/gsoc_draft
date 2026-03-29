import pandas as pd
from transformers import MT5ForConditionalGeneration, MT5Tokenizer
import torch
from tqdm import tqdm
import os

def load_model(model_path="models/triple_model"):
    if not os.path.exists(model_path):
        print(f"❌ Error: Model folder '{model_path}' not found!")
        return None, None
        
    print("Loading your trained Hindi MT5 model...")
    tokenizer = MT5Tokenizer.from_pretrained(model_path)
    model = MT5ForConditionalGeneration.from_pretrained(model_path)
    return model, tokenizer

def generate_triple(text, model, tokenizer):
    input_text = f"extract triple: {text}"
    inputs = tokenizer(input_text, return_tensors="pt", padding=True)
    
    outputs = model.generate(
        inputs["input_ids"], 
        max_length=64, 
        num_beams=4, 
        early_stopping=True
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def run_full_evaluation(csv_path="data/hindi_triples_500.csv"):
    model, tokenizer = load_model()
    if model is None: return

    df = pd.read_csv(csv_path)
    col_names = df.columns.tolist()

    # --- SPO Glue Logic ---
    # Since your CSV has separate columns, we combine them to match model output format
    if 'subject' in col_names and 'predicate' in col_names and 'object' in col_names:
        print("🔗 Formatting Triple components (S, P, O) for evaluation...")
        ground_truths = df.apply(
            lambda row: f"({str(row['subject']).strip()}, {str(row['predicate']).strip()}, {str(row['object']).strip()})", 
            axis=1
        ).tolist()
    else:
        print(f"❌ Error: Required columns not found. Found: {col_names}")
        return

    sentences = df['sentence'].tolist()
    predictions = []
    
    print(f"🚀 Running inference on {len(sentences)} sentences...")
    for sentence in tqdm(sentences):
        pred = generate_triple(sentence, model, tokenizer)
        predictions.append(pred)
    
    # --- Metric Calculation ---
    tp, exact_matches = 0, 0
    
    for p, g in zip(predictions, ground_truths):
        # Clean: lowercase, remove brackets, and strip whitespace
        p_c = p.strip().replace("(", "").replace(")", "").lower()
        g_c = g.strip().replace("(", "").replace(")", "").lower()
        
        if p_c == g_c:
            tp += 1
            exact_matches += 1

    # Precision/Recall/F1 logic for 1-to-1 mapping
    total = len(ground_truths)
    precision = tp / total if total > 0 else 0
    recall = tp / total if total > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    em_score = (exact_matches / total) * 100

    print("\n" + "="*40)
    print("📢 FINAL GSOC EVALUATION REPORT")
    print("="*40)
    print(f"Total Samples:    {total}")
    print(f"Exact Match:      {em_score:.2f}%")
    print(f"F1-Score:         {f1:.2f}")
    print(f"Precision:        {precision:.2f}")
    print(f"Recall:           {recall:.2f}")
    print("="*40)
    
    # Save a comparison file for your mentor
    results_df = pd.DataFrame({
        'Sentence': sentences,
        'Original Triple': ground_truths,
        'Model Predicted': predictions
    })
    results_df.to_csv("data/evaluation_results.csv", index=False, encoding='utf-8-sig')
    print("✅ Results saved to: data/evaluation_results.csv")

if __name__ == "__main__":
    run_full_evaluation()