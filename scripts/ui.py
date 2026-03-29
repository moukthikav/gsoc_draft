import streamlit as st
import pandas as pd
import json

# --- Page Config ---
st.set_page_config(page_title="DBpedia Hindi HITL", layout="wide")
st.title("🇮🇳 Hindi Relational Triple Extractor")
st.caption("GSoC 2026: DBpedia Hindi Chapter Feedback Loop")

# --- Model Loading (Mocked for UI demo) ---
def mock_inference(text):
    # This is where your model.generate() logic goes
    return [
        {"subject": "हिमालय", "predicate": "स्थित", "object": "एशिया"},
        {"subject": "हिमालय", "predicate": "प्रकार", "object": "पर्वत-शृंखला"}
    ]

# --- Sidebar: Dataset Stats ---
with st.sidebar:
    st.header("Dataset Progress")
    st.metric("Validated Triples", "42")
    st.metric("Pending Review", "15")

# --- Main UI ---
input_text = st.text_area("Enter Hindi Wikipedia Sentence:", 
                          "हिमालय एशिया में स्थित एक प्राचीन पर्वत-शृंखला है।")

if st.button("Extract & Review"):
    results = mock_inference(input_text)
    st.session_state['current_triples'] = results

if 'current_triples' in st.session_state:
    st.subheader("Human-in-the-Loop Validation")
    
    # 1. Editable Table for Feedback
    df = pd.DataFrame(st.session_state['current_triples'])
    
    # Use data_editor to allow users to fix model mistakes
    edited_df = st.data_editor(
        df,
        column_config={
            "predicate": st.column_config.SelectboxColumn(
                "Ontology Mapping",
                help="Map to DBpedia Predicate",
                options=["dbo:location", "dbo:type", "dbo:partOf", "Other"],
                required=True,
            )
        },
        num_rows="dynamic"
    )

    # 2. Action Buttons
    col1, col2 = st.columns(2)
    with col1:
        if st.button("✅ Confirm & Save to Gold Set"):
            # Logic to append to your JSONL training file
            with open("data/gold_standard.jsonl", "a", encoding="utf-8") as f:
                for _, row in edited_df.iterrows():
                    f.write(json.dumps(row.to_dict(), ensure_ascii=False) + "\n")
            st.success("Triple saved for iterative fine-tuning!")
            
    with col2:
        if st.button("🗑️ Flag as Hallucination"):
            st.warning("Flagged. This sentence will be reviewed for training bias.")