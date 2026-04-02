"""
NEISS Vitamin and Supplement ED Visit Analysis (Ollama + Statistical Patterns)

This script analyzes NEISS data to identify trends in emergency department visits related to vitamins and supplements. 

script Key Features:
1. scriptGround Truth Comparisonscript: Labels cases based on product codes (1927, 1931, 1932).
2. scriptStatistical Word Analysisscript: Automatically identifies words that distinguish supplements from non-supplements.
3. scriptFew-Shot Ollama Classificationscript: Uses the discovered words to guide a Llama model via Ollama for zero-manual-effort classification.
4. scriptMetric Validationscript: Calculates accuracy and recall targeting >80% recall.
"""


#%%
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json, re
import ollama
import os
from tqdm.auto import tqdm

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from collections import Counter
from src.load_data import load_and_preprocess_data

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)



def get_word_freq(text_series):
    """
    Get word frequencies from a text series.
    """
    all_text = ' '.join(text_series.dropna().astype(str)).upper()
    words = re.findall(r'\b[A-Z]{3,}\b', all_text)  # correct word boundary
    counter = Counter(words)
    total = sum(counter.values())
    freqs = {w: c / total for w, c in counter.items()} if total > 0 else {}
    return freqs, counter



def analyze_word_frequencies(df):
    """
    Analyze word frequencies to identify discriminative words.
    """
    threshold = 1000 # max unique values to consider categorical
    cat_cols = []
    for col in df.columns:
        if (
            df[col].dtype == 'object' or 
            df[col].dtype.name == 'category' or
            (df[col].dtype in ['int64','float64'] and df[col].nunique() < threshold)
        ):
            cat_cols.append(col)

    print("Categorical columns detected:")
    print(cat_cols)

    VITAMIN_CODES = [1927, 1931, 1932, '1927', '1931', '1932']
    # Ground Truth Definition
    # check across all three product columns
    df['Ground_Truth'] = (
        df[['Product_1','Product_2','Product_3']]
        .isin(VITAMIN_CODES)
        .any(axis=1)
        .astype(int)
    )

    print("Analyzing word distributions...")
    freq1, counts1 = get_word_freq(df[df['Ground_Truth'] == 1]['Narrative'])
    freq0, counts0 = get_word_freq(df[df['Ground_Truth'] == 0]['Narrative'])
    print("Vitamin class total tokens:", sum(counts1.values()))
    print("Vitamin class max token count:", max(counts1.values()) if counts1 else 0)

    # discriminative score (smoothed ratio)
    min_count = 10
    alpha = 1

    scores = []
    for word, c1 in counts1.items():
        if c1 >= min_count:
            c0 = counts0.get(word, 0)
            score = (c1 + alpha) / (c0 + alpha)
            scores.append((word, c1, c0, score))

    scores.sort(key=lambda x: x[3], reverse=True)
    top_supplement_words = [w for w, _, _, _ in scores[:20]]
    print("Top words indicating a Supplement exposure:")
    print(top_supplement_words)



MODEL_NAME = "llama3.1:8b"

SYSTEM_MSG = """
You are a strict binary classifier for emergency department (ED) narratives.

OUTPUT FORMAT (must be valid JSON on a single line):
{"reason": "short reason", "label": 0 or 1}

TASK:
Classify whether the narrative involves exposure/ingestion/overdose/adverse reaction to a VITAMIN or DIETARY SUPPLEMENT.

LABEL DEFINITIONS:
- label=1 ONLY if the narrative clearly involves a vitamin or dietary supplement AND it does NOT contain IRON.
  Examples (label=1): multivitamin WITHOUT iron, vitamin D, vitamin C gummies, melatonin gummies, herbal supplements, creatine, protein supplements.
- label=0 for EVERYTHING ELSE, including:
  - ANY product/formulation that contains IRON (including "iron", "Fe", "ferrous", "ferric", "ferro-", "prenatal with iron", "multivitamin with iron", "iron + vitamin C", "iron gummies", etc.)
  - Prescription (e.g., fexofenadine, Tylenol, cough medicine, antibiotics, “tabs” with no supplement mention)
  - Household chemicals/toxins (e.g., ammonia, detergent, bleach, windshield fluid)
  - Unknown ingestion where the substance is not clearly a vitamin/supplement

IRON EXCLUSION RULE (highest priority):
- If the narrative mentions IRON or an iron formulation (e.g., "iron", "Fe", "ferrous sulfate", "ferrous", "prenatal with iron", "iron gummies", "iron + vit C"),
  you MUST output label=0 even if it is a vitamin gummy or multivitamin.

GENERAL RULES:
- Do not guess. If supplement/vitamin is not explicit, choose 0.
- Ingestion alone does NOT imply label=1.
- Reason must cite the key phrase that triggered your decision.
""".strip()

# Few-shot examples WITH reasons (include hard negatives for iron exception)
FEW_SHOTS = [
    # Positives (supplements/vitamins WITHOUT iron)
    ("2 YOM INGESTED MULTIVITAMIN (NO IRON) GUMMIES.", 1, "Mentions 'multivitamin (no iron) gummies' which are supplements without iron."),
    ("CHILD TOOK SEVERAL VITAMIN D PILLS.", 1, "Explicit 'vitamin D' ingestion (no iron mentioned)."),
    ("ADULT TOOK MELATONIN GUMMIES; DIZZY.", 1, "Mentions 'melatonin gummies' (supplement; no iron)."),
    ("2YOF ATE VITAMIN C GUMMIES.", 1, "Mentions 'vitamin C gummies' (vitamin supplement; no iron)."),

    # Iron exception hard negatives (MUST be 0)
    ("2YOF ATE ADULT IRON + VIT C 18MG GUMMIES.", 0, "Mentions 'IRON + VIT C' / 'iron gummies' -> iron-containing formulation is excluded."),
    ("PT TOOK TOO MANY IRON SUPPLEMENTS.", 0, "Mentions 'iron supplements' -> iron-containing formulation is excluded."),
    ("PRENATAL VITAMINS WITH IRON INGESTION.", 0, "Mentions 'with iron' -> iron-containing formulation is excluded."),
    ("PT INGESTED FERROUS SULFATE TABLETS.", 0, "Mentions 'ferrous sulfate' -> iron formulation is excluded."),

    # Other hard negatives (non-supplement)
    ("10MOM GOT INTO A BOTTLE OF FEXOFENADINE TABLETS.", 0, "Fexofenadine is an OTC medication, not a supplement."),
    ("43YOF ALLERGIC RXN TO COUGH MED.", 0, "Cough medicine is medication, not a supplement."),
    ("PT DRANK 4-8 OZ OF WINDSHIELD FLUID.", 0, "Windshield fluid is a toxic chemical, not a supplement."),
    ("AMMONIA INGESTION - DRANK AMMONIA.", 0, "Ammonia is a household chemical, not a supplement."),
    ("2YOF ATE POWDERED LAUNDRY DETERGENT.", 0, "Laundry detergent is a chemical, not a supplement."),
]

def build_prompt(narrative: str):
    examples_block = "\n".join(
        [f'Narrative: {t}\nOutput: {json.dumps({"reason": r, "label": l})}'
         for t, l, r in FEW_SHOTS]
    )
    return f"""
    Classify the FINAL narrative.

    EXAMPLES:
    {examples_block}

    FINAL Narrative: {narrative}
    Output:
    """.strip()



def parse_json_output(text: str):
    """Parses the strict JSON output from Ollama."""
    try:
        obj = json.loads(text.strip())
        if "label" in obj and "reason" in obj and obj["label"] in [0, 1]:
            obj["reason"] = str(obj["reason"]).strip()
            return obj
    except Exception:
        return None
    return None



def get_ollama_prediction_with_reason(narrative: str):
    IRON_PAT = re.compile(r"\b(iron|fe|ferrous|ferric|ferro)\b", re.IGNORECASE)
    # Hard rule: any iron mention => 0 (prevents LLM mistakes)
    if IRON_PAT.search(narrative):
        return 0, "Mentions iron formulation (e.g., 'iron/Fe/ferrous'); iron-containing products are excluded."

    prompt = build_prompt(narrative)

    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        format="json",  # <-- Added JSON Mode
        options={"temperature": 0}
    )

    out = resp["message"]["content"]
    parsed = parse_json_output(out)

    # If parsing fails (e.g., missing keys), retry once
    if parsed is None:
        retry_prompt = prompt + '\n\nREMINDER: Output MUST contain exact keys: {"reason":"...", "label":0 or 1}'
        resp2 = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": retry_prompt},
            ],
            format="json", # <-- Added JSON Mode here too
            options={"temperature": 0}
        )
        parsed = parse_json_output(resp2["message"]["content"])

    # Conservative final fallback (no guessing)
    if parsed is None:
        return 0, "Invalid model output; defaulted to 0 (do not guess)."

    return int(parsed["label"]), parsed["reason"]




def run_ollama_classification(df, n_samples=200):
    tqdm.pandas(desc="Classifying with Ollama")

    print("Running Llama3.1:8b few-shot classification with reasons on 200 samples...")

    # --- Balanced sample: 50% label 0 and 50% label 1 --- 
    n_each = n_samples // 2

    # Extract the samples
    df_0 = df[df["Ground_Truth"] == 0].sample(n_each, random_state=42)
    df_1 = df[df["Ground_Truth"] == 1].sample(n_each, random_state=42)
    df_sample = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)

    # Use progress_apply instead of apply to see the progress bar!
    preds = df_sample["Narrative"].progress_apply(get_ollama_prediction_with_reason)

    # Save the results into the dataframe
    df_sample["Ollama_Label"] = preds.apply(lambda x: x[0])
    df_sample["Ollama_Reason"] = preds.apply(lambda x: x[1])


    df_0 = df_sample[df_sample["Ground_Truth"] == 0].head(5)
    df_1 = df_sample[df_sample["Ground_Truth"] == 1].head(5)
    df_display = pd.concat([df_0, df_1])[["Narrative", "Ground_Truth", "Ollama_Label", "Ollama_Reason"]]
    print(df_display)
    
    return df_sample



def evaluate_and_save(df_sample, out_excel="../data/NEISS_Supplement_200_Samples.xlsx"):
    """
    Evaluate the classification results and export data for downstream BERT training.
    """
    print(f"Saving evaluation file to {out_excel}...")
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        df_sample.to_excel(writer, sheet_name="sample_eval", index=False)

    # Clean export for BERT script
    os.makedirs("../data", exist_ok=True)
    out_csv = "../data/bert_training_data.csv"
    print(f"Saving clean dataset for BERT to {out_csv}...")
    
    df_bert = df_sample[["Narrative", "Ground_Truth"]].copy()
    df_bert.rename(columns={"Ground_Truth": "label", "Narrative": "text"}, inplace=True)
    df_bert.to_csv(out_csv, index=False)


    y_true = df_sample["Ground_Truth"].astype(int)
    pred_col = "Ollama_Label" if "Ollama_Label" in df_sample.columns else "Ollama_Prediction"
    y_pred = df_sample[pred_col].astype(int)

    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")

    print("\n--- Detailed Classification Report ---")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))

    cm = confusion_matrix(y_true, y_pred)
    print("--- Confusion Matrix ---")
    print(f"True Negatives:  {cm[0][0]:<4} | False Positives: {cm[0][1]:<4}")
    print(f"False Negatives: {cm[1][0]:<4} | True Positives:  {cm[1][1]:<4}\n")



def main():
    """
    Main pipeline orchestrator.
    """
    print("--- Starting NEISS Pipeline ---")
    
    # 1. Data Loading
    df = load_and_preprocess_data('../data/PoisonedOnly_NEISS_2004-2023.xlsx')
    
    # 2. Statistical Analysis (printed to console)
    analyze_word_frequencies(df)
    
    # 3. LLM Inference
    df_classified = run_ollama_classification(df, n_samples=200)
    
    # 4. Metrics & Export
    evaluate_and_save(df_classified)
    
    print("--- Pipeline Completed ---")

# Execute the main function only if this script is run directly
if __name__ == "__main__":
    main()
# %%
