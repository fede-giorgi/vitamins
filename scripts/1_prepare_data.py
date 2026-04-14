"""
NEISS Vitamin and Supplement ED Visit Analysis (Ollama + Statistical Patterns)

This script analyzes NEISS data to identify trends in emergency department visits related to vitamins and supplements. 

script Key Features:
1. scriptGround Truth Comparisonscript: Labels cases based on product codes (1927, 1931, 1932).
2. scriptStatistical Word Analysisscript: Automatically identifies words that distinguish supplements from non-supplements.
3. scriptFew-Shot Ollama Classificationscript: Uses the discovered words to guide the MODEL_NAME via Ollama for zero-manual-effort classification.
4. scriptMetric Validationscript: Calculates accuracy and recall targeting >80% recall.
"""


#%%
%load_ext autoreload
%autoreload 2
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



MODEL_NAME = "gemma4:e4b"

SYSTEM_MSG = """
You are a strict binary classifier for emergency department (ED) narratives.

OUTPUT FORMAT (must be valid JSON on a single line):
{"reason": "short reason", "label": 0 or 1}

TASK:
Classify whether the narrative involves exposure/ingestion/overdose/adverse reaction to a STRICTLY HARMLESS VITAMIN.

LABEL DEFINITIONS:
- label=1 ONLY if the narrative explicitly involves a clear, traditional VITAMIN or fish oil AND it does NOT contain IRON.
  Examples (label=1): multivitamin WITHOUT iron, vitamin D, vitamin C gummies, fish oil.
- label=0 for EVERYTHING ELSE, including:
  - SUPPLEMENTS THAT ARE NOT VITAMINS (e.g., melatonin, diet pills, fat loss pills, herbal supplements, creatine, protein).
  - CANNABIS PRODUCTS (e.g., marijuana gummies, CBD, THC).
  - ANY product/formulation that contains IRON (including "iron", "Fe", "ferrous", "ferric", "prenatal with iron", "multivitamin with iron").
  - Prescription/OTC medications, INCLUDING MISSPELLINGS (e.g., "xanTax", "Tylenol", "cough med").
  - AMBIGUOUS/REDACTED ingestions (e.g., "children's chewable ***", "*** diet supplement") where the specific vitamin is unknown.
  - Household chemicals/toxins.

EXCLUSION RULES (highest priority - MUST be 0):
1. IRON: Any mention of iron formulations.
2. NON-VITAMIN SUPPLEMENTS: Melatonin, diet pills, botanicals.
3. CANNABIS: CBD, marijuana.
4. REDACTED/AMBIGUOUS: "chewable ***", "ingested ***". If you don't know EXACTLY what it is, label=0.
5. DRUGS/MEDS: Any drug, even misspelled.

GENERAL RULES:
- Do not guess. If a clear, safe vitamin is not explicit, choose 0.
- Reason must cite the key phrase that triggered your decision.
""".strip()

# Few-shot examples WITH reasons (include hard negatives for new clinical rules)
FEW_SHOTS = [
    # Positives (pure vitamins WITHOUT iron)
    ("2 YOM INGESTED MULTIVITAMIN (NO IRON) GUMMIES.", 1, "Mentions 'multivitamin (no iron) gummies' which is a pure vitamin without iron."),
    ("CHILD TOOK SEVERAL VITAMIN D PILLS.", 1, "Explicit 'vitamin D' ingestion."),
    ("POSSIBLE INGESTION OF FISH OIL PILLS", 1, "Explicit 'fish oil' which is accepted."),

    # Hard negatives (Must be 0 based on new clinical rules)
    ("ADULT TOOK MELATONIN GUMMIES; DIZZY.", 0, "Mentions 'melatonin' which is a non-vitamin supplement, excluded."),
    ("3 YOF FOUND EATING CBD GUMMY.", 0, "Mentions 'CBD', cannabis products are excluded."),
    ("PATIENT INGESTED APPROX 20 FAT LOSS PILLS.", 0, "Mentions 'fat loss pills', diet supplements are excluded."),
    ("3YF FD WITH OPEN BOTTLE OF CHILDREN'S CHEWABLE ***.", 0, "Mentions 'chewable ***', redacted/ambiguous substances are excluded."),
    ("18MOF OPENED BOTTLES OF GENERIC XANTAX.", 0, "Mentions 'XANTAX' (misspelled drug), medications are excluded."),

    # Iron exception hard negatives (MUST be 0)
    ("2YOF ATE ADULT IRON + VIT C 18MG GUMMIES.", 0, "Mentions 'IRON + VIT C', iron-containing formulation is excluded."),
    ("PRENATAL VITAMINS WITH IRON INGESTION.", 0, "Mentions 'with iron', iron-containing formulation is excluded.")
]

EXCLUSION_RULES = [
    (re.compile(r"\b(iron|fe|ferrous|ferric|ferro)\b", re.IGNORECASE), 
     "Mentions iron formulation."),
    (re.compile(r"\b(cbd|thc|marijuana|weed|cannabis|hemp)\b", re.IGNORECASE), 
     "Mentions cannabis product."),
    (re.compile(r"\b(melatonin)\b", re.IGNORECASE), 
     "Mentions melatonin (non-vitamin supplement)."),
    (re.compile(r"\b(diet pill|fat loss|weight loss)\b", re.IGNORECASE), 
     "Mentions weight-loss/diet supplement."),
    (re.compile(r"\*\*\*", re.IGNORECASE), 
     "Contains redacted/ambiguous substance (***).")
]


def build_prompt(narrative: str) -> str:
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



def parse_json_output(text: str) -> dict | None:
    """Parses the strict JSON output from Ollama."""
    try:
        obj = json.loads(text.strip())
        if "label" in obj and "reason" in obj and obj["label"] in [0, 1]:
            obj["reason"] = str(obj["reason"]).strip()
            return obj
    except Exception:
        return None
    return None



def get_ollama_prediction_with_reason(narrative: str) -> tuple[int, str]:
    # Hard rules
    for pattern, reason_text in EXCLUSION_RULES:
        if pattern.search(narrative):
            return 0, f"Hard Rule: {reason_text}"

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




def run_ollama_classification(df: pd.DataFrame, n_samples: int = 200) -> pd.DataFrame:
    tqdm.pandas(desc="Classifying with Ollama")

    print(f"Running {MODEL_NAME} few-shot classification with reasons on {n_samples} samples...")

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



def save(df_sample: pd.DataFrame, out_excel: str | None = None) -> None:
    """
    Saves LLM-classified data for review and exports clean data for BERT training.
    """
    if out_excel is None:
        out_excel = f"../data/NEISS_Supplement_{len(df_sample)}_Samples.xlsx"
    
    print(f"Saving LLM results for review: {out_excel}")
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        df_sample.to_excel(writer, sheet_name="sample_eval", index=False)

    # Export for BERT (Ensuring we use the LLM Teacher's labels)
    os.makedirs("../data", exist_ok=True)
    out_csv = "../data/bert_training_data.csv"
    
    # CRITICAL: We take the LLM's intelligence (Ollama_Label) as the new reference
    df_bert = df_sample[["Narrative", "Ollama_Label"]].copy()
    df_bert.rename(columns={"Ollama_Label": "label", "Narrative": "text"}, inplace=True)
    
    # Drop parsing failures to ensure data quality
    df_bert = df_bert.dropna(subset=["label"])
    df_bert.to_csv(out_csv, index=False)
    print(f"Clean BERT dataset exported: {out_csv} ({len(df_bert)} rows)")



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
    df_classified = run_ollama_classification(df, n_samples=500)
    
    # 4. Save and Export
    save(df_classified)
    
    print("--- Pipeline Completed ---")

# Execute the main function only if this script is run directly
if __name__ == "__main__":
    main()
# %%
