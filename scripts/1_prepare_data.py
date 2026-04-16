"""
NEISS Vitamin and Supplement ED Visit Analysis (Ollama + Statistical Patterns)

This script analyzes NEISS data to identify trends in emergency department visits related to vitamins and supplements. 

Key Features:
1. Ground Truth Comparison: Labels cases based on product codes (1927, 1931, 1932).
2. Statistical Word Analysis: Automatically identifies words that distinguish supplements from non-supplements.
3. Few-Shot Ollama Classification: Uses the discovered words to guide the designated LLM via Ollama for zero-manual-effort classification.
4. Metric Validation: Calculates accuracy and recall targeting >80% recall.
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
    """
    Constructs the prompt for the Ollama model by combining the dynamic narrative
    with predefined few-shot examples.
    
    Args:
        narrative (str): The clinical narrative text to classify.
        
    Returns:
        str: The fully formatted prompt ready for the LLM.
    """
    # Create a block of examples showing the desired JSON output format
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
    """
    Parses the strict JSON output from Ollama and validates the required fields.
    
    Args:
        text (str): The raw text output generated by the LLM.
        
    Returns:
        dict | None: The parsed dictionary containing 'label' and 'reason' if valid,
                     otherwise None if parsing or validation fails.
    """
    try:
        # Attempt to load the JSON string into a Python dictionary
        obj = json.loads(text.strip())
        # Validate that required keys are present and the label is binary
        if "label" in obj and "reason" in obj and obj["label"] in [0, 1]:
            obj["reason"] = str(obj["reason"]).strip()
            return obj
    except Exception:
        # If any exception occurs during parsing (e.g., JSONDecodeError), return None
        return None
    return None



def get_ollama_prediction_with_reason(narrative: str) -> tuple[int, str]:
    """
    Gets the classification prediction and reasoning for a narrative using the Ollama model.
    It first applies hardcoded exclusion rules before querying the LLM to save compute
    and enforce strict medical guidelines.
    
    Args:
        narrative (str): The clinical narrative describing the ED visit.
        
    Returns:
        tuple[int, str]: A tuple containing the binary label (0 or 1) and the reasoning string.
    """
    # 1. Apply hard rules: instantly reject narratives matching exclusion keywords
    for pattern, reason_text in EXCLUSION_RULES:
        if pattern.search(narrative):
            return 0, f"Hard Rule: {reason_text}"

    # 2. Build the few-shot prompt for the remaining narratives
    prompt = build_prompt(narrative)

    # 3. Query the Ollama model enforcing JSON format
    resp = ollama.chat(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_MSG},
            {"role": "user", "content": prompt},
        ],
        format="json",  # Enforce JSON output mode
        options={"temperature": 0}  # Use zero temperature for deterministic outputs
    )

    out = resp["message"]["content"]
    parsed = parse_json_output(out)

    # 4. Fallback retry: If parsing fails (e.g., missing keys), retry the query once
    # with an explicit reminder of the required JSON structure.
    if parsed is None:
        retry_prompt = prompt + '\n\nREMINDER: Output MUST contain exact keys: {"reason":"...", "label":0 or 1}'
        resp2 = ollama.chat(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_MSG},
                {"role": "user", "content": retry_prompt},
            ],
            format="json", # Enforce JSON mode on retry
            options={"temperature": 0}
        )
        parsed = parse_json_output(resp2["message"]["content"])

    # 5. Conservative final fallback: If it still fails, default to 0 (harmless)
    if parsed is None:
        return 0, "Invalid model output; defaulted to 0 (do not guess)."

    return int(parsed["label"]), parsed["reason"]




def run_ollama_classification(df: pd.DataFrame, n_samples: int | None = 200) -> pd.DataFrame:
    """
    Runs the Ollama-based few-shot classification pipeline.
    If n_samples is provided, it runs on a balanced subset.
    If n_samples is None, it runs on the entire dataset.
    
    Args:
        df (pd.DataFrame): The full dataframe containing the loaded clinical narratives.
        n_samples (int | None): The total number of samples to classify, or None for the full dataset.
        
    Returns:
        pd.DataFrame: A new dataframe containing the LLM predictions and reasons.
    """
    tqdm.pandas(desc="Classifying with Ollama")

    if n_samples is not None:
        print(f"Running {MODEL_NAME} few-shot classification with reasons on {n_samples} balanced samples...")
        # --- Balanced sample: 50% label 0 and 50% label 1 --- 
        n_each = n_samples // 2

        # Extract the samples, forcing a balanced dataset
        df_0 = df[df["Ground_Truth"] == 0].sample(n_each, random_state=42)
        df_1 = df[df["Ground_Truth"] == 1].sample(n_each, random_state=42)
        df_sample = pd.concat([df_0, df_1]).sample(frac=1, random_state=42).reset_index(drop=True)
    else:
        print(f"Running {MODEL_NAME} few-shot classification with reasons on the FULL dataset ({len(df)} samples)...")
        df_sample = df.copy()

    # Apply the classification function using progress_apply to display a progress bar
    preds = df_sample["Narrative"].progress_apply(get_ollama_prediction_with_reason)

    # Save the generated labels and reasons into the dataframe
    df_sample["Ollama_Label"] = preds.apply(lambda x: x[0])
    df_sample["Ollama_Reason"] = preds.apply(lambda x: x[1])

    # Display a small preview of the results
    df_0_display = df_sample[df_sample["Ground_Truth"] == 0].head(5)
    df_1_display = df_sample[df_sample["Ground_Truth"] == 1].head(5)
    df_display = pd.concat([df_0_display, df_1_display])[["Narrative", "Ground_Truth", "Ollama_Label", "Ollama_Reason"]]
    print(df_display)
    
    return df_sample



def save(df_sample: pd.DataFrame, out_excel: str | None = None) -> None:
    """
    Saves the LLM-classified data into an Excel file for manual review and exports 
    a clean CSV dataset for subsequent BERT model training.
    
    Args:
        df_sample (pd.DataFrame): The dataframe containing the classified narratives.
        out_excel (str | None): Optional custom path for the output Excel file. 
                                Defaults to a generic name in the data folder.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    
    if out_excel is None:
        out_excel = os.path.join(data_dir, f"NEISS_Supplement_{len(df_sample)}_Samples.xlsx")
    
    print(f"Saving LLM results for review: {out_excel}")
    # Export full detailed data to Excel for human evaluation
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        df_sample.to_excel(writer, sheet_name="sample_eval", index=False)

    # Export for BERT (Ensuring we use the LLM Teacher's labels)
    os.makedirs(data_dir, exist_ok=True)
    out_csv = os.path.join(data_dir, "bert_training_data.csv")
    
    # CRITICAL: We take the LLM's intelligence (Ollama_Label) as the new reference (teacher labels)
    df_bert = df_sample[["Narrative", "Ollama_Label"]].copy()
    df_bert.rename(columns={"Ollama_Label": "label", "Narrative": "text"}, inplace=True)
    
    # Drop rows that experienced parsing failures to ensure data quality for fine-tuning
    df_bert = df_bert.dropna(subset=["label"])
    df_bert.to_csv(out_csv, index=False)
    print(f"Clean BERT dataset exported: {out_csv} ({len(df_bert)} rows)")



def main():
    """
    Main pipeline orchestrator for the NEISS data preparation process.
    Loads data, runs the active LLM classifier, and saves the output datasets.
    """
    print("--- Starting NEISS Pipeline ---")
    
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_path = os.path.join(base_dir, 'data', 'PoisonedOnly_NEISS_2004-2023.xlsx')
    
    # 1. Data Loading: Gather raw ED visits
    df = load_and_preprocess_data(data_path)
    
    # 2. LLM Inference: Generate pseudo-labels using Ollama
    df_classified = run_ollama_classification(df, n_samples=None)
    
    # 3. Save and Export: Prepare data for human analysis and BERT training
    save(df_classified)
    
    print("--- Pipeline Completed ---")

# Execute the main function only if this script is run directly
if __name__ == "__main__":
    main()
# %%
