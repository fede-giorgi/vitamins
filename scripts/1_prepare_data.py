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
import os
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from collections import Counter
from src.load_data import load_and_preprocess_data

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)

load_dotenv()

# --- Configurazione LLM ---
#LLM_PROVIDER = "gemini"       # "ollama", "openai", "gemini"
#MODEL_NAME = "gemini-3.1-flash-lite-preview" # "gemma4:e4b", "gpt-4o-mini", "gemini-3.1-flash-lite-preview"

LLM_PROVIDER = "ollama" 
MODEL_NAME = "gemma4:e4b"

def get_llm():
    """Factory function per istanziare il modello LangChain in base al provider."""
    if LLM_PROVIDER == "ollama":
        return ChatOllama(model=MODEL_NAME, temperature=0, format="json")
    elif LLM_PROVIDER == "openai":
        return ChatOpenAI(model=MODEL_NAME, temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    elif LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(model=MODEL_NAME, temperature=0)
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

# Inizializza il modello a livello globale
llm = get_llm()

SYSTEM_MSG = """
You are a strict binary classifier for emergency department (ED) narratives.
You will receive a batch of narratives formatted as a JSON dictionary: {"id_1": "text_1", "id_2": "text_2", ...}

OUTPUT FORMAT: You MUST return a JSON array containing EXACTLY ONE dictionary for each narrative provided.
Example:
[
  {"id": "id_1", "reason": "short reason", "label": 0},
  {"id": "id_2", "reason": "short reason", "label": 1}
]

TASK:
Classify whether the narrative involves exposure/ingestion/overdose/adverse reaction to a STRICTLY HARMLESS VITAMIN.

LABEL 1 (POSITIVE) - INCLUSION RULES:
- label=1 ONLY if the narrative explicitly involves a clear, traditional VITAMIN or fish oil.
  Examples: "multivitamin", "childrens gummy multivitamins", "vitamin D", "vitamin C", "fish oil".
- SAFE CO-INGESTIONS: If the patient ingested a vitamin alongside another SAFE substance (e.g., "ate vitamin C and a multivitamin", "drank juice and took vitamins"), label it 1.
- MISSPELLINGS & TYPOS: ED narratives are badly written. You MUST tolerate and accept typos for vitamins (e.g., "mutivitiamins", "vitmin", "gummis").
- REDACTED BRANDS (***): If the word "vitamin" is present but the brand is redacted (e.g., "***VITAMINS", "VITAMINS ***"), label it 1. The asterisks just hide the brand name, it is still a vitamin.

LABEL 0 (NEGATIVE) - EXCLUSION RULES (Highest Priority):
- EXCLUSIONS OVERRIDE INCLUSIONS: If ANY of the following are true, you MUST label=0, even if the word "vitamin" is present.
  1. IRON & PRENATAL EXCEPTION: Any explicit mention of iron (e.g., "WITH IRON", "IRON", "Fe", "ferrous"). AND ALL "PRENATAL" or "PRENAT" vitamins MUST be excluded (label=0) because they implicitly contain toxic amounts of iron. *Note: Standard "multivitamins" are safe unless iron is stated.*
  2. DANGEROUS CO-INGESTIONS: If the patient ingested a safe vitamin ALONG WITH a dangerous/prescription medication (e.g., beta blocker, Tylenol, potassium pill, cough med), label it 0. The ED visit is driven by the dangerous drug, not the vitamin.
  3. VITAMINS GIVEN AS TREATMENT: If the vitamin was ADMINISTERED by doctors/hospital as a treatment for something else (e.g., "GIVEN THIAMINE", "GIVEN MVI", "Rx vitamins" for alcohol/whiskey intoxication), label it 0. We ONLY want accidental poisonings/ingestions of vitamins, not medical treatments.
  4. NON-VITAMIN SUPPLEMENTS: Melatonin, diet pills, fat loss pills, herbal supplements, botanicals, creatine, protein.
  5. CANNABIS: CBD, THC, marijuana gummies, weed.
  6. FULLY REDACTED: ONLY if the *entire* substance is unknown AND the word 'vitamin' is missing (e.g., "ate chewable ***"), label it 0.
  7. PRESCRIPTION/OTC DRUGS: Any standard medication.
  8. DRUG TYPOS: Be highly vigilant for misspelled drugs (e.g., "xanTax", "tylenol", "ibuprofin").
  9. HOUSEHOLD/COSMETICS TOXINS: Shampoos, lotions, creams, or soaps that happen to have "vitamin" in their name (e.g., "shampoo with vitamin E").

GENERAL RULES:
- ACCEPT GENERIC TERMS (BENEFIT OF THE DOUBT): If the text mentions generic "vitamins", "gummy vitamins", "vitamin pills", or "childrens vitamins" without specifying the exact type, ASSUME THEY ARE SAFE (label=1) as long as NO exclusion keywords (iron, melatonin, drugs, etc.) are present. Do not punish the narrative for being vague.
- Reason must cite the key phrase that triggered your decision.
""".strip()

# Few-shot examples WITH reasons (include hard negatives for new clinical rules)
FEW_SHOTS = [
    # Positives
    ("4YOM WITH ABD PAIN S/P EATING HANDFUL OF CHILDRENS GUMMY MULTIVITAMINS", 1, "Explicit mention of 'childrens gummy multivitamins'. This is a standard safe vitamin."),
    ("PT INGESTED 10 MUTIVITIAMINS THEY WERE IN A PLASTIC BAG", 1, "Mentions 'mutivitiamins' (misspelled). Assumed safe unless iron is explicitly mentioned."),
    ("16MOF GOT INTO VITAMINS 4 DAYS AGO, INGESTED UNKNOWN NUMBER OF ***,***,***", 1, "The generic word 'VITAMINS' is present. The asterisks merely hide the brand. Assumed safe."),
    ("PT ATE ***VITAMINS", 1, "The word 'VITAMINS' is attached to the redaction. It is a vitamin, asterisks just hide the brand."),
    ("3YOF INGESTION OF 20 MULTIVITAMIN GUMMIES, *** GUMMIES.", 1, "Clear mention of 'multivitamin gummies'. The presence of redacted '***' does not negate the vitamin ingestion."),

    # Hard negatives (Drugs, Cosmetics, Melatonin, Cannabis)
    ("23MOM SWALLOWED *** SHAMPOO LUXURIOUS MOISTURE & VITAMINE", 0, "Mentions 'shampoo'. Cosmetics and household chemicals are excluded, even if they contain the word vitamin."),
    ("ADULT TOOK MELATONIN GUMMIES; DIZZY.", 0, "Mentions 'melatonin', which is an excluded non-vitamin supplement."),
    ("3 YOF FOUND EATING CBD GUMMY.", 0, "Mentions 'CBD', cannabis products are excluded."),
    ("3YF FD WITH OPEN BOTTLE OF CHILDREN'S CHEWABLE ***.", 0, "The specific substance is totally redacted ('***'). We cannot confirm it is a vitamin."),

    # Co-ingestion overrides (Must be 0)
    ("INGESTED BETA BLOCKER PILL, MULTI VITAMIN AND POTASSIUM PILL FROM GRANDMAS DAILY PILL MINDER", 0, "Co-ingestion. A 'MULTI VITAMIN' is mentioned, but was ingested alongside dangerous medications ('BETA BLOCKER', 'POTASSIUM'). Excluded."),

    # Vitamin as Treatment overrides (Must be 0)
    ("PT FOUND UNRESPONSIVE ON THE FLOOR AT HOME LYING NEXT TO AN EMPTY BOTTLE OF WHISKEY BAC- 299, GIVEN FLUIDS, THIAMINE MVI", 0, "Mentions 'THIAMINE' and 'MVI' (multivitamin), but they were GIVEN as treatment by the hospital for alcohol ingestion. Not an accidental vitamin ingestion."),

    # Iron exceptions (Must be 0)
    ("PATIENT INGESTED 1/2 BOTTLE *** VITAMINS 10 MG IRON", 0, "Explicitly mentions 'IRON' in the formulation."),
    ("PRENATAL VITAMINS WITH IRON INGESTION.", 0, "Explicitly mentions 'PRENATAL' and 'IRON'. Both trigger exclusion."),
    ("PT INGESTED 1 *** PRENAT VITAMIN", 0, "Mentions 'PRENAT VITAMIN'. Prenatal vitamins implicitly contain high iron and must be excluded.")
]

EXCLUSION_RULES = [
    (re.compile(r"\b(cbd|thc|marijuana|weed|cannabis|hemp)\b", re.IGNORECASE), 
     "Mentions cannabis product."),
    (re.compile(r"\b(melatonin)\b", re.IGNORECASE), 
     "Mentions melatonin (non-vitamin supplement)."),
    (re.compile(r"\b(diet pill|fat loss|weight loss)\b", re.IGNORECASE), 
     "Mentions weight-loss/diet supplement.")
]


def build_prompt(narratives_dict: dict) -> str:
    """
    Constructs the prompt for the LangChain model by combining the dynamic batch of narratives
    with predefined few-shot examples.
    """
    examples_block = "[\n" + ",\n".join(
        [f'  {{"id": "ex{i}", "narrative": {json.dumps(t)}, "output": {json.dumps({"reason": r, "label": l})}}}'
         for i, (t, l, r) in enumerate(FEW_SHOTS)]
    ) + "\n]"
    
    return f"""
    Classify the following FINAL batch of narratives.

    EXAMPLES of correctly classified items:
    {examples_block}

    FINAL BATCH TO CLASSIFY:
    {json.dumps(narratives_dict, indent=2)}
    
    Output exactly a JSON array containing the predictions. Do not add markdown text around the array.
    """.strip()



def parse_json_output(text) -> list | None:
    """
    Parses the JSON array output from LangChain and validates the required fields.
    Also strips markdown wrappers (```json ... ```) and handles Gemini's list output.
    """
    try:
        # Handle langchain-google-genai returning lists instead of strings
        if isinstance(text, list):
            if len(text) > 0 and isinstance(text[0], dict) and "text" in text[0] and "id" not in text[0]:
                for item in text:
                    if isinstance(item, dict) and "text" in item:
                        text = item["text"]
                        break
                else:
                    text = str(text)
            else:
                pass # Already a parsed array

        if not isinstance(text, str) and not isinstance(text, list):
            text = str(text)

        if isinstance(text, str):
            text = text.strip()
            # Strip potential markdown formatting that LLMs often incorrectly add
            if text.startswith("```json"):
                text = text[7:]
            elif text.startswith("```"):
                text = text[3:]
                
            if text.endswith("```"):
                text = text[:-3]
                
            text = text.strip()
    
            obj = json.loads(text)
        else:
            obj = text

        if isinstance(obj, list):
            return obj
        elif isinstance(obj, dict) and "id" in obj:
            return [obj]
            
    except Exception:
        return None
    return None



def get_llm_batch_predictions(batch: dict) -> dict:
    """
    Gets the classification prediction and reasoning for a batch of narratives using the LangChain model.
    It first applies hardcoded exclusion rules locally (0 costs) before sending to the LLM.
    """
    results = {}
    pending_batch = {}
    
    # 1. Apply hard rules locally immediately (saves tokens and API costs!)
    for rid, narrative in batch.items():
        excluded = False
        for pattern, reason_text in EXCLUSION_RULES:
            if pattern.search(narrative):
                results[rid] = (0, f"Hard Rule: {reason_text}")
                excluded = True
                break
        if not excluded:
            pending_batch[rid] = narrative

    if not pending_batch:
        return results

    # 2. Build the prompt for the remaining narratives
    prompt = build_prompt(pending_batch)

    messages = [
        SystemMessage(content=SYSTEM_MSG),
        HumanMessage(content=prompt)
    ]

    try:
        # 3. Query the LangChain model enforcing JSON format
        resp = llm.invoke(messages)
        parsed = parse_json_output(resp.content)
    
        # 4. Fallback retry
        if parsed is None or not isinstance(parsed, list):
            retry_prompt = prompt + '\n\nREMINDER: Output MUST be a valid JSON array of objects exactly like: [{"id": "...", "reason":"...", "label":0 or 1}]'
            messages[-1] = HumanMessage(content=retry_prompt)
            resp2 = llm.invoke(messages)
            parsed = parse_json_output(resp2.content)
    
        # 5. Extract answers
        if parsed is not None and isinstance(parsed, list):
            for item in parsed:
                if "id" in item and "label" in item and "reason" in item:
                    results[str(item["id"])] = (int(item["label"]), str(item["reason"]))
                    
        # 6. Fallback final for missing predictions
        for rid in pending_batch.keys():
            if str(rid) not in results:
                results[str(rid)] = (0, "Invalid model output or missing from batch; defaulted to 0.")
    
        return results
        
    except Exception as e:
        for rid in pending_batch.keys():
            results[str(rid)] = (0, f"Error calling {LLM_PROVIDER}: {str(e)}")
        return results




def run_llm_classification(df: pd.DataFrame, n_samples: int | None = 200) -> pd.DataFrame:
    """
    Runs the LLM-based few-shot classification pipeline.
    If n_samples is provided, it runs on a balanced subset.
    If n_samples is None, it runs on the entire dataset.
    
    Args:
        df (pd.DataFrame): The full dataframe containing the loaded clinical narratives.
        n_samples (int | None): The total number of samples to classify, or None for the full dataset.
        
    Returns:
        pd.DataFrame: A new dataframe containing the LLM predictions and reasons.
    """
    tqdm.pandas(desc=f"Classifying with {LLM_PROVIDER}")

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

    # Optimize concurrency and batch size
    is_local = (LLM_PROVIDER == "ollama")
    batch_size = 10 if is_local else 40
    max_workers = 1 if is_local else 10
    
    print(f"\nProcessing in parallel using {max_workers} thread(s) with BATCH SIZE = {batch_size}...")

    # Create batches explicitly using the DataFrame Index as ID
    batches = []
    current_batch = {}
    for idx, row in df_sample.iterrows():
        current_batch[str(idx)] = row["Narrative"]
        if len(current_batch) == batch_size:
            batches.append(current_batch)
            current_batch = {}
    if current_batch:
        batches.append(current_batch)

    final_results = {}

    # ThreadPool mapping (preserves order naturally or through ID tracking)
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        for batch_res in tqdm(executor.map(get_llm_batch_predictions, batches), total=len(batches), desc=f"Classifying batched {LLM_PROVIDER}"):
            final_results.update(batch_res)

    # Save the generated labels and reasons into the dataframe mapped by exact ID
    df_sample["LLM_Label"] = [final_results[str(idx)][0] for idx in df_sample.index]
    df_sample["LLM_Reason"] = [final_results[str(idx)][1] for idx in df_sample.index]

    # Display a small preview of the results
    df_0_display = df_sample[df_sample["Ground_Truth"] == 0].head(5)
    df_1_display = df_sample[df_sample["Ground_Truth"] == 1].head(5)
    df_display = pd.concat([df_0_display, df_1_display])[["Narrative", "Ground_Truth", "LLM_Label", "LLM_Reason"]]
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
    
    # Prevent Excel formula errors by prefixing text starting with =, +, -, or @ with a single quote
    df_excel = df_sample.copy()
    for col in df_excel.select_dtypes(include=['object', 'string']):
        df_excel[col] = df_excel[col].apply(
            lambda x: f"'{x}" if isinstance(x, str) and str(x).startswith(('=', '+', '-', '@')) else x
        )
        
    with pd.ExcelWriter(out_excel, engine="openpyxl") as writer:
        df_excel.to_excel(writer, sheet_name="sample_eval", index=False)

    # Export for BERT (Ensuring we use the LLM Teacher's labels)
    os.makedirs(data_dir, exist_ok=True)
    out_csv = os.path.join(data_dir, "bert_training_data.csv")
    
    # CRITICAL: We take the LLM's intelligence (LLM_Label) as the new reference (teacher labels)
    df_bert = df_sample[["Narrative", "LLM_Label"]].copy()
    df_bert.rename(columns={"LLM_Label": "label", "Narrative": "text"}, inplace=True)
    
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
    
    # 2. LLM Inference: Generate pseudo-labels using LangChain
    df_classified = run_llm_classification(df, n_samples=1000)
    
    # 3. Save and Export: Prepare data for human analysis and BERT training
    save(df_classified)
    
    print("--- Pipeline Completed ---")

# Execute the main function only if this script is run directly
if __name__ == "__main__":
    main()
# %%
