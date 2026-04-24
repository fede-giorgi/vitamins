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
import time
from dotenv import load_dotenv

from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI, HarmCategory, HarmBlockThreshold
from langchain_core.messages import SystemMessage, HumanMessage
from tqdm.auto import tqdm
from concurrent.futures import ThreadPoolExecutor
from src.load_data import load_and_preprocess_data
from src.classification import SYSTEM_MSG, FEW_SHOTS, EXCLUSION_RULES

plt.style.use('ggplot')
plt.rcParams['figure.figsize'] = (12, 6)

load_dotenv()

# --- LLM Configuration ---
#LLM_PROVIDER = "gemini"       # "ollama", "openai", "gemini"
#MODEL_NAME = "gemini-2.5-flash-lite" # "gemma4:e4b", "gpt-4o-mini", "gemini-3.1-flash-lite-preview"

LLM_PROVIDER = "gemini" 
MODEL_NAME = "gemini-2.5-flash-lite"

def get_llm():
    """Factory function to instantiate the LangChain model based on the provider."""
    if LLM_PROVIDER == "ollama":
        return ChatOllama(model=MODEL_NAME, temperature=0, format="json")
    elif LLM_PROVIDER == "openai":
        return ChatOpenAI(model=MODEL_NAME, temperature=0, model_kwargs={"response_format": {"type": "json_object"}})
    elif LLM_PROVIDER == "gemini":
        return ChatGoogleGenerativeAI(
            model=MODEL_NAME, 
            temperature=0,
            safety_settings={
                HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
                HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            }
        )
    else:
        raise ValueError(f"Unknown LLM_PROVIDER: {LLM_PROVIDER}")

# Initialize the model globally
llm = get_llm()
# Initialize fallback local model: this model will only be called if the primary Google API fails (e.g., safety filters). It uses a local model via Ollama
try:
    fallback_llm = ChatOllama(model="gemma4:e4b", temperature=0, format="json")
except Exception as e:
    print(f"Warning: Could not initialize fallback local model. Fallback will be disabled. Error: {e}")
    fallback_llm = None



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
    If the primary LLM fails or is blocked (safety filters), it triggers a 'Split & Rescue' strategy:
    it splits the batch into smaller pieces to try and rescue as many as possible with Gemini 
    before finally falling back to Ollama for only the truly problematic narratives.
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

    # 2. Try Primary LLM (e.g., Gemini)
    def call_primary(b: dict) -> dict | None:
        prompt = build_prompt(b)
        messages = [SystemMessage(content=SYSTEM_MSG), HumanMessage(content=prompt)]
        
        for attempt in range(3):  # Retry up to 3 times for micro API errors
            try:
                resp = llm.invoke(messages)
                parsed = parse_json_output(resp.content)
                
                # If Gemini formats the JSON poorly, force it to correct itself before calling Ollama
                if parsed is None or not isinstance(parsed, list):
                    retry_prompt = prompt + '\n\nREMINDER: Output MUST be a valid JSON array of objects exactly like: [{"id": "...", "reason":"...", "label":0 or 1}]'
                    messages[-1] = HumanMessage(content=retry_prompt)
                    resp2 = llm.invoke(messages)
                    parsed = parse_json_output(resp2.content)
                    
                if parsed is not None and isinstance(parsed, list):
                    return parsed
                return None
                
            except Exception as e:
                err_str = str(e).lower()
                # If it's a "Too Many Requests" (429) error, wait 3 seconds and retry
                if "429" in err_str or "quota" in err_str or "exhausted" in err_str or "503" in err_str:
                    time.sleep(3)
                    continue
                else:
                    # If it's a GENUINE security block, trigger the local Split & Rescue
                    return None
        return None

    # 3. Strategy: Try primary, if blocked, split and retry primary on smaller chunks
    # This keeps things fast by only sending the 'toxic' chunks to the slow local fallback.
    parsed_items = call_primary(pending_batch)
    
    if parsed_items is None:
        # Gemini blocked the whole batch. Let's try to rescue parts of it by splitting.
        items = list(pending_batch.items())
        # Split into 4 smaller sub-batches
        mid = len(items) // 2
        sub_batches = [dict(items[:mid]), dict(items[mid:])]
        
        # Further split if they are still large (e.g., if we started with 40, we now have 20)
        # We'll try to process these sub-batches
        all_parsed = []
        for sub in sub_batches:
            if not sub: continue
            res = call_primary(sub)
            if res is None:
                # Still blocked! If it's small enough, go to Ollama. Otherwise, split again.
                if len(sub) <= 10:
                    # Final fallback to Local Model for this specific problematic sub-batch
                    if fallback_llm is not None:
                        try:
                            prompt = build_prompt(sub)
                            fallback_resp = fallback_llm.invoke([SystemMessage(content=SYSTEM_MSG), HumanMessage(content=prompt)])
                            res = parse_json_output(fallback_resp.content)
                        except:
                            res = None
                else:
                    # Split one more time
                    sub_items = list(sub.items())
                    quarter = len(sub_items) // 2
                    for q_sub in [dict(sub_items[:quarter]), dict(sub_items[quarter:])]:
                        q_res = call_primary(q_sub)
                        if q_res is None and fallback_llm is not None:
                            try:
                                prompt = build_prompt(q_sub)
                                fallback_resp = fallback_llm.invoke([SystemMessage(content=SYSTEM_MSG), HumanMessage(content=prompt)])
                                q_res = parse_json_output(fallback_resp.content)
                            except:
                                q_res = None
                        if q_res: all_parsed.extend(q_res)
            
            if res: all_parsed.extend(res)
        parsed_items = all_parsed

    # 4. Extract answers
    if parsed_items:
        for item in parsed_items:
            if isinstance(item, dict) and "id" in item and "label" in item and "reason" in item:
                results[str(item["id"])] = (int(item["label"]), str(item["reason"]))
                
    # 5. Final fallback for missing predictions (safety net)
    for rid in pending_batch.keys():
        if str(rid) not in results:
            results[str(rid)] = (0, "Blocked/Failed in all attempts (Gemini + Local Fallback).")

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
    data_path = os.path.join(base_dir, 'data', 'raw', 'PoisonedOnly_NEISS_2004-2023.xlsx')
    
    # 1. Data Loading: Gather raw ED visits
    df = load_and_preprocess_data(data_path)
    
    # 2. LLM Inference: Generate pseudo-labels using LangChain
    df_classified = run_llm_classification(df, n_samples=None)
    
    # 3. Save and Export: Prepare data for human analysis and BERT training
    save(df_classified)
    
    print("--- Pipeline Completed ---")

# Execute the main function only if this script is run directly
if __name__ == "__main__":
    main()
# %%
