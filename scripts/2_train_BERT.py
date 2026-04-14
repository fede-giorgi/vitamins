# %%
%load_ext autoreload
%autoreload 2

import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.load_data import load_and_preprocess_data

import pandas as pd
import numpy as np
from tqdm import trange, tqdm

import src

from src.model import build_lora_model
from src.train import train_model
from src.metrics import calculate_classification_metrics

import torch
from torch.utils.data import TensorDataset, DataLoader, RandomSampler, SequentialSampler
from transformers import BertTokenizer, BertForSequenceClassification
from sklearn.model_selection import train_test_split
from torchmetrics.classification import Recall, Accuracy, AUROC, Precision
from functools import partial



def load_datasets(train_csv_path, full_excel_path):
    """
    Loads the Ollama-labeled dataset for training and the full dataset for inference.
    """
    print(f"Loading training data from {train_csv_path}...")
    df_train = pd.read_csv(train_csv_path)
    
    print(f"Loading full dataset for inference from {full_excel_path}...")
    # Assumes the function from src.load_data returns the parsed dataframe
    df_full = load_and_preprocess_data(full_excel_path)
    
    # Ensure text columns are string
    df_train['text'] = df_train['text'].astype(str)
    df_full['Narrative'] = df_full['Narrative'].astype(str)
    
    print(f"Training set size: {len(df_train)}")
    print(f"Full inference set size: {len(df_full)}")
    
    return df_train, df_full

def preprocessing(input_text, tokenizer):
    """Helper function to tokenize properly."""
    return tokenizer(
        str(input_text),
        add_special_tokens=True,
        max_length=32,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

def prepare_dataloaders(df_train, tokenizer, batch_size=32):
    """
    Tokenizes the training text and creates PyTorch DataLoaders.
    """
    print("Tokenizing training data...")
    token_id = []
    attention_masks = []
    
    text = df_train['text'].values
    labels = df_train['label'].values.astype(int)

    for sample in text:
        encoding_dict = preprocessing(sample, tokenizer)
        token_id.append(encoding_dict['input_ids'])
        attention_masks.append(encoding_dict['attention_mask'])

    token_id = torch.cat(token_id, dim=0)
    attention_masks = torch.cat(attention_masks, dim=0)
    labels_tensor = torch.tensor(labels)

    # Train/Validation Split (20% validation)
    val_ratio = 0.2
    train_idx, val_idx = train_test_split(
        np.arange(len(labels)),
        test_size=val_ratio,
        stratify=labels,
        random_state=42
    )

    train_set = TensorDataset(token_id[train_idx], attention_masks[train_idx], labels_tensor[train_idx])
    val_set = TensorDataset(token_id[val_idx], attention_masks[val_idx], labels_tensor[val_idx])

    train_dataloader = DataLoader(train_set, sampler=RandomSampler(train_set), batch_size=batch_size)
    validation_dataloader = DataLoader(val_set, sampler=SequentialSampler(val_set), batch_size=batch_size)

    return train_dataloader, validation_dataloader



def run_inference_on_full_dataset(model, df_full, tokenizer, device, batch_size=32):
    """
    Runs the trained BERT model on the complete NEISS dataset to generate final predictions.
    """
    print(f"Running inference on {len(df_full)} total cases...")
    
    # Batched tokenization: massively faster than a row-by-row for-loop
    print("Tokenizing full inference dataset...")
    encodings = tokenizer(
        df_full['Narrative'].tolist(),
        add_special_tokens=True,
        max_length=32,
        padding='max_length',
        truncation=True,
        return_attention_mask=True,
        return_tensors='pt'
    )

    pred_dataset = TensorDataset(encodings['input_ids'], encodings['attention_mask'])
    pred_dataloader = DataLoader(pred_dataset, sampler=SequentialSampler(pred_dataset), batch_size=batch_size)

    model.eval()
    predictions, probabilities = [], []

    print("Generating predictions...")
    # Added tqdm here for the inference progress bar
    for batch in tqdm(pred_dataloader, desc="Inference"):
        b_input_ids, b_attn_masks = tuple(t.to(device) for t in batch)
        
        with torch.no_grad():
            outputs = model(b_input_ids, attention_mask=b_attn_masks)

        logits = outputs.logits
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        probs = torch.nn.functional.softmax(logits, dim=1).cpu().numpy()[:, 1]

        predictions.extend(preds)
        probabilities.extend(probs)

    df_full['bert_prediction'] = predictions
    df_full['bert_confidence'] = probabilities
    
    print("Inference completed successfully!")
    return df_full

    

def main():
    """Main pipeline orchestrator for BERT Fine-Tuning."""
    print("--- Starting BERT Training & Inference Pipeline ---")
    
    # Paths
    TRAIN_CSV = '../data/bert_training_data.csv'
    FULL_EXCEL = '../data/PoisonedOnly_NEISS_2004-2023.xlsx'
    OUTPUT_CSV = '../data/NEISS_Final_Classified.csv'
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")  # Apple Silicon GPU support
    else:
        device = torch.device("cpu")
        
    print(f"Using device: {device}")

    # 1. Load Data
    df_train, df_full = load_datasets(TRAIN_CSV, FULL_EXCEL)
    
    # 2. Tokenize and prepare loaders
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
    train_loader, val_loader = prepare_dataloaders(df_train, tokenizer)
    
    # 3. Build and Train Model
    model = build_lora_model(device)
    trained_model = train_model(model, train_loader, val_loader, device, epochs=5)
    
    # 4. Final Inference on the entire database
    df_final = run_inference_on_full_dataset(trained_model, df_full, tokenizer, device)
    
    # 5. Export results
    print(f"Saving fully classified dataset to {OUTPUT_CSV}...")
    df_final.to_csv(OUTPUT_CSV, index=False)
    print("--- Pipeline Completed ---")

if __name__ == '__main__':
    main()
# %%
