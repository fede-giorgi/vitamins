import torch
import numpy as np
from tqdm import trange
from torchmetrics.classification import Recall, Accuracy, Precision

def train_model(model, train_dataloader, validation_dataloader, device, epochs=10):
    """Executes the training loop using standard cross-entropy on hybrid-sampled data."""
    # Lowering the learning rate to prevent gradient overshooting on cloned data
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-5)
    accuracy = Accuracy(task="binary").to(device)
    precision = Precision(task="binary").to(device)
    recall = Recall(task="binary").to(device)

    print("Starting Hybrid-Sampled training...")
    for epoch in trange(epochs, desc='Epoch'):
        model.train()
        tr_loss = 0
        
        for batch in train_dataloader:
            b_input_ids, b_attn_masks, b_labels = tuple(t.to(device) for t in batch)
            optimizer.zero_grad()
            
            # HuggingFace handles the standard loss automatically when labels are passed
            outputs = model(b_input_ids, attention_mask=b_attn_masks, labels=b_labels)
            loss = outputs.loss
            loss.backward()
            optimizer.step()
            tr_loss += loss.item()

        model.eval()
        val_acc, val_prec, val_rec = [], [], []
        
        for batch in validation_dataloader:
            b_input_ids, b_attn_masks, b_labels = tuple(t.to(device) for t in batch)
            with torch.no_grad():
                outputs = model(b_input_ids, attention_mask=b_attn_masks)
            
            preds = torch.argmax(outputs.logits, dim=1)
            val_acc.append(accuracy(preds, b_labels).item())
            val_prec.append(precision(preds, b_labels).item())
            val_rec.append(recall(preds, b_labels).item())

        print(f"\nEpoch {epoch+1} | Loss: {tr_loss/len(train_dataloader):.4f} | Val Acc: {np.mean(val_acc):.4f} | Val Prec: {np.mean(val_prec):.4f} | Val Rec: {np.mean(val_rec):.4f}")
    
    return model