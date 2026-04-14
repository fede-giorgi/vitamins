#%%

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import pandas as pd



def calculate_classification_metrics(y_true: pd.Series, y_pred: pd.Series, title: str = "Classification Report") -> None:
    """
    Calculates and prints standard ML metrics. 
    Useful for comparing LLM vs GT or Student vs Teacher.
    """
    print(f"\n--- {title} ---")
    print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
    print(f"Precision: {precision_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"Recall:    {recall_score(y_true, y_pred, zero_division=0):.4f}")
    print(f"F1-Score:  {f1_score(y_true, y_pred, zero_division=0):.4f}")
    
    print("\nDetailed Report:")
    print(classification_report(y_true, y_pred, digits=4, zero_division=0))
    
    cm = confusion_matrix(y_true, y_pred)
    print("Confusion Matrix:")
    print(f"TN: {cm[0][0]:<4} | FP: {cm[0][1]:<4}")
    print(f"FN: {cm[1][0]:<4} | TP: {cm[1][1]:<4}\n")