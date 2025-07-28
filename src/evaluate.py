import torch
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
import os

def evaluate(model, loader, device, class_names, export_csv=False, split_name="test"):
    model.eval()
    all_preds, all_labels = [], []

    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            preds = model(x)
            all_preds.extend(preds.argmax(1).cpu().numpy())
            all_labels.extend(y.cpu().numpy())

    # ✅ Use the correct variable names
    report = classification_report(all_labels, all_preds, target_names=class_names, zero_division=0)
    print(report)

    # ✅ Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)
    plot_confusion_matrix(cm, class_names, split_name)

    if export_csv:
        df = pd.DataFrame({
            "True_Label": [class_names[l] for l in all_labels],
            "Predicted_Label": [class_names[p] for p in all_preds],
        })
        os.makedirs("outputs", exist_ok=True)
        out_path = f"outputs/{split_name}_predictions.csv"
        df.to_csv(out_path, index=False)
        print(f"\n📁 Saved prediction CSV to: {out_path}")

    return report


def plot_confusion_matrix(cm, class_names, split_name=""):
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.title(f"Confusion Matrix - {split_name.capitalize()} Set")
    plt.tight_layout()
    os.makedirs("outputs", exist_ok=True)
    plt.savefig(f"outputs/confusion_matrix_{split_name}.png")
    plt.show()
