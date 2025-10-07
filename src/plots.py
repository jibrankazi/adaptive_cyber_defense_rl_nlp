import json, os
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, confusion_matrix, ConfusionMatrixDisplay

def plot_learning_curve(rewards, out="results/figures/learning_curve.png"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    plt.figure(); plt.plot(rewards); plt.title("Learning Curve"); plt.xlabel("Episode"); plt.ylabel("Reward"); plt.savefig(out); plt.close()

def plot_pr(y_true, y_score, out="results/figures/pr.png"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    p,r,_ = precision_recall_curve(y_true, y_score)
    plt.figure(); plt.plot(r,p); plt.title("Precision–Recall"); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.savefig(out); plt.close()

def plot_cm(y_true, y_pred, out="results/figures/cm.png"):
    os.makedirs(os.path.dirname(out), exist_ok=True)
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(cm)
    disp.plot()
    plt.title("Confusion Matrix"); plt.savefig(out); plt.close()
