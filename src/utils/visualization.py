import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import roc_curve, auc, precision_recall_curve
import math

def plot_training_history(history):
    """
    Plot training history including loss and metrics over epochs.
    
    Parameters:
    history (dict): Dictionary containing training history metrics.
    
    Returns:
    None
    """
    plt.figure(figsize=(12, 8))

    # Plot loss
    plt.subplot(2, 1, 1)
    plt.plot(history['loss'], label='Training Loss')
    plt.plot(history.get('val_loss'), label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Training and Validation Loss')
    plt.legend()

    # Plot additional metrics
    plt.subplot(2, 1, 2)
    plt.plot(history['mse'], label='Training MSE')
    plt.plot(history.get('val_mse'), label='Validation MSE')
    plt.plot(history['mae'], label='Training MAE')
    plt.plot(history.get('val_mae'), label='Validation MAE')
    plt.xlabel('Epochs')
    plt.ylabel('Metrics')
    plt.title('Training and Validation Metrics')
    plt.legend()

    plt.tight_layout()
    plt.show()

def roc_fig_plot(y_true_l, **kwargs):
    """
    Plot ROC curves for multiple classifiers.
    
    Parameters:
    y_true_l (list): List of true labels.
    kwargs (dict): Dictionary of classifier names and their scores.
    
    Returns:
    None
    """
    plt.rcParams["figure.figsize"] = (10, 10)
    styles = ["-", "--"]
    colors = ["b", "g", "r", "c", "m", "y", "k", "orange"]

    for i, (key, value) in enumerate(kwargs.items()):
        assert len(y_true_l[i]) == len(value), "Length of true labels and scores must be equal"

        style = styles[i % 2]
        color_idx = math.floor(i / 2)
        color = colors[color_idx % len(colors)]

        fpr, tpr, thresholds = roc_curve(y_true_l[i], value)
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, style, label=f'AUC {key.replace("_", " ")} = {round(roc_auc, 3)}', color=color)

    plt.title('Receiver Operating Characteristic')
    plt.legend(loc='lower right')
    plt.plot([0, 1], [0, 1], 'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate (TPR)')
    plt.xlabel('False Positive Rate (FPR)')
    plt.show()

def precision_recall_fig_plot(y_true_l, **kwargs):
    """
    Plot Precision-Recall curves for multiple classifiers.
    
    Parameters:
    y_true_l (list): List of true labels.
    kwargs (dict): Dictionary of classifier names and their scores.
    
    Returns:
    None
    """
    plt.rcParams["figure.figsize"] = (10, 10)
    styles = ["-", "--"]
    colors = ["b", "g", "r", "c", "m", "y", "k", "orange"]

    for i, (key, value) in enumerate(kwargs.items()):
        assert len(y_true_l[i]) == len(value), "Length of true labels and scores must be equal"

        style = styles[i % 2]
        color_idx = math.floor(i / 2)
        color = colors[color_idx % len(colors)]

        precision, recall, thresholds = precision_recall_curve(y_true_l[i], value)
        prc_auc = auc(recall, precision)
        plt.plot(recall, precision, style, label=f'AUC {key.replace("_", " ")} = {round(prc_auc, 3)}', color=color)

    plt.title('Precision-Recall Curve')
    plt.legend(loc='lower right')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.show()
    
def print_dict(d, indent=0):
    for key, value in d.items():
        print('  ' * indent + str(key), end='')
        if isinstance(value, dict):
            print()
            print_dict(value, indent + 1)
        else:
            print(':' + ' ' * (20 - len(key) - 2 * indent) + str(value))