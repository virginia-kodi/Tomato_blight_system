import os
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_curve, auc, precision_recall_curve, average_precision_score
)
import seaborn as sns
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import config

# ==============================================================================
# Dissertation-Quality Style Configuration
# ==============================================================================

# Landscape chart dimensions: 3000 x 1800 px
LANDSCAPE_FIGSIZE = (10, 6)
# Square chart dimensions: 2400 x 2400 px
SQUARE_FIGSIZE = (8, 8)
# Resolution
DPI = 300

# Professional colour palette
COLORS = {
    'train': '#1B4F72',       # Deep navy blue
    'val': '#E74C3C',         # Vibrant red
    'primary': '#2C3E50',     # Dark charcoal
    'secondary': '#E67E22',   # Warm orange
    'accent': '#27AE60',      # Emerald green
    'grid': '#D5D8DC',        # Soft grey
    'background': '#FDFEFE',  # Near-white
}


def _apply_dissertation_style():
    """Applies global matplotlib rcParams for dissertation-quality figures."""
    plt.rcParams.update({
        'font.family': 'serif',
        'font.size': 12,
        'axes.titlesize': 16,
        'axes.titleweight': 'bold',
        'axes.labelsize': 14,
        'axes.labelweight': 'bold',
        'axes.grid': True,
        'axes.facecolor': COLORS['background'],
        'grid.alpha': 0.4,
        'grid.linestyle': '--',
        'grid.color': COLORS['grid'],
        'legend.fontsize': 11,
        'legend.framealpha': 0.9,
        'legend.edgecolor': '#CCCCCC',
        'xtick.labelsize': 11,
        'ytick.labelsize': 11,
        'lines.linewidth': 2.2,
        'lines.markersize': 6,
        'figure.facecolor': 'white',
        'savefig.facecolor': 'white',
        'savefig.bbox': 'tight',
        'savefig.pad_inches': 0.15,
    })


# Apply style on import
_apply_dissertation_style()


# ==============================================================================
# Training History Persistence
# ==============================================================================

def save_training_history(history):
    """Saves the raw training history dict to JSON for future reference.

    This ensures per-epoch metrics are never lost and can be re-plotted
    without retraining.
    """
    history_dict = {}
    for key, values in history.history.items():
        history_dict[key] = [float(v) for v in values]

    path = os.path.join(config.METRICS_DIR, 'training_history.json')
    with open(path, 'w') as f:
        json.dump(history_dict, f, indent=4)
    print(f"  -> Training history saved to {path}")


# ==============================================================================
# Training Curve Plots (Landscape: 10x6 @ 300 DPI)
# ==============================================================================

def _plot_training_curve(epochs, train_vals, val_vals, title, ylabel, filename):
    """Helper to plot a single training vs validation curve."""
    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)

    ax.plot(epochs, train_vals, color=COLORS['train'], marker='o',
            markersize=5, label=f'Training {ylabel}', linewidth=2.2)
    ax.plot(epochs, val_vals, color=COLORS['val'], marker='s',
            markersize=5, label=f'Validation {ylabel}', linewidth=2.2)

    ax.set_title(title)
    ax.set_xlabel('Epoch')
    ax.set_ylabel(ylabel)
    ax.legend(loc='best')
    ax.set_xticks(epochs)

    filepath = os.path.join(config.PLOTS_DIR, filename)
    fig.savefig(filepath, dpi=DPI)
    plt.close(fig)
    print(f"  -> Saved: {filepath}")


def save_training_curves(history):
    """Saves all per-epoch training curves: Accuracy, Loss, Precision, Recall, F1."""
    h = history.history
    epochs = list(range(1, len(h['accuracy']) + 1))

    # 1. Accuracy Curve
    _plot_training_curve(
        epochs, h['accuracy'], h['val_accuracy'],
        'Training and Validation Accuracy', 'Accuracy',
        'accuracy_curve.png'
    )

    # 2. Loss Curve
    _plot_training_curve(
        epochs, h['loss'], h['val_loss'],
        'Training and Validation Loss', 'Loss',
        'loss_curve.png'
    )

    # 3. Precision Curve
    _plot_training_curve(
        epochs, h['precision'], h['val_precision'],
        'Training and Validation Precision', 'Precision',
        'precision_curve.png'
    )

    # 4. Recall Curve
    _plot_training_curve(
        epochs, h['recall'], h['val_recall'],
        'Training and Validation Recall', 'Recall',
        'recall_curve.png'
    )

    # 5. F1 Curve (computed from Precision and Recall)
    train_p = np.array(h['precision'])
    train_r = np.array(h['recall'])
    val_p = np.array(h['val_precision'])
    val_r = np.array(h['val_recall'])

    # F1 = 2 * (P * R) / (P + R), handle division by zero
    train_f1 = np.where(
        (train_p + train_r) > 0,
        2 * (train_p * train_r) / (train_p + train_r),
        0.0
    )
    val_f1 = np.where(
        (val_p + val_r) > 0,
        2 * (val_p * val_r) / (val_p + val_r),
        0.0
    )

    _plot_training_curve(
        epochs, train_f1.tolist(), val_f1.tolist(),
        'Training and Validation F1 Score', 'F1 Score',
        'f1_curve.png'
    )


# ==============================================================================
# Confusion Matrix (Square: 8x8 @ 300 DPI)
# ==============================================================================

def save_confusion_matrix(y_true, y_pred, class_names):
    """Saves the confusion matrix as a dissertation-quality square heatmap."""
    cm = confusion_matrix(y_true, y_pred)

    # Calculate percentages for annotation
    cm_percent = cm.astype('float') / cm.sum(axis=1, keepdims=True) * 100
    annot_labels = np.array([
        [f"{count}\n({pct:.1f}%)" for count, pct in zip(row_c, row_p)]
        for row_c, row_p in zip(cm, cm_percent)
    ])

    fig, ax = plt.subplots(figsize=SQUARE_FIGSIZE)
    sns.heatmap(
        cm, annot=annot_labels, fmt='', cmap='Blues',
        xticklabels=class_names, yticklabels=class_names,
        linewidths=1.5, linecolor='white',
        cbar_kws={'label': 'Count', 'shrink': 0.8},
        annot_kws={'size': 14, 'fontweight': 'bold'},
        ax=ax
    )
    ax.set_xlabel('Predicted Label')
    ax.set_ylabel('True Label')
    ax.set_title('Confusion Matrix')

    filepath = os.path.join(config.PLOTS_DIR, 'confusion_matrix.png')
    fig.savefig(filepath, dpi=DPI)
    plt.close(fig)
    print(f"  -> Saved: {filepath}")


# ==============================================================================
# ROC Curve (Landscape: 10x6 @ 300 DPI)
# ==============================================================================

def save_roc_curve(y_true, y_pred_probs, class_names):
    """Saves the Receiver Operating Characteristic curve with AUC."""
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)

    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    ax.plot(fpr, tpr, color=COLORS['train'], linewidth=2.5,
            label=f'ROC Curve (AUC = {roc_auc:.4f})')
    ax.plot([0, 1], [0, 1], color=COLORS['grid'], linestyle='--',
            linewidth=1.5, label='Random Classifier')

    ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS['train'])

    ax.set_title('Receiver Operating Characteristic (ROC) Curve')
    ax.set_xlabel('False Positive Rate')
    ax.set_ylabel('True Positive Rate')
    ax.legend(loc='lower right')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    filepath = os.path.join(config.PLOTS_DIR, 'roc_curve.png')
    fig.savefig(filepath, dpi=DPI)
    plt.close(fig)
    print(f"  -> Saved: {filepath}")


# ==============================================================================
# Precision-Recall Curve (Landscape: 10x6 @ 300 DPI)
# ==============================================================================

def save_pr_curve(y_true, y_pred_probs, class_names):
    """Saves the Precision-Recall curve with Average Precision score."""
    precision_vals, recall_vals, _ = precision_recall_curve(y_true, y_pred_probs)
    ap = average_precision_score(y_true, y_pred_probs)

    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    ax.plot(recall_vals, precision_vals, color=COLORS['secondary'], linewidth=2.5,
            label=f'PR Curve (AP = {ap:.4f})')

    ax.fill_between(recall_vals, precision_vals, alpha=0.15, color=COLORS['secondary'])

    ax.set_title('Precision-Recall (PR) Curve')
    ax.set_xlabel('Recall')
    ax.set_ylabel('Precision')
    ax.legend(loc='lower left')
    ax.set_xlim([-0.02, 1.02])
    ax.set_ylim([-0.02, 1.02])

    filepath = os.path.join(config.PLOTS_DIR, 'pr_curve.png')
    fig.savefig(filepath, dpi=DPI)
    plt.close(fig)
    print(f"  -> Saved: {filepath}")


# ==============================================================================
# Prediction Distribution (Landscape: 10x6 @ 300 DPI)
# ==============================================================================

def save_prediction_distribution(y_pred_probs, class_names):
    """Saves a histogram of predicted probability scores."""
    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)

    ax.hist(y_pred_probs, bins=50, color=COLORS['train'], edgecolor='white',
            linewidth=0.8, alpha=0.85)

    ax.axvline(x=0.5, color=COLORS['val'], linestyle='--', linewidth=2,
               label='Decision Threshold (0.5)')

    ax.set_title('Prediction Probability Distribution')
    ax.set_xlabel('Predicted Probability (Positive Class)')
    ax.set_ylabel('Frequency')
    ax.legend(loc='upper center')

    filepath = os.path.join(config.PLOTS_DIR, 'prediction_distribution.png')
    fig.savefig(filepath, dpi=DPI)
    plt.close(fig)
    print(f"  -> Saved: {filepath}")


# ==============================================================================
# Dataset Split Chart (Landscape: 10x6 @ 300 DPI)
# ==============================================================================

def save_dataset_split_chart(train_size, val_size, test_size):
    """Saves a bar chart showing the train/validation/test split proportions."""
    labels = ['Training', 'Validation', 'Test']
    sizes = [train_size, val_size, test_size]
    bar_colors = [COLORS['train'], COLORS['secondary'], COLORS['accent']]

    total = sum(sizes)
    percentages = [(s / total) * 100 for s in sizes]

    fig, ax = plt.subplots(figsize=LANDSCAPE_FIGSIZE)
    bars = ax.bar(labels, sizes, color=bar_colors, edgecolor='white',
                  linewidth=1.5, width=0.55)

    # Add count and percentage labels on each bar
    for bar, count, pct in zip(bars, sizes, percentages):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + (total * 0.01),
                f'{count}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=12, fontweight='bold')

    ax.set_title('Dataset Split Distribution')
    ax.set_ylabel('Number of Batches')

    filepath = os.path.join(config.PLOTS_DIR, 'dataset_split.png')
    fig.savefig(filepath, dpi=DPI)
    plt.close(fig)
    print(f"  -> Saved: {filepath}")


# ==============================================================================
# Metrics & Reports (unchanged logic, kept for completeness)
# ==============================================================================

def save_metrics(metrics_dict):
    """Saves numerical metrics as a JSON file."""
    with open(os.path.join(config.METRICS_DIR, 'metrics.json'), 'w') as f:
        json.dump(metrics_dict, f, indent=4)

def save_classification_report(y_true, y_pred, class_names):
    """Saves the detailed classification report."""
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(os.path.join(config.METRICS_DIR, 'classification_report.txt'), 'w') as f:
        f.write(report)

def save_experiment_summary(summary_text):
    """Saves experiment summary."""
    with open(os.path.join(config.REPORTS_DIR, 'experiment_summary.txt'), 'w') as f:
        f.write(summary_text)
