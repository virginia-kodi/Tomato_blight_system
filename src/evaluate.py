import os
import sys
import tensorflow as tf
import numpy as np

# Ensure sibling modules in src/ and parent config are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data_loader import get_data_generators
import logger
import config



def evaluate_model():
    """Evaluates the trained model on the test dataset and logs all metrics."""
    _, _, test_ds, class_names, _ = get_data_generators()
    
    model_path = os.path.join(config.MODELS_DIR, 'model_v1.h5')
    if not os.path.exists(model_path):
        print(f"Model not found at {model_path}. Please train first.")
        return
        
    print("Loading Model...")
    model = tf.keras.models.load_model(model_path)
    
    print("Evaluating Model...")
    # Gather true labels and predictions
    y_true = []
    y_pred_probs = []
    
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_probs.extend(preds)
        y_true.extend(labels.numpy())
        
    y_true = np.array(y_true)
    y_pred_probs = np.array(y_pred_probs).flatten()
    y_pred = (y_pred_probs > 0.5).astype(int)
    
    # Calculate Metrics
    from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)
    
    metrics_dict = {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }
    
    print("Saving Metrics and Plots...")
    logger.save_metrics(metrics_dict)
    logger.save_confusion_matrix(y_true, y_pred, class_names)
    logger.save_classification_report(y_true, y_pred, class_names)
    logger.save_roc_curve(y_true, y_pred_probs, class_names)
    logger.save_pr_curve(y_true, y_pred_probs, class_names)
    logger.save_prediction_distribution(y_pred_probs, class_names)
    
    summary = f"Experiment Evaluation Summary\n"
    summary += f"Accuracy: {accuracy:.4f}\n"
    summary += f"Precision: {precision:.4f}\n"
    summary += f"Recall: {recall:.4f}\n"
    summary += f"F1-Score: {f1:.4f}\n"
    logger.save_experiment_summary(summary)
    
    print("Evaluation Complete. Results saved in metrics/ plots/ and reports/")

if __name__ == '__main__':
    evaluate_model()
