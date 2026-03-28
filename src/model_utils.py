import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
from tensorflow.keras.utils import plot_model

def setup_logs_and_models_dir(model_name):
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    logs_dir = os.path.join(base_dir, 'models_logs')
    models_dir = os.path.join(base_dir, 'models')
    os.makedirs(logs_dir, exist_ok=True)
    os.makedirs(models_dir, exist_ok=True)
    
    log_path = os.path.join(logs_dir, f'training_log_{model_name}.txt')
    open(log_path, 'w').close()
    
    return log_path, models_dir, logs_dir

def write_log(msg, log_path):
    print(msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

def save_model_structure(model, logs_dir, model_name):
    plot_path = os.path.join(logs_dir, f'model_structure_{model_name}.png')
    plot_model(model, to_file=plot_path, show_shapes=True, show_layer_names=True)

def plot_and_save_history(history, logs_dir, model_name, title_prefix, color_train, color_val):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc', color=color_train)
    plt.plot(history.history['val_accuracy'], label='Val Acc', color=color_val)
    plt.title(f'{title_prefix} - Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color=color_train)
    plt.plot(history.history['val_loss'], label='Val Loss', color=color_val)
    plt.title(f'{title_prefix} - Loss')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(logs_dir, f'training_history_{model_name}.png'))
    plt.close()

def evaluate_and_save_cm(model, X_test, y_test, classes, log_path, logs_dir, model_name, title_prefix, cmap):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred, target_names=classes)
    write_log(f"\n================ CLASSIFICATION REPORT ({title_prefix}) ================", log_path)
    write_log(report, log_path)
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=cmap, ax=ax)
    plt.title(f"Confusion Matrix - {title_prefix} (Test Set)")
    
    plt.savefig(os.path.join(logs_dir, f'confusion_matrix_{model_name}.png'))
    plt.close()