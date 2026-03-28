import os
import io
import argparse
import contextlib
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv1D, GlobalAveragePooling1D, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical, plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

import config
from dataset_reader import DatasetReader

def _get_log_path():
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    return os.path.join(base_dir, 'training_log_cnn_1layer.txt')

def _log(msg, log_path):
    print(msg)
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(msg + '\n')

MAX_FRAMES = config.SEQUENCE_LENGTH
FEATURES = config.NUM_FEATURES
CLASSES = config.CLASSES
NUM_CLASSES = len(CLASSES)

def build_cnn_1layer_model():
    model = Sequential()

    model.add(Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=(MAX_FRAMES, FEATURES)))
    model.add(GlobalAveragePooling1D())

    model.add(Dense(16, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc', color='deeppink')
    plt.plot(history.history['val_accuracy'], label='Val Acc', color='lightpink')
    plt.title('1D-CNN 1-Layer - Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='deeppink')
    plt.plot(history.history['val_loss'], label='Val Loss', color='lightpink')
    plt.title('1D-CNN 1-Layer - Loss')
    plt.legend()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'training_history_cnn_1layer.png'))
    plt.close()

def evaluate_and_plot_cm(model, X_test, y_test, log_path):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)

    report = classification_report(y_true, y_pred, target_names=CLASSES)
    _log(f"\n================ CLASSIFICATION REPORT (1D-CNN 1-Layer) ================", log_path)
    _log(report, log_path)

    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.RdPu, ax=ax)
    plt.title("Confusion Matrix - 1D-CNN 1-Layer (Test Set)")

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    plt.savefig(os.path.join(base_dir, 'confusion_matrix_cnn_1layer.png'))
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['early_stop', 'fixed'], default='early_stop',
        help="early_stop = หยุดเมื่อ val_loss ไม่ดีขึ้น (default) | fixed = เทรนครบ 50 epochs"
    )
    args = parser.parse_args()

    log_path = _get_log_path()
    open(log_path, 'w').close()
    _log(f"\n[INFO] Starting 1D-CNN 1-Layer Training -> Target Classes: {NUM_CLASSES}", log_path)
    _log(f"[INFO] Training mode: {args.mode}", log_path)

    reader = DatasetReader()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        X_train, X_val, X_test, y_train, y_val, y_test = reader.load_data_split()
    dataset_log = buf.getvalue()
    print(dataset_log, end='')
    with open(log_path, 'a', encoding='utf-8') as f:
        f.write(dataset_log)

    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val = to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

    tf.random.set_seed(42)
    model = build_cnn_1layer_model()
    model.summary()

    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    plot_model(model, to_file=os.path.join(base_dir, 'model_structure_cnn_1layer.png'), show_shapes=True, show_layer_names=True)

    if args.mode == 'early_stop':
        epochs    = 100
        callbacks = [EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)]
    else:
        epochs    = 50
        callbacks = []

    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=32,
        validation_data=(X_val, y_val),
        callbacks=callbacks
    )

    _log("\n[INFO] Final Evaluation on Unseen Test Set", log_path)
    loss, acc = model.evaluate(X_test, y_test)
    _log(f"[RESULT] Test Accuracy (1D-CNN 1-Layer): {acc * 100:.2f}%", log_path)

    model.save(os.path.join(base_dir, 'exercise_model_cnn_1layer.keras'))

    plot_training_history(history)
    evaluate_and_plot_cm(model, X_test, y_test, log_path)

if __name__ == "__main__":
    main()