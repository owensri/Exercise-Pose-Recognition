import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical, plot_model
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import matplotlib.pyplot as plt

import config
from dataset_reader import DatasetReader

MAX_FRAMES = config.SEQUENCE_LENGTH
FEATURES = 48
CLASSES = config.CLASSES
NUM_CLASSES = len(CLASSES)

def build_lstm_1layer_model():
    model = Sequential()
    
    model.add(LSTM(64, return_sequences=False, input_shape=(MAX_FRAMES, FEATURES)))
    
    model.add(Dense(8, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Acc', color='blue')
    plt.plot(history.history['val_accuracy'], label='Val Acc', color='cornflowerblue')
    plt.title('1-Layer LSTM - Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss', color='blue')
    plt.plot(history.history['val_loss'], label='Val Loss', color='cornflowerblue')
    plt.title('1-Layer LSTM - Loss')
    plt.legend()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, 'training_history_lstm_1layer.png'))
    plt.close()

def evaluate_and_plot_cm(model, X_test, y_test):
    y_pred = np.argmax(model.predict(X_test), axis=1)
    y_true = np.argmax(y_test, axis=1)
    
    print("\n================ CLASSIFICATION REPORT (1-LAYER LSTM) ================")
    print(classification_report(y_true, y_pred, target_names=CLASSES))
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=CLASSES)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp.plot(cmap=plt.cm.Blues, ax=ax)
    plt.title("Confusion Matrix - 1-Layer LSTM (Test Set)")
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    plt.savefig(os.path.join(base_dir, 'confusion_matrix_lstm_1layer.png'))
    plt.close()

def main():
    print(f"\n[INFO] Starting 1-Layer LSTM Training -> Target Classes: {NUM_CLASSES}")
    
    reader = DatasetReader()
    X_train, X_val, X_test, y_train, y_val, y_test = reader.load_data_split()
    
    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val = to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)
    
    tf.random.set_seed(42)
    model = build_lstm_1layer_model()
    model.summary()
    
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
    plot_model(model, to_file=os.path.join(base_dir, 'model_structure_lstm_1layer.png'), show_shapes=True, show_layer_names=True)
    
    early_stop = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True)
    history = model.fit(
        X_train, y_train, 
        epochs=100, 
        batch_size=32, 
        validation_data=(X_val, y_val),
        callbacks=[early_stop]
    )
    
    print("\n[INFO] Final Evaluation on Unseen Test Set")
    loss, acc = model.evaluate(X_test, y_test)
    print(f"[RESULT] Test Accuracy (1-Layer LSTM): {acc * 100:.2f}%")
    
    model.save(os.path.join(base_dir, 'exercise_model_lstm_1layer.keras'))
    
    plot_training_history(history)
    evaluate_and_plot_cm(model, X_test, y_test)

if __name__ == "__main__":
    main()