import os
import io
import argparse
import contextlib
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import to_categorical

import config
from dataset_reader import DatasetReader
import model_utils

MAX_FRAMES = config.SEQUENCE_LENGTH
FEATURES = config.NUM_FEATURES
CLASSES = config.CLASSES
NUM_CLASSES = len(CLASSES)

def build_lstm_1layer_model():
    model = Sequential()

    model.add(LSTM(64, return_sequences=False, input_shape=(MAX_FRAMES, FEATURES)))

    model.add(Dense(8, activation='relu'))
    model.add(Dense(NUM_CLASSES, activation='softmax'))
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    return model

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--mode', choices=['early_stop', 'fixed'], default='early_stop'
    )
    args = parser.parse_args()

    log_path, models_dir, logs_dir = model_utils.setup_logs_and_models_dir('lstm_1layer')
    
    model_utils.write_log(f"\n[INFO] Starting 1-Layer LSTM Training -> Target Classes: {NUM_CLASSES}", log_path)
    model_utils.write_log(f"[INFO] Training mode: {args.mode}", log_path)

    reader = DatasetReader()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        X_train, X_val, X_test, y_train, y_val, y_test = reader.load_data_split()
    model_utils.write_log(buf.getvalue(), log_path)

    y_train = to_categorical(y_train, num_classes=NUM_CLASSES)
    y_val = to_categorical(y_val, num_classes=NUM_CLASSES)
    y_test = to_categorical(y_test, num_classes=NUM_CLASSES)

    tf.random.set_seed(42)
    model = build_lstm_1layer_model()
    model.summary()

    model_utils.save_model_structure(model, logs_dir, 'lstm_1layer')

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

    model_utils.write_log("\n[INFO] Final Evaluation on Unseen Test Set", log_path)
    loss, acc = model.evaluate(X_test, y_test)
    model_utils.write_log(f"[RESULT] Test Accuracy (1-Layer LSTM): {acc * 100:.2f}%", log_path)

    model.save(os.path.join(models_dir, 'exercise_model_lstm_1layer.keras'))

    model_utils.plot_and_save_history(history, logs_dir, 'lstm_1layer', '1-Layer LSTM', 'blue', 'cornflowerblue')
    model_utils.evaluate_and_save_cm(model, X_test, y_test, CLASSES, log_path, logs_dir, 'lstm_1layer', '1-Layer LSTM', 'Blues')

if __name__ == "__main__":
    main()