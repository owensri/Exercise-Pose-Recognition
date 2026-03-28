import os

DATASET_FOLDER  = "dataset"
RAW_CSV_FOLDER  = "output_csv_raw"
NORM_CSV_FOLDER = "output_csv_norm"
REJECTED_FOLDER = "rejected_clips"

CLASSES = ["pushup", "squat", "lunge", "other"]

TARGET_FPS          = 30
TARGET_VIDEO_FRAMES = 120
SEQUENCE_LENGTH     = 30
WINDOW_STEP_SIZE    = 15

TARGET_LANDMARKS = [
    (11, "left_shoulder"), (12, "right_shoulder"),
    (13, "left_elbow"),    (14, "right_elbow"),
    (15, "left_wrist"),    (16, "right_wrist"),
    (23, "left_hip"),      (24, "right_hip"),
    (25, "left_knee"),     (26, "right_knee"),
    (27, "left_ankle"),    (28, "right_ankle"),
]
TARGET_LANDMARK_INDICES = [idx for idx, _ in TARGET_LANDMARKS]
FEATURES_PER_LANDMARK   = 4
NUM_FEATURES            = len(TARGET_LANDMARKS) * FEATURES_PER_LANDMARK

MIN_FRAME_COUNT     = 1
MIN_DETECTED_FRAMES = 30
PADDING_WINDOW      = 30
VISIBILITY_THRESHOLD = 0.5

MODEL_PATH = "exercise_model.keras"