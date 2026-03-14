import os

# Directory settings
DATASET_FOLDER = "dataset"
RAW_CSV_FOLDER = "output_csv_raw"
NORM_CSV_FOLDER = "output_csv_norm"
REJECTED_FOLDER = "rejected_clips"

# Dataset settings
CLASSES = ["pushup", "squat", "lunge", "other"]
#CLASSES = ["other"]

# Processing & Resampling settings
TARGET_FPS = 30
TARGET_VIDEO_FRAMES = 120  # Interpolate all videos to exactly 120 frames
SEQUENCE_LENGTH = 30       # Window size (e.g., frame 0-29)
WINDOW_STEP_SIZE = 15      # Overlap by half (e.g., frame 15-44)

# MIN_FRAME_COUNT = 1 คือ ให้ทุกคลิปผ่านได้ (อ่านทุกเฟรม)
# การกรองจริงทำที่ MIN_DETECTED_FRAMES แทน:
#   ต้องมีเฟรมที่ detect pose ได้จริงอย่างน้อยเท่านี้ จึงจะบันทึก CSV
#   ป้องกัน CSV ว่างเปล่าจากคลิปที่ไม่มี pose เลย
MIN_FRAME_COUNT = 1
MIN_DETECTED_FRAMES = 30   # ต้องมีเฟรม pose จริงอย่างน้อย 30 เฟรม

# Frame Padding settings
PADDING_WINDOW = 30        # ใช้ค่าเฉลี่ยจากกี่เฟรมก่อนหน้าในการ pad

# Model settings
MODEL_PATH = "exercise_model.keras"