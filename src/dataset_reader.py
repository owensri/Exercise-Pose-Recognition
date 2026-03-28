import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from collections import Counter
import config

class DatasetReader:
    def __init__(self, data_path="../output_csv_norm"):
        base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
        self.data_path = os.path.join(base_dir, data_path.replace('../', ''))

        self.classes       = config.CLASSES
        self.target_frames = config.TARGET_VIDEO_FRAMES
        self.window_size   = config.SEQUENCE_LENGTH
        self.step_size     = config.WINDOW_STEP_SIZE

    def get_person_id(self, class_name, filename):
        prefix = filename.split('_')[0]
        return f"{class_name}_{prefix}"

    def read_csv_file(self, file_path):
        df = pd.read_csv(file_path)
        cols_to_drop = [c for c in ['frame_number', 'class'] if c in df.columns]
        return df.drop(columns=cols_to_drop, errors='ignore').values

    def interpolate_frames(self, video_data):
        num_frames = len(video_data)
        if num_frames == self.target_frames:
            return video_data

        orig_indices   = np.linspace(0, num_frames - 1, num=num_frames)
        target_indices = np.linspace(0, num_frames - 1, num=self.target_frames)

        resampled = []
        for col in range(video_data.shape[1]):
            col_data = np.interp(target_indices, orig_indices, video_data[:, col])
            resampled.append(col_data)

        return np.column_stack(resampled)

    def read_dataset(self):
        X_videos   = []
        y_labels   = []
        person_ids = []

        for label_idx, class_name in enumerate(self.classes):
            class_path = os.path.join(self.data_path, class_name)
            if not os.path.exists(class_path):
                continue

            csv_files = sorted([f for f in os.listdir(class_path) if f.endswith('.csv')])

            for file_name in csv_files:
                file_path        = os.path.join(class_path, file_name)
                features_data    = self.read_csv_file(file_path)
                normalized_video = self.interpolate_frames(features_data)

                X_videos.append(normalized_video)
                y_labels.append(label_idx)
                person_ids.append(self.get_person_id(class_name, file_name))

        return np.array(X_videos), np.array(y_labels), person_ids

    def split_by_person(self, person_ids, y_labels, random_state):
        train_persons = set()
        val_persons   = set()
        test_persons  = set()

        for label_idx, class_name in enumerate(self.classes):
            class_persons = list(dict.fromkeys(
                pid for pid, lbl in zip(person_ids, y_labels) if lbl == label_idx
            ))

            train_p, temp_p = train_test_split(
                class_persons, test_size=0.30, random_state=random_state
            )
            val_p, test_p = train_test_split(
                temp_p, test_size=0.50, random_state=random_state
            )

            train_persons.update(train_p)
            val_persons.update(val_p)
            test_persons.update(test_p)

        return train_persons, val_persons, test_persons

    def create_sliding_windows(self, video_data):
        windows = []
        for start_idx in range(0, self.target_frames - self.window_size + 1, self.step_size):
            end_idx = start_idx + self.window_size
            windows.append(video_data[start_idx:end_idx])
        return windows

    def process_split(self, X_video_split, y_video_split):
        X_final, y_final = [], []
        for video, label in zip(X_video_split, y_video_split):
            windows = self.create_sliding_windows(video)
            for w in windows:
                X_final.append(w)
                y_final.append(label)
        return np.array(X_final), np.array(y_final)

    def load_data_split(self, random_state=42):
        print(f"\n[INFO] Loading and Interpolating Videos to {self.target_frames} frames...")
        X_vids, y_vids, person_ids = self.read_dataset()

        train_persons, val_persons, test_persons = self.split_by_person(
            person_ids, y_vids, random_state
        )

        print(f"[INFO] Person-level split:")
        print(f"  Train persons : {len(train_persons)}")
        print(f"  Val persons   : {len(val_persons)}")
        print(f"  Test persons  : {len(test_persons)}")

        X_train_vids, y_train_vids = [], []
        X_val_vids,   y_val_vids   = [], []
        X_test_vids,  y_test_vids  = [], []

        for video, label, pid in zip(X_vids, y_vids, person_ids):
            if pid in train_persons:
                X_train_vids.append(video)
                y_train_vids.append(label)
            elif pid in val_persons:
                X_val_vids.append(video)
                y_val_vids.append(label)
            elif pid in test_persons:
                X_test_vids.append(video)
                y_test_vids.append(label)

        print(f"[INFO] Slicing videos into {self.window_size}-frame windows...")
        X_train, y_train = self.process_split(X_train_vids, y_train_vids)
        X_val,   y_val   = self.process_split(X_val_vids,   y_val_vids)
        X_test,  y_test  = self.process_split(X_test_vids,  y_test_vids)

        print(f"[SUCCESS] Final Data Shapes:")
        print(f"  Train shape : {X_train.shape}")
        print(f"  Val shape   : {X_val.shape}")
        print(f"  Test shape  : {X_test.shape}")
        print(f"  Test class distribution: {Counter(y_test.tolist())}")

        return X_train, X_val, X_test, y_train, y_val, y_test