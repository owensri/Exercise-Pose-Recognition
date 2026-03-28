import numpy as np
import config

class Preprocessor:
    def __init__(self):
        self.target_indices = config.TARGET_LANDMARK_INDICES

    def reset(self):
        pass

    def get_landmarks(self, results):
        if not results.pose_landmarks:
            return None
        lms = []
        for lm in results.pose_landmarks.landmark:
            lms.append((lm.x, lm.y, lm.z, lm.visibility))
        return lms

    def get_raw_values(self, landmarks):
        raw_data = []
        for idx in self.target_indices:
            raw_data.extend(landmarks[idx])
        return raw_data

    def normalize(self, landmarks):
        nose_x = landmarks[0][0]
        nose_y = landmarks[0][1]
        nose_z = landmarks[0][2]

        avg_toe_y   = (landmarks[31][1] + landmarks[32][1]) / 2.0
        body_height = avg_toe_y - nose_y

        if body_height <= 0.01:
            body_height = 0.01

        norm_data = []
        for idx in self.target_indices:
            x, y, z, v = landmarks[idx]
            nx = (x - nose_x) / body_height
            ny = (y - nose_y) / body_height
            nz = (z - nose_z) / body_height
            norm_data.extend([nx, ny, nz, v])

        return norm_data