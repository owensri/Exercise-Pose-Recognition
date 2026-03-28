import numpy as np
import config

class LandmarkFiller:
    def __init__(self, threshold=config.VISIBILITY_THRESHOLD):
        self.threshold = threshold

    def fill(self, all_frames):
        if not all_frames:
            return all_frames

        n_frames     = len(all_frames)
        n_landmarks  = len(config.TARGET_LANDMARKS)
        frames_array = np.array(all_frames)

        for lm_idx in range(n_landmarks):
            col_start = lm_idx * config.FEATURES_PER_LANDMARK
            col_x     = col_start + 0
            col_y     = col_start + 1
            col_z     = col_start + 2
            col_v     = col_start + 3

            visibility_all = frames_array[:, col_v]
            good_indices   = np.where(visibility_all >= self.threshold)[0]

            if len(good_indices) == 0 or len(good_indices) == n_frames:
                continue

            bad_indices       = np.where(visibility_all < self.threshold)[0]
            all_frame_indices = np.arange(n_frames)

            for col in [col_x, col_y, col_z]:
                good_values   = frames_array[good_indices, col]
                filled_values = np.interp(all_frame_indices, good_indices, good_values)
                frames_array[bad_indices, col] = filled_values[bad_indices]

        return frames_array.tolist()