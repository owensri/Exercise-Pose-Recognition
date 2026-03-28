import numpy as np

class FrameFiller:
    def fill(self, detected_frames, total_frames):
        if len(detected_frames) == 0:
            return []

        if len(detected_frames) == 1:
            return [detected_frames[0][1]] * total_frames

        known_indices = np.array([i for i, _ in detected_frames])
        known_values  = np.array([v for _, v in detected_frames])
        all_indices   = np.arange(total_frames)
        n_features    = known_values.shape[1]
        filled_array  = np.zeros((total_frames, n_features))

        for col in range(n_features):
            filled_array[:, col] = np.interp(
                all_indices, known_indices, known_values[:, col]
            )

        return filled_array.tolist()