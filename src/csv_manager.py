import csv
import os

class CSVManager:
    def __init__(self, filepath):
        self.filepath = filepath
        self.file = None
        self.writer = None
        self.is_header_written = False
        self.frame_count = 1

        # Map landmark ID with its descriptive name (MediaPipe Topology)
        # IMPORTANT: ลำดับนี้ต้องตรงกับ target_indices ใน preprocessor.py เสมอ
        # หากแก้ target_indices ใน Preprocessor ต้องอัปเดต list นี้ด้วย
        self.landmarks = [
            (11, "left_shoulder"), (12, "right_shoulder"),
            (13, "left_elbow"),    (14, "right_elbow"),
            (15, "left_wrist"),    (16, "right_wrist"),
            (23, "left_hip"),      (24, "right_hip"),
            (25, "left_knee"),     (26, "right_knee"),
            (27, "left_ankle"),    (28, "right_ankle")
        ]

        try:
            self.file = open(self.filepath, 'w', newline='', encoding='utf-8')
            self.writer = csv.writer(self.file)
        except OSError as e:
            raise OSError(f"[CSVManager] ไม่สามารถเปิดไฟล์ได้: {self.filepath}\nสาเหตุ: {e}")

    # รองรับ context manager: with CSVManager(...) as csv_mgr
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False  # ไม่ suppress exception

    def save_row(self, class_name, features):
        if self.file is None or self.file.closed:
            raise IOError(f"[CSVManager] ไฟล์ถูกปิดแล้ว ไม่สามารถเขียนข้อมูลได้: {self.filepath}")

        # Write header on the first row
        if not self.is_header_written:
            header = ['frame_number', 'class']

            # Generate descriptive column names e.g., 'left_shoulder_11_x'
            for lm_id, name in self.landmarks:
                header.extend([
                    f'{name}_{lm_id}_x',
                    f'{name}_{lm_id}_y',
                    f'{name}_{lm_id}_z',
                    f'{name}_{lm_id}_v'
                ])

            self.writer.writerow(header)
            self.is_header_written = True

        # Write data row
        row = [self.frame_count, class_name] + list(features)
        self.writer.writerow(row)

        # Increment frame counter for the next row
        self.frame_count += 1

    def close(self):
        if self.file and not self.file.closed:
            self.file.close()