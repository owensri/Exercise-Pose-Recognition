import csv
import os
import config

class CSVManager:
    def __init__(self, filepath):
        self.filepath         = filepath
        self.file             = None
        self.writer           = None
        self.is_header_written = False
        self.frame_count      = 1
        self.landmarks        = config.TARGET_LANDMARKS

        try:
            self.file   = open(self.filepath, 'w', newline='', encoding='utf-8')
            self.writer = csv.writer(self.file)
        except OSError as e:
            raise OSError(f"[CSVManager] ไม่สามารถเปิดไฟล์ได้: {self.filepath}\nสาเหตุ: {e}")

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
        return False

    def save_row(self, class_name, features):
        if self.file is None or self.file.closed:
            raise IOError(f"[CSVManager] ไฟล์ถูกปิดแล้ว: {self.filepath}")

        if not self.is_header_written:
            header = ['frame_number', 'class']
            for lm_id, name in self.landmarks:
                header.extend([
                    f'{name}_{lm_id}_x',
                    f'{name}_{lm_id}_y',
                    f'{name}_{lm_id}_z',
                    f'{name}_{lm_id}_v'
                ])
            self.writer.writerow(header)
            self.is_header_written = True

        self.writer.writerow([self.frame_count, class_name] + list(features))
        self.frame_count += 1

    def close(self):
        if self.file and not self.file.closed:
            self.file.close()