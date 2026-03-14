import os
import numpy as np
import cv2
import config
from pose_estimator import PoseDetector
from preprocessor import Preprocessor
from interpolator import Interpolator
from image_processor import ImageProcessor
from dataset_loader import DatasetLoader
from csv_manager import CSVManager
from smoother import LandmarkSmoother


def compute_rolling_average_pad(history, window=config.PADDING_WINDOW):
    """
    คำนวณค่าเฉลี่ยจากเฟรมก่อนหน้าสูงสุด `window` เฟรม
    ใช้สำหรับ padding เมื่อ pose detect ไม่ได้ในเฟรมปัจจุบัน
    - ถ้า history ว่าง → คืน None (ยังไม่มีข้อมูลเลย ข้ามเฟรมนี้ไป)
    - ถ้ามีข้อมูลแล้ว → คืนค่าเฉลี่ยจาก min(window, len(history)) เฟรมล่าสุด
    """
    if not history:
        return None
    recent = history[-window:]
    return list(np.mean(recent, axis=0))


def save_sequence(data, detected_count, original_fps, dir_raw, dir_norm, class_name, fname, is_flip, interpolator):
    suffix = "_flipped" if is_flip else ""

    # กรองคลิปที่ detect pose ได้จริงน้อยเกินไป
    # (MIN_FRAME_COUNT=1 ยังคงให้ผ่านทุกคลิป แต่ต้อง detect ได้อย่างน้อย MIN_DETECTED_FRAMES เฟรม)
    if detected_count < config.MIN_DETECTED_FRAMES:
        return False, f"pose detected เพียง {detected_count} เฟรม (ต้องการ >= {config.MIN_DETECTED_FRAMES})"

    if len(data['norm']) < config.MIN_FRAME_COUNT:
        return False, f"จำนวนเฟรมรวม {len(data['norm'])} น้อยกว่า MIN_FRAME_COUNT={config.MIN_FRAME_COUNT}"

    # Resample to target FPS
    resamp_raw  = interpolator.process(data['raw'],  original_fps, config.TARGET_FPS)
    resamp_norm = interpolator.process(data['norm'], original_fps, config.TARGET_FPS)

    try:
        # Save Raw Data
        with CSVManager(os.path.join(dir_raw, f"{fname}{suffix}.csv")) as csv_raw:
            for r in resamp_raw:
                csv_raw.save_row(class_name, r)

        # Save Normalized Data
        with CSVManager(os.path.join(dir_norm, f"{fname}{suffix}.csv")) as csv_norm:
            for r in resamp_norm:
                csv_norm.save_row(class_name, r)

    except OSError as e:
        return False, f"ไม่สามารถบันทึกไฟล์ได้: {e}"

    return True, None


def run():
    print(f"\n[INFO] Starting Extraction Pipeline")
    print(f"[INFO] Target Classes: {', '.join(config.CLASSES)}")

    # Ensure directories exist
    for folder in [config.RAW_CSV_FOLDER, config.NORM_CSV_FOLDER, config.REJECTED_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    loader        = DatasetLoader()
    interpolator  = Interpolator()
    img_processor = ImageProcessor()

    proc_orig,   proc_flip   = Preprocessor(),       Preprocessor()
    smooth_orig, smooth_flip = LandmarkSmoother(),   LandmarkSmoother()

    rejected_list  = []
    total_videos   = 0
    success_orig_count = 0
    success_flip_count = 0

    for video_path, class_name, video_file in loader.get_video_files():
        total_videos += 1
        print(f"\n[{total_videos}] Processing: {class_name}/{video_file}")

        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"  [ERROR] Cannot open file")
            rejected_list.append(f"{class_name}/{video_file} - File error")
            continue

        # สร้าง detector หลังจากเช็ค cap สำเร็จแล้วเท่านั้น → ป้องกัน memory leak
        detector_orig = PoseDetector()
        detector_flip = PoseDetector()

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if not original_fps or original_fps <= 0:
            print(f"  [WARN] FPS ไม่สามารถอ่านได้ ใช้ค่า fallback = 30")
            original_fps = 30

        proc_orig.reset();   proc_flip.reset()
        smooth_orig.reset(); smooth_flip.reset()

        data_orig = {'raw': [], 'norm': []}
        data_flip = {'raw': [], 'norm': []}

        # History สำหรับ rolling average padding แยกต่างหากจาก data (ไม่รวม padded frames)
        raw_history_orig,  norm_history_orig  = [], []
        raw_history_flip,  norm_history_flip  = [], []

        detected_orig = 0
        detected_flip = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # --- Original Frame ---
            rgb = img_processor.convert_to_rgb(frame)
            res = detector_orig.find_pose(rgb)
            lms = proc_orig.get_landmarks(res)

            if lms:
                lms = smooth_orig.process(lms)
                raw_vals  = proc_orig.get_raw_values(lms)
                norm_vals = proc_orig.normalize(lms)
                data_orig['raw'].append(raw_vals)
                data_orig['norm'].append(norm_vals)
                raw_history_orig.append(raw_vals)
                norm_history_orig.append(norm_vals)
                detected_orig += 1
            else:
                # Padding: ใช้ค่าเฉลี่ยจากเฟรมก่อนหน้า (rolling average)
                pad_raw  = compute_rolling_average_pad(raw_history_orig)
                pad_norm = compute_rolling_average_pad(norm_history_orig)
                if pad_raw is not None and pad_norm is not None:
                    data_orig['raw'].append(pad_raw)
                    data_orig['norm'].append(pad_norm)
                # ถ้า history ว่างอยู่ (ไม่เคย detect ได้เลย) → ข้ามเฟรมนี้ไปเลย ไม่ pad

            # --- Flipped Frame ---
            frame_flip = img_processor.flip_horizontal(frame)
            rgb_flip   = img_processor.convert_to_rgb(frame_flip)
            res_flip   = detector_flip.find_pose(rgb_flip)
            lms_flip   = proc_flip.get_landmarks(res_flip)

            if lms_flip:
                lms_flip  = smooth_flip.process(lms_flip)
                raw_vals  = proc_flip.get_raw_values(lms_flip)
                norm_vals = proc_flip.normalize(lms_flip)
                data_flip['raw'].append(raw_vals)
                data_flip['norm'].append(norm_vals)
                raw_history_flip.append(raw_vals)
                norm_history_flip.append(norm_vals)
                detected_flip += 1
            else:
                # Padding: ใช้ค่าเฉลี่ยจากเฟรมก่อนหน้า (rolling average)
                pad_raw  = compute_rolling_average_pad(raw_history_flip)
                pad_norm = compute_rolling_average_pad(norm_history_flip)
                if pad_raw is not None and pad_norm is not None:
                    data_flip['raw'].append(pad_raw)
                    data_flip['norm'].append(pad_norm)
                # ถ้า history ว่างอยู่ (ไม่เคย detect ได้เลย) → ข้ามเฟรมนี้ไปเลย ไม่ pad

        # Free resources
        cap.release()
        detector_orig.pose.close()
        detector_flip.pose.close()

        dir_raw  = os.path.join(config.RAW_CSV_FOLDER,  class_name)
        dir_norm = os.path.join(config.NORM_CSV_FOLDER, class_name)
        os.makedirs(dir_raw,  exist_ok=True)
        os.makedirs(dir_norm, exist_ok=True)

        fname = os.path.splitext(video_file)[0]

        # Save original
        ok_orig, reason_orig = save_sequence(
            data_orig, detected_orig, original_fps,
            dir_raw, dir_norm, class_name, fname, False, interpolator
        )
        if ok_orig:
            success_orig_count += 1
            print(f"  [OK] Saved: {fname}.csv  (detected {detected_orig} frames)")
        else:
            rejected_list.append(f"{class_name}/{video_file} - Orig: {reason_orig}")
            print(f"  [SKIP] Orig → {reason_orig}")

        # Save flipped
        ok_flip, reason_flip = save_sequence(
            data_flip, detected_flip, original_fps,
            dir_raw, dir_norm, class_name, fname, True, interpolator
        )
        if ok_flip:
            success_flip_count += 1
            print(f"  [OK] Saved: {fname}_flipped.csv  (detected {detected_flip} frames)")
        else:
            rejected_list.append(f"{class_name}/{video_file} - Flip: {reason_flip}")
            print(f"  [SKIP] Flip → {reason_flip}")

    # Final summary
    if rejected_list:
        reject_file_path = os.path.join(config.REJECTED_FOLDER, "rejected_log.txt")
        with open(reject_file_path, "w", encoding="utf-8") as f:
            for item in rejected_list:
                f.write(item + "\n")

    total_success = success_orig_count + success_flip_count
    print(f"\n========================================")
    print(f" EXTRACTION SUMMARY")
    print(f"========================================")
    print(f"Total Videos Processed : {total_videos}")
    print(f"Target CSV Files       : {total_videos * 2}  (orig + flipped)")
    print(f"Successfully Created   : {total_success}  ({success_orig_count} orig, {success_flip_count} flipped)")
    print(f"Failed/Rejected Files  : {len(rejected_list)}")
    print(f"========================================\n")


if __name__ == "__main__":
    run()