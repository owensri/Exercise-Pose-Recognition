import os
import numpy as np
import cv2
import config
from pose_estimator import PoseDetector
from preprocessor import Preprocessor
from fps_aligner import Interpolator
from image_processor import ImageProcessor
from dataset_loader import DatasetLoader
from csv_manager import CSVManager
from frame_filler import FrameFiller
from landmark_filler import LandmarkFiller


def save_sequence(data, detected_count, original_fps, dir_raw, dir_norm, class_name, fname, is_flip, interpolator):
    suffix = "_flipped" if is_flip else ""

    if detected_count < config.MIN_DETECTED_FRAMES:
        return False, f"pose detected เพียง {detected_count} เฟรม (ต้องการ >= {config.MIN_DETECTED_FRAMES})"

    if len(data['norm']) < config.MIN_FRAME_COUNT:
        return False, f"จำนวนเฟรมรวม {len(data['norm'])} น้อยกว่า MIN_FRAME_COUNT={config.MIN_FRAME_COUNT}"

    resamp_raw  = interpolator.process(data['raw'],  original_fps, config.TARGET_FPS)
    resamp_norm = interpolator.process(data['norm'], original_fps, config.TARGET_FPS)

    try:
        with CSVManager(os.path.join(dir_raw, f"{fname}{suffix}.csv")) as csv_raw:
            for r in resamp_raw:
                csv_raw.save_row(class_name, r)

        with CSVManager(os.path.join(dir_norm, f"{fname}{suffix}.csv")) as csv_norm:
            for r in resamp_norm:
                csv_norm.save_row(class_name, r)

    except OSError as e:
        return False, f"ไม่สามารถบันทึกไฟล์ได้: {e}"

    return True, None


def collect_detected_frames(cap, detector, img_processor, preprocessor, flip=False):
    detected_raw   = []
    detected_norm  = []
    total_frames   = 0
    detected_count = 0

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_index  = total_frames
        total_frames += 1

        if flip:
            frame = img_processor.flip_horizontal(frame)

        rgb = img_processor.convert_to_rgb(frame)
        res = detector.find_pose(rgb)
        lms = preprocessor.get_landmarks(res)

        if lms:
            raw_vals  = preprocessor.get_raw_values(lms)
            norm_vals = preprocessor.normalize(lms)
            detected_raw.append((frame_index, raw_vals))
            detected_norm.append((frame_index, norm_vals))
            detected_count += 1

    return detected_raw, detected_norm, total_frames, detected_count


def run():
    print(f"\n[INFO] Starting Extraction Pipeline")
    print(f"[INFO] Target Classes: {', '.join(config.CLASSES)}")

    for folder in [config.RAW_CSV_FOLDER, config.NORM_CSV_FOLDER, config.REJECTED_FOLDER]:
        os.makedirs(folder, exist_ok=True)

    loader          = DatasetLoader()
    interpolator    = Interpolator()
    img_processor   = ImageProcessor()
    filler          = FrameFiller()
    landmark_filler = LandmarkFiller()

    proc_orig = Preprocessor()
    proc_flip = Preprocessor()

    rejected_list      = []
    total_videos       = 0
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

        detector_orig = PoseDetector()
        detector_flip = PoseDetector()

        original_fps = cap.get(cv2.CAP_PROP_FPS)
        if not original_fps or original_fps <= 0:
            print(f"  [WARN] FPS ไม่สามารถอ่านได้ ใช้ค่า fallback = 30")
            original_fps = 30

        proc_orig.reset()
        proc_flip.reset()

        det_raw_orig, det_norm_orig, total_frames, detected_orig = collect_detected_frames(
            cap, detector_orig, img_processor, proc_orig, flip=False
        )

        det_raw_flip, det_norm_flip, _, detected_flip = collect_detected_frames(
            cap, detector_flip, img_processor, proc_flip, flip=True
        )

        cap.release()
        detector_orig.pose.close()
        detector_flip.pose.close()

        filled_raw_orig  = filler.fill(det_raw_orig,  total_frames)
        filled_norm_orig = filler.fill(det_norm_orig, total_frames)
        filled_raw_flip  = filler.fill(det_raw_flip,  total_frames)
        filled_norm_flip = filler.fill(det_norm_flip, total_frames)

        filled_raw_orig  = landmark_filler.fill(filled_raw_orig)
        filled_norm_orig = landmark_filler.fill(filled_norm_orig)
        filled_raw_flip  = landmark_filler.fill(filled_raw_flip)
        filled_norm_flip = landmark_filler.fill(filled_norm_flip)

        data_orig = {'raw': filled_raw_orig,  'norm': filled_norm_orig}
        data_flip = {'raw': filled_raw_flip,  'norm': filled_norm_flip}

        dir_raw  = os.path.join(config.RAW_CSV_FOLDER,  class_name)
        dir_norm = os.path.join(config.NORM_CSV_FOLDER, class_name)
        os.makedirs(dir_raw,  exist_ok=True)
        os.makedirs(dir_norm, exist_ok=True)

        fname = os.path.splitext(video_file)[0]

        ok_orig, reason_orig = save_sequence(
            data_orig, detected_orig, original_fps,
            dir_raw, dir_norm, class_name, fname, False, interpolator
        )
        if ok_orig:
            success_orig_count += 1
            print(f"  [OK] Saved: {fname}.csv  (detected {detected_orig}/{total_frames} frames)")
        else:
            rejected_list.append(f"{class_name}/{video_file} - Orig: {reason_orig}")
            print(f"  [SKIP] Orig → {reason_orig}")

        ok_flip, reason_flip = save_sequence(
            data_flip, detected_flip, original_fps,
            dir_raw, dir_norm, class_name, fname, True, interpolator
        )
        if ok_flip:
            success_flip_count += 1
            print(f"  [OK] Saved: {fname}_flipped.csv  (detected {detected_flip}/{total_frames} frames)")
        else:
            rejected_list.append(f"{class_name}/{video_file} - Flip: {reason_flip}")
            print(f"  [SKIP] Flip → {reason_flip}")

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