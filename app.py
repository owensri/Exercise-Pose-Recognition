import streamlit as st
import os
import sys
import tempfile
import cv2
import numpy as np
import mediapipe as mp
import time
import subprocess
import urllib.request
import pandas as pd
import altair as alt
from collections import deque
from tensorflow.keras.models import load_model
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
import config
from preprocessor import Preprocessor
from smoother import LandmarkSmoother
from rep_counter import RepCounter

st.set_page_config(page_title="AI Exercise Tracker", page_icon="🏋️‍♂️", layout="wide")

st.markdown("""
    <style>
    .main { padding-top: 2rem; }
    h1 { font-weight: 600; letter-spacing: -0.5px; }
    h2, h3 { font-weight: 500; }
    div[data-testid="stMetric"] { background-color: #111111; padding: 20px; border-radius: 12px; text-align: center; }
    div[data-testid="stMetricValue"] { font-size: 28px; font-weight: 600; }
    div[data-testid="stMetricLabel"] { font-size: 14px; text-transform: uppercase; letter-spacing: 1px; }
    </style>
""", unsafe_allow_html=True)

TARGET_INDICES = [11, 12, 13, 14, 15, 16, 23, 24, 25, 26, 27, 28]

CUSTOM_CONNECTIONS = [
    (11, 12), (11, 13), (13, 15), (12, 14), (14, 16),
    (11, 23), (12, 24), (23, 24),
    (23, 25), (25, 27), (24, 26), (26, 28)
]

OTHER_THRESHOLD = 0.85
VOTING_WINDOW   = 5

# ----------------------------------------------------------------
# ค่าสี — แยก BGR (OpenCV) และ RGB (PIL) ชัดเจน ไม่สับสน
# ----------------------------------------------------------------
# landmark ทั่วไป → เขียว
CV_COLOR_NORMAL    = (0,   200,  50)      # BGR สำหรับ OpenCV
PIL_COLOR_NORMAL   = (50,  200,   0)      # RGB สำหรับ PIL

# มุมหลัก (Primary) → น้ำเงิน
CV_COLOR_PRIMARY   = (255,  80,   0)      # BGR สำหรับ OpenCV  (B=255, G=80, R=0)
PIL_COLOR_PRIMARY  = (0,    80, 255)      # RGB สำหรับ PIL     (R=0,   G=80, B=255)

# มุมรอง (Secondary) → แดง
CV_COLOR_SECONDARY = (0,    0,  220)      # BGR สำหรับ OpenCV  (B=0, G=0, R=220)
PIL_COLOR_SECONDARY= (220,  0,    0)      # RGB สำหรับ PIL     (R=220, G=0, B=0)

# ขนาด canvas
CANVAS_W    = 800
DASHBOARD_H = 120   # เพิ่มจาก 100 → 120 เพื่อให้ข้อความไม่ทับกัน
VIDEO_H     = 450
CANVAS_H    = DASHBOARD_H + VIDEO_H      # 570


def draw_custom_landmarks(image, landmarks_list, angle_data=None, current_action=None):
    """
    วาด skeleton + landmark บนภาพ RGB (OpenCV array)
    - ทั่วไป       → เขียว
    - หลัก         → น้ำเงิน + วงใหญ่
    - รอง          → แดง    + วงใหญ่
    - lunge knees  → ส้ม    + วงใหญ่ (ทั้ง 2 เข่า)
    """
    h, w, _ = image.shape

    primary_idx        = None
    secondary_idx      = None
    lunge_knee_indices = set()

    if angle_data:
        p_lm = angle_data['primary']['landmark']
        s_lm = angle_data['secondary']['landmark']
        for idx in TARGET_INDICES:
            lm = landmarks_list.landmark[idx]
            if abs(lm.x - p_lm.x) < 1e-5 and abs(lm.y - p_lm.y) < 1e-5:
                primary_idx = idx
            if abs(lm.x - s_lm.x) < 1e-5 and abs(lm.y - s_lm.y) < 1e-5:
                secondary_idx = idx

        # เฉพาะ lunge — หา index เข่าทั้ง 2 ข้าง
        if current_action == 'lunge':
            for key in ('lunge_left_knee', 'lunge_right_knee'):
                knee_data = angle_data.get(key)
                if knee_data and knee_data['landmark'] is not None:
                    k_lm = knee_data['landmark']
                    for idx in TARGET_INDICES:
                        lm = landmarks_list.landmark[idx]
                        if abs(lm.x - k_lm.x) < 1e-5 and abs(lm.y - k_lm.y) < 1e-5:
                            lunge_knee_indices.add(idx)

    # วาดเส้น skeleton
    for idx1, idx2 in CUSTOM_CONNECTIONS:
        lm1 = landmarks_list.landmark[idx1]
        lm2 = landmarks_list.landmark[idx2]
        if lm1.visibility > 0.5 and lm2.visibility > 0.5:
            pt1 = (int(lm1.x * w), int(lm1.y * h))
            pt2 = (int(lm2.x * w), int(lm2.y * h))
            cv2.line(image, pt1, pt2, (180, 180, 180), 2)

    # วาดจุด landmark
    for idx in TARGET_INDICES:
        lm = landmarks_list.landmark[idx]
        if lm.visibility > 0.5:
            pt = (int(lm.x * w), int(lm.y * h))
            if idx in lunge_knee_indices:
                # lunge — เข่าทั้ง 2 ข้าง → น้ำเงิน (Primary)
                cv2.circle(image, pt, 10, CV_COLOR_PRIMARY,  -1)
                cv2.circle(image, pt, 13, (255, 255, 255),    2)
            elif idx == primary_idx:
                cv2.circle(image, pt, 10, CV_COLOR_PRIMARY,   -1)
                cv2.circle(image, pt, 13, (255, 255, 255),     2)
            elif idx == secondary_idx:
                cv2.circle(image, pt, 10, CV_COLOR_SECONDARY, -1)
                cv2.circle(image, pt, 13, (255, 255, 255),     2)
            else:
                cv2.circle(image, pt,  6, CV_COLOR_NORMAL,    -1)
                cv2.circle(image, pt,  9, (255, 255, 255),     1)


def draw_angle_labels(image_rgb, angle_data, current_action=None):
    """
    วาดตัวเลของศาถัดจากจุดมุม
    - มุมหลัก  → น้ำเงิน
    - มุมรอง   → แดง
    - lunge    → องศาเข่าทั้ง 2 ข้าง → ส้ม (แทน primary/secondary)
    """
    if not angle_data:
        return
    h, w, _ = image_rgb.shape

    if current_action == 'lunge':
        # วาดองศาเข่าทั้ง 2 ข้างด้วยสีส้ม แทน primary
        for key in ('lunge_left_knee', 'lunge_right_knee'):
            knee = angle_data.get(key)
            if knee and knee['landmark'] is not None and knee['angle'] is not None:
                k_lm  = knee['landmark']
                k_ang = knee['angle']
                kx, ky = int(k_lm.x * w), int(k_lm.y * h)
                cv2.putText(image_rgb, f"{int(k_ang)}", (kx + 14, ky),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),         4, cv2.LINE_AA)
                cv2.putText(image_rgb, f"{int(k_ang)}", (kx + 14, ky),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, CV_COLOR_PRIMARY,   2, cv2.LINE_AA)  # BGR น้ำเงิน

        # วาดองศา secondary (ลำตัว) ด้วยสีแดงเหมือนเดิม
        s_lm    = angle_data['secondary']['landmark']
        s_angle = angle_data['secondary']['angle']
        sx, sy  = int(s_lm.x * w), int(s_lm.y * h)
        cv2.putText(image_rgb, f"{int(s_angle)}", (sx + 14, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),           4, cv2.LINE_AA)
        cv2.putText(image_rgb, f"{int(s_angle)}", (sx + 14, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CV_COLOR_SECONDARY,   2, cv2.LINE_AA)
    else:
        # Pushup / Squat — primary น้ำเงิน, secondary แดง
        p_lm    = angle_data['primary']['landmark']
        p_angle = angle_data['primary']['angle']
        px, py  = int(p_lm.x * w), int(p_lm.y * h)
        cv2.putText(image_rgb, f"{int(p_angle)}", (px + 14, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),         4, cv2.LINE_AA)
        cv2.putText(image_rgb, f"{int(p_angle)}", (px + 14, py),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CV_COLOR_PRIMARY,   2, cv2.LINE_AA)

        s_lm    = angle_data['secondary']['landmark']
        s_angle = angle_data['secondary']['angle']
        sx, sy  = int(s_lm.x * w), int(s_lm.y * h)
        cv2.putText(image_rgb, f"{int(s_angle)}", (sx + 14, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0),           4, cv2.LINE_AA)
        cv2.putText(image_rgb, f"{int(s_angle)}", (sx + 14, sy),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, CV_COLOR_SECONDARY,   2, cv2.LINE_AA)


@st.cache_resource
def load_model_cached(path):
    return load_model(path, compile=False)


@st.cache_resource
def get_thai_font(size):
    font_name = "Sarabun-Regular.ttf"
    font_url  = "https://github.com/google/fonts/raw/main/ofl/sarabun/Sarabun-Regular.ttf"
    if not os.path.exists(font_name):
        try:
            urllib.request.urlretrieve(font_url, font_name)
        except Exception:
            pass
    font_paths = [
        font_name,
        "tahoma.ttf",
        "/System/Library/Fonts/Supplemental/Tahoma.ttf",
        "/System/Library/Fonts/Thonburi.ttc",
        "C:\\Windows\\Fonts\\tahoma.ttf",
    ]
    for path in font_paths:
        if os.path.exists(path):
            try:
                return ImageFont.truetype(path, size)
            except Exception:
                continue
    return ImageFont.load_default()


def build_dashboard_bar(canvas, draw, stages, counters,
                         current_action, confidence, window_frames,
                         feedback_msg, fb_color,
                         font_action, font_feedback, font_counter, font_legend):
    """
    วาด dashboard bar ในโซน y = 0..DASHBOARD_H บน canvas
    แบ่งเป็น 3 โซนแนวนอน:
      [Legend 0..200] [Action+Feedback 200..580] [Counter 580..800]
    """
    # พื้นหลังทั้งแถบ
    cv2.rectangle(canvas, (0, 0), (CANVAS_W, DASHBOARD_H), (20, 20, 20), -1)
    # พื้นหลัง zone counter (ขวา)
    cv2.rectangle(canvas, (580, 0), (CANVAS_W, DASHBOARD_H), (40, 40, 40), -1)
    # เส้นแบ่ง legend | action
    cv2.line(canvas, (200, 4), (200, DASHBOARD_H - 4), (80, 80, 80), 1)
    # เส้นแบ่ง action | counter
    cv2.line(canvas, (580, 4), (580, DASHBOARD_H - 4), (80, 80, 80), 1)
    # เส้นขอบล่าง dashboard
    cv2.line(canvas, (0, DASHBOARD_H), (CANVAS_W, DASHBOARD_H), (180, 180, 180), 1)

    # ---- ZONE 1: Legend (x=0..200) ----
    dot_x, dot_r = 14, 6
    # แถว 1 — เขียว landmark ทั่วไป
    draw.ellipse([dot_x - dot_r, 10, dot_x + dot_r, 10 + dot_r*2],
                 fill=PIL_COLOR_NORMAL, outline=(255, 255, 255))
    draw.text((28, 8),  "Landmark ทั่วไป",        font=font_legend, fill=(200, 200, 200))
    # แถว 2 — น้ำเงิน มุมหลัก
    draw.ellipse([dot_x - dot_r, 46, dot_x + dot_r, 46 + dot_r*2],
                 fill=PIL_COLOR_PRIMARY, outline=(255, 255, 255))
    draw.text((28, 44), "มุมหลัก (Primary)",       font=font_legend, fill=PIL_COLOR_PRIMARY)
    # แถว 3 — แดง มุมรอง
    draw.ellipse([dot_x - dot_r, 82, dot_x + dot_r, 82 + dot_r*2],
                 fill=PIL_COLOR_SECONDARY, outline=(255, 255, 255))
    draw.text((28, 80), "มุมรอง (Secondary)",      font=font_legend, fill=PIL_COLOR_SECONDARY)

    # ---- ZONE 2: Action + Feedback (x=208..572) ----
    current_stage = stages.get(current_action) or "-"
    if len(window_frames) < config.SEQUENCE_LENGTH:
        action_text  = f"กำลังเก็บข้อมูล... {len(window_frames)}/{config.SEQUENCE_LENGTH}"
        action_color = (255, 220, 0)
    else:
        action_text  = f"{current_action.upper()}  {confidence*100:.0f}%  |  {current_stage.upper()}"
        action_color = (80, 255, 80) if confidence > 0.8 else (255, 165, 0)

    draw.text((208,  8), action_text,                    font=font_action,   fill=action_color)
    draw.text((208, 52), f"▶ {feedback_msg}",            font=font_feedback, fill=fb_color)

    # ---- ZONE 3: Counter (x=588..800) ----
    hi = (255, 165, 0)
    lo = (255, 255, 255)
    draw.text((590,  8), f"PUSH : {counters['pushup']}", font=font_counter,
              fill=hi if current_action == 'pushup' else lo)
    draw.text((590, 46), f"SQUAT: {counters['squat']}",  font=font_counter,
              fill=hi if current_action == 'squat'  else lo)
    draw.text((590, 84), f"LUNGE: {counters['lunge']}",  font=font_counter,
              fill=hi if current_action == 'lunge'  else lo)


def main():
    # ================================================================
    #  DEVELOPER CONFIG — ปรับตรงนี้จุดเดียว
    #  SHOW_MODEL_SELECTOR = True  → ให้ผู้ใช้เลือกโมเดลเองได้ (สำหรับนักพัฒนา)
    #  SHOW_MODEL_SELECTOR = False → ใช้ DEFAULT_MODEL อัตโนมัติ (สำหรับผู้ใช้ทั่วไป)
    SHOW_MODEL_SELECTOR = False
    DEFAULT_MODEL       = "LSTM + CNN (Hybrid)"
    # ================================================================

    # ---- Header ----
    st.markdown("<h1 style='text-align: center;'>🏋️ AI Exercise Tracker</h1>",
                unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; color: #888888;'>"
                "อัปโหลดวิดีโอท่าออกกำลังกาย แล้วให้ AI นับครั้งและแนะนำฟอร์มให้คุณ</p>",
                unsafe_allow_html=True)

    st.markdown("---")

    # ---- Model config (hidden จากผู้ใช้) ----
    model_files = {
        "1-Layer LSTM":         "exercise_model_lstm_1layer.keras",
        "2-Layer LSTM":         "exercise_model_lstm_2layer.keras",
        "LSTM + CNN (Hybrid)":  "exercise_model_hybrid.keras",
        "CNN":                  "exercise_model_cnn.keras",
    }
    if SHOW_MODEL_SELECTOR:
        model_choice = st.selectbox("เลือกโมเดล AI:", list(model_files.keys()))
    else:
        model_choice = DEFAULT_MODEL

    selected_model_file = model_files[model_choice]
    model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), selected_model_file)

    # ---- STEP 1: เลือกระดับ ----
    st.markdown("### 🎯 Step 1 — เลือกระดับของคุณ")
    st.caption("ระดับจะกำหนดว่าต้องย่อตัวลึกแค่ไหนจึงจะนับ 1 ครั้ง")

    diff_display = st.radio(
        label="ระดับผู้ใช้งาน",
        options=["🟢  Beginner", "🔴  Advanced"],
        horizontal=True,
        label_visibility="collapsed"
    )
    difficulty_level = "beginner" if "Beginner" in diff_display else "advanced"

    if difficulty_level == "beginner":
        st.info(
            "**Beginner** เหมาะสำหรับคนที่เพิ่งเริ่มออกกำลังกาย หรือยังไม่ค่อยยืดหยุ่น — "
            "ย่อตัวลงครึ่งทางก็นับให้"
        )
    else:
        st.warning(
            "**Advanced** เหมาะสำหรับคนที่ออกกำลังกายสม่ำเสมอ — "
            "ต้องย่อตัวลงลึกและเหยียดขึ้นให้สุดจึงจะนับ"
        )

    st.markdown("---")

    # ---- STEP 2: อัปโหลดวิดีโอ ----
    st.markdown("### 📂 Step 2 — อัปโหลดวิดีโอ")

    st.markdown(
        "💡 **ท่าที่รองรับ:** Pushup, Squat, Lunge &nbsp;|&nbsp; "
        "รองรับไฟล์ .mp4 .mov .avi &nbsp;|&nbsp; แนะนำคลิปไม่เกิน 1 นาที",
        unsafe_allow_html=True
    )
    st.markdown(
        "> ⚠️ **ข้อแนะนำสำคัญ:** ถ่ายวิดีโอจาก**มุมด้านข้างลำตัว**เท่านั้น (ซ้ายหรือขวา) "
        "ไม่ควรถ่ายจากด้านหน้าหรือด้านหลัง เพราะระบบจะตรวจจับท่าได้ไม่ถูกต้อง "
        "และให้ร่างกายทั้งตัวอยู่ในกรอบภาพตลอดเวลา"
    )

    uploaded_file = st.file_uploader(
        label="อัปโหลดวิดีโอ",
        type=["mp4", "mov", "avi"],
        label_visibility="collapsed"
    )

    if uploaded_file is not None:
        st.success(f"✅ อัปโหลด '{uploaded_file.name}' สำเร็จ!")

    st.markdown("---")

    # ---- ปุ่มเริ่มวิเคราะห์ (กลางหน้า) ----
    col_l, col_c, col_r = st.columns([2, 1, 2])
    with col_c:
        start_button = st.button("🚀 เริ่มวิเคราะห์", type="primary", use_container_width=True)

    if uploaded_file is None:
        return

    if not start_button:
        return

    # ==================== Processing ====================
    st.subheader(f"กำลังประมวลผล... ระดับ: {'ผู้เริ่มต้น (Beginner)' if difficulty_level == 'beginner' else 'ขั้นสูง (Advanced)'}")

    if not os.path.exists(model_path):
        st.error(f"ไม่พบไฟล์โมเดล '{selected_model_file}'")
        return

    overall_start_time = time.time()
    model = load_model_cached(model_path)

    # โหลดฟอนต์ขนาดต่างๆ
    font_action   = get_thai_font(20)
    font_feedback = get_thai_font(16)
    font_counter  = get_thai_font(18)
    font_legend   = get_thai_font(14)

    classes     = config.CLASSES
    mp_pose     = mp.solutions.pose
    pose        = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)

    preprocessor  = Preprocessor()
    smoother      = LandmarkSmoother()
    rep_counter   = RepCounter(difficulty=difficulty_level)

    window_frames      = deque(maxlen=config.SEQUENCE_LENGTH)
    prediction_buffer  = deque(maxlen=VOTING_WINDOW)

    progress_bar = st.progress(0)
    status_text  = st.empty()
    status_text.info("กำลังประมวลผล... กรุณารอจนกว่าแถบสถานะจะเต็ม")

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(uploaded_file.read())
    cap         = cv2.VideoCapture(tfile.name)
    fps         = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    temp_out_mp4  = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    final_out_mp4 = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4').name
    writer = cv2.VideoWriter(temp_out_mp4, cv2.VideoWriter_fourcc(*'mp4v'),
                             fps, (CANVAS_W, CANVAS_H))

    current_action = "กำลังวิเคราะห์..."
    confidence     = 0.0
    frame_counter  = 0
    counters       = {'pushup': 0, 'squat': 0, 'lunge': 0}
    stages         = {'pushup': None, 'squat': None, 'lunge': None}
    feedback_msg   = "รอภาพ..."
    fb_color       = (255, 255, 255)
    angle_data     = None

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        frame_counter += 1
        frame     = cv2.resize(frame, (CANVAS_W, VIDEO_H))
        image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        image_rgb.flags.writeable = False
        results = pose.process(image_rgb)
        image_rgb.flags.writeable = True

        angle_data = None

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            lms = preprocessor.get_landmarks(results)
            if lms:
                lms      = smoother.process(lms)
                features = preprocessor.normalize(lms)
                window_frames.append(features)

            if len(window_frames) == config.SEQUENCE_LENGTH and frame_counter % 3 == 0:
                input_data  = np.expand_dims(np.array(window_frames), axis=0)
                predictions = model.predict(input_data, verbose=0)[0]
                best_idx    = np.argmax(predictions)

                if "other" in classes:
                    other_idx = classes.index("other")
                    if best_idx == other_idx and predictions[other_idx] < OTHER_THRESHOLD:
                        tmp = predictions.copy()
                        tmp[other_idx] = 0.0
                        best_idx = np.argmax(tmp)

                prediction_buffer.append(best_idx)
                voted_idx = (max(set(prediction_buffer), key=prediction_buffer.count)
                             if len(prediction_buffer) == prediction_buffer.maxlen
                             else best_idx)
                confidence     = predictions[voted_idx]
                current_action = classes[voted_idx]

            try:
                counters, stages, feedback_msg, fb_color, angle_data = rep_counter.process(
                    current_action, confidence, landmarks, mp_pose
                )
            except Exception:
                pass

            # วาด landmark พร้อมไฮไลต์จุดมุม
            draw_custom_landmarks(image_rgb, results.pose_landmarks, angle_data, current_action)

        # วาดตัวเลของศา
        draw_angle_labels(image_rgb, angle_data, current_action)

        # ---- สร้าง canvas และ paste วิดีโอลง ----
        canvas = np.zeros((CANVAS_H, CANVAS_W, 3), dtype=np.uint8)
        canvas[DASHBOARD_H:CANVAS_H, 0:CANVAS_W] = image_rgb

        # วาด dashboard bar ผ่าน PIL
        pil_img = Image.fromarray(canvas)
        draw    = ImageDraw.Draw(pil_img)

        build_dashboard_bar(
            canvas, draw, stages, counters,
            current_action, confidence, window_frames,
            feedback_msg, fb_color,
            font_action, font_feedback, font_counter, font_legend
        )

        final_frame = np.array(pil_img)
        writer.write(cv2.cvtColor(final_frame, cv2.COLOR_RGB2BGR))

        if frame_counter % 15 == 0 and total_frames > 0:
            pct = min(frame_counter / total_frames, 1.0)
            progress_bar.progress(pct)
            status_text.info(f"กำลังประมวลผล... {int(pct * 100)}%")

    cap.release()
    writer.release()

    status_text.info("กำลังเตรียมวิดีโอผลลัพธ์...")
    try:
        subprocess.run(
            ['ffmpeg', '-y', '-i', temp_out_mp4,
             '-vcodec', 'libx264', '-pix_fmt', 'yuv420p', '-preset', 'fast', '-an',
             final_out_mp4],
            check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except Exception:
        final_out_mp4 = temp_out_mp4

    progress_bar.empty()
    status_text.empty()
    st.success("ประมวลผลเสร็จสมบูรณ์!")

    st.markdown("<h2 style='text-align:center;'>วิดีโอผลลัพธ์</h2>", unsafe_allow_html=True)
    st.video(final_out_mp4)

    st.markdown("<hr>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align:center;'>สรุปจำนวนครั้ง</h2>", unsafe_allow_html=True)
    col1, col2, col3 = st.columns(3)
    col1.metric("Pushup", counters['pushup'])
    col2.metric("Squat",  counters['squat'])
    col3.metric("Lunge",  counters['lunge'])

    source = pd.DataFrame({
        'Exercise': ['Pushup', 'Squat', 'Lunge'],
        'Reps':     [counters['pushup'], counters['squat'], counters['lunge']]
    })
    chart = alt.Chart(source).mark_bar(color='#88c2f8').encode(
        x=alt.X('Exercise', sort=['Pushup', 'Squat', 'Lunge'], title=None,
                axis=alt.Axis(labelAngle=0)),
        y=alt.Y('Reps', title=None, axis=alt.Axis(tickMinStep=1)),
        tooltip=['Exercise', 'Reps']
    ).properties(height=350).configure_view(strokeWidth=0)
    st.altair_chart(chart, use_container_width=True, theme="streamlit")

    st.markdown("<h2 style='text-align:center;'>ประสิทธิภาพระบบ</h2>", unsafe_allow_html=True)
    total_duration = time.time() - overall_start_time
    c1, c2 = st.columns(2)
    c1.metric("เวลาที่ใช้ทั้งหมด (วินาที)", f"{total_duration:.2f}")
    c2.metric("ความเร็วเฉลี่ย (FPS)",        f"{frame_counter / total_duration:.1f}")


if __name__ == "__main__":
    main()
