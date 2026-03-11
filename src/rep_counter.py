import numpy as np
from collections import namedtuple

# =============================================================================
#  THRESHOLD CONFIGURATION
#  แก้ไขค่าองศาได้ที่นี่เพียงที่เดียว — ไม่ต้องแตะส่วนอื่นของโค้ด
#
#  Threshold(down, up, form)
#    down : มุมหลักต้องน้อยกว่านี้  → นับว่าย่อลงถึงจุดต่ำสุด  (stage = "down")
#    up   : มุมหลักต้องมากกว่านี้  → นับว่าขึ้นถึงจุดสูงสุด   (นับ +1)
#    form : มุมรองต้องมากกว่านี้   → ถือว่าฟอร์มถูกต้อง
#
#  มุมหลัก (Primary angle)   🔵
#    Pushup : Shoulder → Elbow  → Wrist   (มุมข้อศอก)
#    Squat  : Hip      → Knee   → Ankle   (มุมเข่า)
#    Lunge  : Hip      → Knee   → Ankle   (มุมเข่า ทั้ง 2 ข้าง AND)
#
#  มุมรอง (Secondary angle)  🔴
#    Pushup : Shoulder → Hip    → Knee    (ความตรงของลำตัว)
#    Squat  : Shoulder → Hip    → Knee    (ความตรงของลำตัว)
#    Lunge  : Shoulder → Hip    → Knee    (ความตรงของลำตัว — วัดจากขาหลัง)
# =============================================================================

Threshold = namedtuple('Threshold', ['down', 'up', 'form'])

THRESHOLDS = {
    # -------------------------------------------------------------------------
    #  BEGINNER
    # -------------------------------------------------------------------------
    'beginner': {
        #              down   up   form
        'pushup': Threshold(115, 140, 140),
        'squat':  Threshold(115, 140,  40),
        'lunge':  Threshold(115, 140, 130),
    },

    # -------------------------------------------------------------------------
    #  ADVANCED
    # -------------------------------------------------------------------------
    'advanced': {
        #              down  up   form
        'pushup': Threshold( 90, 160, 160),
        'squat':  Threshold( 90, 160,  60),
        'lunge':  Threshold( 90, 160, 150),
    },
}

# =============================================================================
#  DETECTION SETTINGS
#  ค่าความมั่นใจขั้นต่ำก่อนระบบเริ่มทำงาน
# =============================================================================

MIN_CONFIDENCE = 0.80   # ความมั่นใจของโมเดลในการจำแนกท่า (0.0 – 1.0)
MIN_VISIBILITY = 0.60   # ความชัดของ landmark แต่ละจุด      (0.0 – 1.0)

# =============================================================================
#  JOINT MAP  (ไม่แนะนำให้แก้ไข)
#  กำหนด landmark ที่ใช้คำนวณมุมหลัก [p1, p2, p3] และมุมรอง [s1, s2, s3]
#  สำหรับ Pushup และ Squat — Lunge มี logic แยกใน get_angles()
# =============================================================================

JOINT_MAP = {
    'pushup': {
        # มุมหลัก: Shoulder→Elbow→Wrist  |  มุมรอง: Shoulder→Hip→Knee
        'left':  ['LEFT_SHOULDER',  'LEFT_ELBOW',  'LEFT_WRIST',  'LEFT_SHOULDER',  'LEFT_HIP',  'LEFT_KNEE'],
        'right': ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST', 'RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'],
    },
    'squat': {
        # มุมหลัก: Hip→Knee→Ankle  |  มุมรอง: Shoulder→Hip→Knee
        'left':  ['LEFT_HIP',  'LEFT_KNEE',  'LEFT_ANKLE',  'LEFT_SHOULDER',  'LEFT_HIP',  'LEFT_KNEE'],
        'right': ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE', 'RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'],
    },
    'lunge': {
        # ใช้เฉพาะ p1,p2,p3 (มุมหลักเข่าทั้ง 2 ข้าง)
        # มุมรองของ Lunge คำนวณแยกใน get_angles() โดยใช้ขาหลัง (Hip Y)
        'left':  ['LEFT_HIP',  'LEFT_KNEE',  'LEFT_ANKLE',  'RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'],
        'right': ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE', 'LEFT_SHOULDER',  'LEFT_HIP',  'LEFT_KNEE'],
    },
}

# Landmark ที่ใช้คำนวณมุมหลักของขาหน้า Lunge
LUNGE_FRONT_JOINTS = {
    'left':  ['LEFT_HIP',  'LEFT_KNEE',  'LEFT_ANKLE'],
    'right': ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'],
}

# Landmark ที่ใช้คำนวณมุมรองของขาหลัง Lunge (Shoulder→Hip→Knee)
LUNGE_BACK_JOINTS = {
    'left':  ['LEFT_SHOULDER',  'LEFT_HIP',  'LEFT_KNEE'],
    'right': ['RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'],
}


# =============================================================================
#  REP COUNTER CLASS
# =============================================================================

class RepCounter:

    def __init__(self, difficulty="beginner"):
        self.difficulty   = difficulty.lower()
        self.counters     = {'pushup': 0, 'squat': 0, 'lunge': 0}
        self.stages       = {'pushup': None, 'squat': None, 'lunge': None}
        self.thresholds   = THRESHOLDS
        self.min_confidence = MIN_CONFIDENCE
        self.min_visibility = MIN_VISIBILITY
        self._last_action = None

    def set_difficulty(self, new_difficulty):
        self.difficulty = new_difficulty.lower()

    # ─── คำนวณมุมระหว่าง 3 จุด ───────────────────────────────────────────────
    def calculate_angle(self, a, b, c):
        try:
            a, b, c = np.array(a), np.array(b), np.array(c)
            radians = (np.arctan2(c[1] - b[1], c[0] - b[0])
                       - np.arctan2(a[1] - b[1], a[0] - b[0]))
            angle = np.abs(radians * 180.0 / np.pi)
            angle = 360 - angle if angle > 180.0 else angle
            return float(np.clip(angle, 0, 360))
        except Exception:
            return None

    # ─── ดึงมุมของฝั่งใดฝั่งหนึ่ง (ใช้กับ Pushup/Squat และ Lunge both-knees) ─
    def _get_one_side_angles(self, landmarks, mp_pose, action, side):
        joint_names = JOINT_MAP[action][side]
        p1, p2, p3, s1, s2, s3 = [
            landmarks[getattr(mp_pose.PoseLandmark, name).value]
            for name in joint_names
        ]
        if min(p1.visibility, p2.visibility, p3.visibility) < self.min_visibility:
            return None, None, None, None

        primary_angle   = self.calculate_angle([p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y])
        secondary_angle = self.calculate_angle([s1.x, s1.y], [s2.x, s2.y], [s3.x, s3.y])

        if primary_angle is None or secondary_angle is None:
            return None, None, None, None
        return primary_angle, secondary_angle, p2, s2

    # ─── เลือกฝั่งและคำนวณมุม (ส่งคืน primary, secondary, landmark ทั้งคู่) ──
    def get_angles(self, landmarks, mp_pose, action):
        """
        Pushup / Squat : ใช้ shoulder visibility เลือกฝั่งใกล้กล้อง
        Lunge          : มุมหลัก  = ขาหน้า  (Hip Y น้อยกว่า)
                         มุมรอง   = ขาหลัง  (Hip Y มากกว่า — normalize แล้วเสถียร)
        """
        left_vis  = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
        right_vis = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
        side = 'left' if left_vis >= right_vis else 'right'

        if action == 'lunge':
            # หาขาหลังจาก hip Y
            left_hip_y  = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            back_side   = 'left'  if left_hip_y  > right_hip_y else 'right'
            front_side  = 'right' if back_side  == 'left'      else 'left'

            # มุมหลัก — Hip→Knee→Ankle ของขาหน้า
            p1, p2, p3 = [
                landmarks[getattr(mp_pose.PoseLandmark, name).value]
                for name in LUNGE_FRONT_JOINTS[front_side]
            ]
            if min(p1.visibility, p2.visibility, p3.visibility) < self.min_visibility:
                return None, None, None, None
            primary_angle = self.calculate_angle([p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y])

            # มุมรอง — Shoulder→Hip→Knee ของขาหลัง
            s1, s2, s3 = [
                landmarks[getattr(mp_pose.PoseLandmark, name).value]
                for name in LUNGE_BACK_JOINTS[back_side]
            ]
            secondary_angle = self.calculate_angle([s1.x, s1.y], [s2.x, s2.y], [s3.x, s3.y])

            if primary_angle is None or secondary_angle is None:
                return None, None, None, None
            return primary_angle, secondary_angle, p2, s2

        else:
            # Pushup / Squat
            joint_names = JOINT_MAP[action][side]
            p1, p2, p3, s1, s2, s3 = [
                landmarks[getattr(mp_pose.PoseLandmark, name).value]
                for name in joint_names
            ]
            if min(p1.visibility, p2.visibility, p3.visibility) < self.min_visibility:
                return None, None, None, None

            primary_angle   = self.calculate_angle([p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y])
            secondary_angle = self.calculate_angle([s1.x, s1.y], [s2.x, s2.y], [s3.x, s3.y])

            if primary_angle is None or secondary_angle is None:
                return None, None, None, None
            return primary_angle, secondary_angle, p2, s2

    # ─── ดึงมุมเข่าทั้ง 2 ข้างสำหรับ Lunge (AND condition) ─────────────────
    def get_lunge_both_knees(self, landmarks, mp_pose):
        left_angles  = self._get_one_side_angles(landmarks, mp_pose, 'lunge', 'left')
        right_angles = self._get_one_side_angles(landmarks, mp_pose, 'lunge', 'right')
        return left_angles, right_angles

    # ─── ประมวลผลหลัก ─────────────────────────────────────────────────────────
    def process(self, current_action, confidence, landmarks, mp_pose):
        feedback = "กำลังวิเคราะห์ท่าทาง..."
        color    = (255, 255, 255)

        if confidence < self.min_confidence or current_action not in self.counters:
            return self.counters, self.stages, "รอจับภาพท่าทางที่ชัดเจน...", (200, 200, 200), None

        # Reset stage เมื่อเปลี่ยน action
        if current_action != self._last_action:
            if self._last_action in self.stages:
                self.stages[self._last_action] = None
            self._last_action = current_action

        th = self.thresholds[self.difficulty][current_action]

        primary_angle, secondary_angle, p_landmark, s_landmark = self.get_angles(
            landmarks, mp_pose, current_action
        )
        if primary_angle is None:
            return self.counters, self.stages, "อวัยวะสำคัญหลุดกล้อง!", (255, 0, 0), None

        # ─── เงื่อนไขการนับ ──────────────────────────────────────────────────
        if current_action == 'lunge':
            # เข่าทั้ง 2 ข้างต้องผ่าน threshold พร้อมกัน (AND)
            left_a, right_a = self.get_lunge_both_knees(landmarks, mp_pose)
            if left_a[0] is None or right_a[0] is None:
                # fallback: ใช้ข้างที่มองเห็น
                fallback = left_a[0] if left_a[0] is not None else right_a[0]
                if fallback is None:
                    return self.counters, self.stages, "อวัยวะสำคัญหลุดกล้อง!", (255, 0, 0), None
                is_deep_down = fallback        < th.down
                is_fully_up  = fallback        > th.up
            else:
                is_deep_down = (left_a[0] < th.down) and (right_a[0] < th.down)
                is_fully_up  = (left_a[0] > th.up)   and (right_a[0] > th.up)
            is_bad_form = secondary_angle < th.form

        else:
            # Pushup / Squat — ใช้ฝั่งเดียว
            is_deep_down = primary_angle   < th.down
            is_fully_up  = primary_angle   > th.up
            is_bad_form  = secondary_angle < th.form

        # ─── Feedback ────────────────────────────────────────────────────────
        if is_bad_form:
            color = (255, 0, 0)
            if is_deep_down:
                feedback = {
                    'pushup': "ย่อแขนลึกดีแล้ว! แต่กดสะโพกลงให้ตัวตรง ถึงจะนับให้นะ",
                    'squat':  "ย่อลึกดีแล้ว! แต่ยืดอกขึ้นอีกนิด อย่าก้มตัว",
                    'lunge':  "ย่อลึกดีแล้ว! แต่ตั้งลำตัวให้ตรงแนวดิ่ง",
                }[current_action]
            elif is_fully_up:
                feedback = {
                    'pushup': "ขึ้นสุดแล้วแต่ยังไม่นับ! จัดหลัง-สะโพกให้ตรงก่อนนะ",
                    'squat':  "ขึ้นสุดแล้วแต่ยังไม่นับ! ยืดอกให้ตรงก่อนนะ",
                    'lunge':  "ขึ้นสุดแล้วแต่ยังไม่นับ! ตั้งลำตัวให้ตรงก่อนนะ",
                }[current_action]
            else:
                feedback = {
                    'pushup': "จัดฟอร์มก่อน! ทำหลังและสะโพกให้ตรงเป็นแนวเดียวกัน",
                    'squat':  "จัดฟอร์มก่อน! ยืดอกขึ้น ระวังอย่าก้มหน้า",
                    'lunge':  "จัดฟอร์มก่อน! ตั้งลำตัวให้ตรงแนวดิ่ง",
                }[current_action]

        else:
            if is_deep_down:
                self.stages[current_action] = "down"
                feedback = "ย่อลงลึกสวยมาก! ออกแรงดันตัวขึ้นเลย!"
                color    = (255, 165, 0)

            elif is_fully_up:
                if self.stages[current_action] == 'down':
                    self.stages[current_action] = None
                    self.counters[current_action] += 1
                    feedback = "เยี่ยมมาก! นับ +1 ฟอร์มเป๊ะสุดๆ"
                    color    = (0, 255, 0)
                else:
                    feedback = "ท่าเตรียมพร้อมสวยครับ เริ่มย่อตัวลงได้เลย"
                    color    = (0, 255, 0)
            else:
                if self.stages[current_action] == 'down':
                    feedback = "ดีมาก! ดันตัวขึ้นอีกให้สุดระยะ!"
                    color    = (255, 165, 0)
                else:
                    feedback = {
                        'pushup': "ฟอร์มสวย! ย่อหน้าอกลงให้ลึกกว่านี้อีกนิด",
                        'squat':  "หลังตรงดีแล้ว! ทิ้งก้นย่อลงให้ลึกกว่านี้",
                        'lunge':  "ตัวตรงดีแล้ว! ย่อลงให้ลึกกว่านี้อีกนิด",
                    }[current_action]
                    color = (150, 255, 50)

        # ─── ส่งข้อมูลมุมกลับให้ app.py วาดจุดและองศา ────────────────────────
        angle_data = {
            'primary':   {'angle': primary_angle,   'landmark': p_landmark},
            'secondary': {'angle': secondary_angle, 'landmark': s_landmark},
        }

        # Lunge: ส่งมุมเข่าทั้ง 2 ข้างแยกสำหรับวาดจุดสีน้ำเงิน
        if current_action == 'lunge':
            left_a, right_a = self.get_lunge_both_knees(landmarks, mp_pose)
            angle_data['lunge_left_knee']  = {
                'angle':    left_a[0]  if left_a[0]  is not None else None,
                'landmark': left_a[2]  if left_a[2]  is not None else None,
            }
            angle_data['lunge_right_knee'] = {
                'angle':    right_a[0] if right_a[0] is not None else None,
                'landmark': right_a[2] if right_a[2] is not None else None,
            }

        return self.counters, self.stages, feedback, color, angle_data
