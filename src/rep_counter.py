import numpy as np
from collections import namedtuple

Threshold = namedtuple('Threshold', ['down', 'up', 'form'])

THRESHOLDS = {
    'beginner': {
        'pushup': Threshold(115, 140, 140),
        'squat':  Threshold(115, 140,  40),
        'lunge':  Threshold(115, 140, 130),
    },
    'advanced': {
        'pushup': Threshold( 90, 160, 160),
        'squat':  Threshold( 90, 160,  60),
        'lunge':  Threshold( 90, 160, 150),
    },
}

MIN_CONFIDENCE = 0.80
MIN_VISIBILITY = 0.60

JOINT_MAP = {
    'pushup': {
        'left':  ['LEFT_SHOULDER',  'LEFT_ELBOW',  'LEFT_WRIST',  'LEFT_SHOULDER',  'LEFT_HIP',  'LEFT_KNEE'],
        'right': ['RIGHT_SHOULDER', 'RIGHT_ELBOW', 'RIGHT_WRIST', 'RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'],
    },
    'squat': {
        'left':  ['LEFT_HIP',  'LEFT_KNEE',  'LEFT_ANKLE',  'LEFT_SHOULDER',  'LEFT_HIP',  'LEFT_KNEE'],
        'right': ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE', 'RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'],
    },
    'lunge': {
        'left':  ['LEFT_HIP',  'LEFT_KNEE',  'LEFT_ANKLE',  'RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'],
        'right': ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE', 'LEFT_SHOULDER',  'LEFT_HIP',  'LEFT_KNEE'],
    },
}

LUNGE_FRONT_JOINTS = {
    'left':  ['LEFT_HIP',  'LEFT_KNEE',  'LEFT_ANKLE'],
    'right': ['RIGHT_HIP', 'RIGHT_KNEE', 'RIGHT_ANKLE'],
}

LUNGE_BACK_JOINTS = {
    'left':  ['LEFT_SHOULDER',  'LEFT_HIP',  'LEFT_KNEE'],
    'right': ['RIGHT_SHOULDER', 'RIGHT_HIP', 'RIGHT_KNEE'],
}

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

    def get_angles(self, landmarks, mp_pose, action):
        left_vis  = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].visibility
        right_vis = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].visibility
        side = 'left' if left_vis >= right_vis else 'right'

        if action == 'lunge':
            left_hip_y  = landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y
            right_hip_y = landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y
            back_side   = 'left'  if left_hip_y  > right_hip_y else 'right'
            front_side  = 'right' if back_side  == 'left'      else 'left'

            p1, p2, p3 = [
                landmarks[getattr(mp_pose.PoseLandmark, name).value]
                for name in LUNGE_FRONT_JOINTS[front_side]
            ]
            if min(p1.visibility, p2.visibility, p3.visibility) < self.min_visibility:
                return None, None, None, None
            primary_angle = self.calculate_angle([p1.x, p1.y], [p2.x, p2.y], [p3.x, p3.y])

            s1, s2, s3 = [
                landmarks[getattr(mp_pose.PoseLandmark, name).value]
                for name in LUNGE_BACK_JOINTS[back_side]
            ]
            secondary_angle = self.calculate_angle([s1.x, s1.y], [s2.x, s2.y], [s3.x, s3.y])

            if primary_angle is None or secondary_angle is None:
                return None, None, None, None
            return primary_angle, secondary_angle, p2, s2

        else:
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

    def get_lunge_both_knees(self, landmarks, mp_pose):
        left_angles  = self._get_one_side_angles(landmarks, mp_pose, 'lunge', 'left')
        right_angles = self._get_one_side_angles(landmarks, mp_pose, 'lunge', 'right')
        return left_angles, right_angles

    def process(self, current_action, confidence, landmarks, mp_pose):
        feedback = "กำลังวิเคราะห์ท่าทาง..."
        color    = (255, 255, 255)

        if confidence < self.min_confidence or current_action not in self.counters:
            return self.counters, self.stages, "รอจับภาพท่าทางที่ชัดเจน...", (200, 200, 200), None

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

        if current_action == 'lunge':
            left_a, right_a = self.get_lunge_both_knees(landmarks, mp_pose)
            if left_a[0] is None or right_a[0] is None:
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
            is_deep_down = primary_angle   < th.down
            is_fully_up  = primary_angle   > th.up
            is_bad_form  = secondary_angle < th.form

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
                    self.stages[current_action] = 'up'
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

        angle_data = {
            'primary':   {'angle': primary_angle,   'landmark': p_landmark},
            'secondary': {'angle': secondary_angle, 'landmark': s_landmark},
        }

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