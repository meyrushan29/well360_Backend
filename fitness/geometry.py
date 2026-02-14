import numpy as np
try:
    import mediapipe as mp
    mp_pose = mp.solutions.pose
except Exception as e:
    print(f"Warning: MediaPipe error: {e}. Fitness features will be disabled.")
    mp = None
    mp_pose = None

# ======================================================
# BASIC GEOMETRY
# ======================================================
def calculate_angle(a, b, c):
    """
    Calculate angle ABC in degrees
    """
    a, b, c = np.array(a), np.array(b), np.array(c)

    ba = a - b
    bc = c - b

    na = np.linalg.norm(ba)
    nc = np.linalg.norm(bc)

    if na < 1e-6 or nc < 1e-6:
        return 180.0

    cos = np.clip(np.dot(ba, bc) / (na * nc), -1.0, 1.0)
    return np.degrees(np.arccos(cos))


def lateral_distance(a, b):
    return abs(a[0] - b[0])


# ======================================================
# LANDMARK NAME → INDEX MAP
# ======================================================
LANDMARK_INDEX = {}
if mp_pose:
    LANDMARK_INDEX = {
        name: lm.value for name, lm in mp_pose.PoseLandmark.__members__.items()
    }


# ======================================================
# 🔥 REQUIRED FUNCTION (THIS WAS MISSING)
# ======================================================
def calculate_angles_from_landmarks(landmarks, exercise_config):
    """
    Returns joint angles required for rep counting

    Output:
        { "main": angle }
    """
    angles = {}

    if not landmarks:
        return angles

    joints = exercise_config.get("joints", [])
    if len(joints) != 3:
        return angles

    try:
        p1 = landmarks.landmark[LANDMARK_INDEX[joints[0]]]
        p2 = landmarks.landmark[LANDMARK_INDEX[joints[1]]]
        p3 = landmarks.landmark[LANDMARK_INDEX[joints[2]]]

        a = (p1.x, p1.y)
        b = (p2.x, p2.y)
        c = (p3.x, p3.y)

        angles["main"] = calculate_angle(a, b, c)

    except Exception:
        pass

    return angles
