# ======================================================
# EXERCISE CONFIGURATION (FINAL & SAFE)
# ======================================================

EXERCISE_CONFIG = {

    # ---------------- UPPER BODY ----------------
    "pull_up": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
        "up": 80,
        "down": 150
    },

    "push_up": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
        "up": 95,
        "down": 160
    },

    "barbell_biceps_curl": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
        "up": 65,
        "down": 160
    },

    "hammer_curl": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
        "up": 70,
        "down": 160
    },

    "bench_press": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
        "up": 100,
        "down": 160
    },

    "incline_bench_press": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
        "up": 80,
        "down": 160
    },

    "lat_pulldown": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
        "up": 80,
        "down": 150
    },

    "tricep_dips": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
        "up": 100,
        "down": 160
    },

    "tricep_pushdown": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
        "up": 80,
        "down": 160
    },

    "shoulder_press": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
        "up": 80,
        "down": 160
    },

    "lateral_raise": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
        "up": 30,
        "down": 90
    },

    "t_bar_row": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_ELBOW", "RIGHT_WRIST"],
        "up": 60,
        "down": 160
    },

    # ---------------- LOWER BODY ----------------
    "squat": {
        "type": "rep",
        "joints": ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
        "up": 110,
        "down": 160
    },

    "deadlift": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"],
        "up": 100,
        "down": 160
    },

    "romanian_deadlift": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"],
        "up": 70,
        "down": 160
    },

    "hip_thrust": {
        "type": "rep",
        "joints": ["RIGHT_SHOULDER", "RIGHT_HIP", "RIGHT_KNEE"],
        "up": 90,
        "down": 160
    },

    "leg_extension": {
        "type": "rep",
        "joints": ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
        "up": 80,
        "down": 160
    },

    "leg_raises": {
        "type": "rep",
        "joints": ["RIGHT_HIP", "RIGHT_KNEE", "RIGHT_ANKLE"],
        "up": 40,
        "down": 160
    },

    # ---------------- CORE / HOLD ----------------
    "russian_twist": {
        "type": "rep",   # handled separately later
        "joints": ["RIGHT_SHOULDER", "LEFT_SHOULDER", "RIGHT_HIP"],
        "up": 30,
        "down": 90
    },

    "plank": {
        "type": "hold",
        "joints": [],
        "up": 0,
        "down": 0
    }
}

# ======================================================
# FORM COLORS
# ======================================================
COLORS = {
    "correct": (0, 255, 0),
    "wrong": (0, 0, 255),
    "unknown": (255, 165, 0)
}
