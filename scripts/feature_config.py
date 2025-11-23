# Canonical window level feature list used for training
# Order matters and must match the columns in *_windows_mouth.csv

WINDOW_FEATURE_COLUMNS = [
    # Arm and hand features
    "hand_to_mouth_min_mean",
    "hand_to_mouth_min_std",
    "hand_to_mouth_min_min",
    "hand_to_mouth_min_max",
    "hand_to_mouth_min_slope",

    "hand_speed_max_mean",
    "hand_speed_max_std",
    "hand_speed_max_min",
    "hand_speed_max_max",
    "hand_speed_max_slope",

    "wrist_rel_shoulder_y_min_mean",
    "wrist_rel_shoulder_y_min_std",
    "wrist_rel_shoulder_y_min_min",
    "wrist_rel_shoulder_y_min_max",
    "wrist_rel_shoulder_y_min_slope",

    "elbow_angle_deg_min_mean",
    "elbow_angle_deg_min_std",
    "elbow_angle_deg_min_min",
    "elbow_angle_deg_min_max",
    "elbow_angle_deg_min_slope",

    # Mouth features per window
    "mouth_width_mean",
    "mouth_width_std",
    "mouth_width_min",
    "mouth_width_max",
]
