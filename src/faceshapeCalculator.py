import numpy as np

def calculate_face_shape(points):
    # Estimate the width of the forehead
    forehead_width = np.linalg.norm(np.array(points[16]) - np.array(points[0]))

    # Estimate the width of the cheekbones
    cheekbone_width = np.linalg.norm(np.array(points[13]) - np.array(points[3]))

    # Estimate the jaw width (bit more complex as we need to consider multiple points)
    jaw_width = np.linalg.norm(np.array(points[11]) - np.array(points[5]))

    # Estimate the face length
    face_length = np.linalg.norm(np.array(points[8]) - np.mean([np.array(points[0]), np.array(points[16])], axis=0))

    # Define the ratios
    fw_ratio = forehead_width / cheekbone_width
    fj_ratio = forehead_width / jaw_width
    cj_ratio = cheekbone_width / jaw_width
    fl_ratio = face_length / cheekbone_width

    # Classify based on the ratios
    if fw_ratio >= 0.8 and fj_ratio >= 0.8 and fl_ratio < 1.3:
        return "Square"
    elif fl_ratio >= 1.3:
        return "Rectangle/Oblong"
    elif cj_ratio >= 1.1:
        return "Round"
    elif fw_ratio > 1.0 and fj_ratio > 1.0 and fl_ratio < 1.3:
        return "Heart"
    elif fl_ratio >= 1.3 and fw_ratio < 1.0:
        return "Diamond"
    else:
        return "Oval"