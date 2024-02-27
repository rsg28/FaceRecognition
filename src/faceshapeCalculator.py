import numpy as np

def calculate_distance(point1, point2):
    return np.linalg.norm(np.array(point1) - np.array(point2))

def calculate_face_shape(points):
    # Measure the widths and lengths
    forehead_width = calculate_distance(points[0], points[16])
    cheekbone_width = calculate_distance(points[36], points[45])
    jawline_width = calculate_distance(points[4], points[12])
    face_length = calculate_distance(points[8], points[27])

    # Define face shape based on the calculated ratios
    shape = "Undefined"

    # Calculate proportions
    face_ratio = face_length / max(forehead_width, cheekbone_width, jawline_width)
    jaw_forehead_ratio = jawline_width / forehead_width

    # Determine face shape
    if face_ratio <= 1.2:
        if jaw_forehead_ratio <= 0.85:
            shape = "Heart"
        elif jaw_forehead_ratio > 0.85 and jaw_forehead_ratio <= 1.15:
            shape = "Round"
        else:
            shape = "Square"
    else:  # face is longer than it is wide
        if jaw_forehead_ratio <= 0.85:
            shape = "Oval"
        elif jaw_forehead_ratio > 0.85 and jaw_forehead_ratio <= 1.15:
            shape = "Oblong"
        else:
            shape = "Rectangle"

    return shape
