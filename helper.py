from math import radians, cos, sin, sqrt

def translate(point, angle, distance):
    """
    Returns a translated point using the distance and angle variables
    x' = x + dcos(angle)
    y' = y + dsin(angle)
    """
    rad = radians(angle)
    return int(point[0] + distance * cos(rad)), \
        int(point[1] + distance * sin(rad))

def get_euclidean_distance(start, end):
    """
    Returns the Euclidean distance between two points
    """
    dist = sqrt((end[0] - start[0]) ** 2 + (end[1] - start[1]) ** 2) / 2
    return 1 if dist == 0 else dist

