import pygame

class Track:
    def __init__(self, start, end, degree, track_name=None):
        self.start = start
        self.end = end
        self.car_degree = degree
        self.surface = pygame.image.load(track_name).convert_alpha() if track_name is not None else None
        self.copy = self.surface.copy() if track_name is not None else None