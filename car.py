# Reference for calculating car edges: Programmers' Place
# https://www.youtube.com/@programmersplace6203

import pygame
from math import radians, cos, sin, pow
import constants as cons
from helper import translate, get_euclidean_distance

class Car:
    def __init__(self, track, angle, center):
        """
        Constructor for the car object
        """
        self.vel = 30
        self.turn_angle = 10
        self.size = (30, 50)
        self.center = center
        self.corners = []
        self.edge_points = []
        self.distances = []
        self.distance_travelled = 0
        self.distance_from_centre = 0
        self.angle = angle
        self.car = pygame.image.load("images/car.png").convert_alpha()
        self.car = pygame.transform.scale(self.car, self.size)
        self.crashed = False
        self.reached_finish = False
        self.update(track)

    def check(self, track):
        """
        Checks if the car has crashed by checking the pixel color at every corner of the car,
        and sets the crashed attribute of car to True
        """
        for corner in self.corners:
            try:
                if track.surface.get_at(corner) == (255, 255, 255):
                    if get_euclidean_distance(self.center, track.end) < 100:
                        self.reached_finish = True
                    self.crashed = True
                    return
            except IndexError:
                # when we get negtive values for coordinates,
                # we give the car the benefit of doubt and say its not crashed
                self.crashed = False
        self.crashed = False

    def draw(self, screen):
        """
        Draws the car on the screen object
        """
        rotated_car = pygame.transform.rotate(self.car, self.angle)
        screen.blit(rotated_car, rotated_car.get_rect(center=self.center).topleft)

    def update(self, track_copy):
        """
        Updates the car's distances from three edges, the left, right and forward ones
        """
        angles = [radians(360 - self.angle), radians(90 - self.angle), radians(180 - self.angle)]
        edge_points = []
        distances = []
        for angle in angles:
            distance = 0
            edge_x, edge_y = self.center
            try:
                while track_copy.get_at((edge_x, edge_y)) != (255, 255, 255):
                    edge_x = int(self.center[0] + distance * cos(angle))
                    edge_y = int(self.center[1] + distance * sin(angle))
                    distance += 1
            except IndexError:
                pass
            edge_points.append((edge_x, edge_y))
            distances.append(distance)

        self.edge_points = edge_points
        self.distances = distances
        self.distance_from_centre += abs(self.distances[0] - self.distances[2])

    def move(self, choice, track):
        """
        Moves the car by the specified velocity, and also turns it according to the output of
        the neural network (0, 1 or 2)
        """
        self.check(track)
        if choice == 0:
            self.angle += self.turn_angle
        elif choice == 1:
            self.angle -= self.turn_angle
        self.center = translate(
            self.center, 90 - self.angle, self.vel)
        dist = get_euclidean_distance(self.size, (0, 0)) / 2

        # 120 |---------| 60
        #     |o       o|
        #     |         |
        #     |         |
        #     |         |
        # 240 |_________| 300

        self.corners = [translate(self.center, 60 - self.angle, dist),
                        translate(self.center, 120 - self.angle, dist),
                        translate(self.center, 240 - self.angle, dist),
                        translate(self.center, 300 - self.angle, dist)]
        self.distance_travelled += self.vel

    def get_fitness(self, end):
        """
        Gets the fitness of the car object by the formula D/d - c/(G-i)^2
        """
        distance_fitness = self.distance_travelled / get_euclidean_distance(self.center, end)
        centre_negative_fitness = self.distance_from_centre / pow(cons.MAX_GENERATIONS - cons.GENERATION + 2, 2)
        return distance_fitness - centre_negative_fitness

    def get_distance_from_end(self, end):
        """
        Returns the distance of the car from the end of the track
        """
        return get_euclidean_distance(self.center, end)

    def is_crashed(self):
        """
        Returns True if the car has crashed, False otherwise
        """
        return self.crashed