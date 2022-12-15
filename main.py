import os
import sys
import neat
import pygame
from car import Car
from track import Track
from constants import *
from plot_results import PlotResults
import constants as cons

pygame.init()
pygame.font.init()
os.environ["SDL_VIDEO_CENTERED"] = "1"
pygame.display.set_caption("Self driving cars")
screen = pygame.display.set_mode(WINDOW_SIZE)
clock = pygame.time.Clock()
font28 = pygame.font.SysFont("Arial", 28)
font20 = pygame.font.SysFont("Arial", 20)
fps = 20

stats = neat.StatisticsReporter()

def load_tracks(file_name):
    """
    Reads the track data from the file track_data.txt
    Assumes that data is in the format of track_name start_y start_x end_x end_y degree
    """
    file = open(file_name, 'r')
    tracks_loc = []
    track_data = file.readlines()
    for track_entry in track_data:
        track_entry = track_entry.strip().split(' ')
        start = int(track_entry[1]), int(track_entry[2])
        end = int(track_entry[3]), int(track_entry[4])
        degree = int(track_entry[5])
        tracks_loc.append(Track(start, end, degree, 'images/' + track_entry[0], ))

    return tracks_loc

track = Track((0, 0), (0, 0), 0)
tracks = load_tracks('track-data.txt')
current_track_ind = 0

def draw_text_centre(text, height):
    """
    Draws text on the centre of the screen at specified height
    """
    render_text = font28.render(text, True, (0, 0, 0))
    screen.blit(render_text, ((WINDOW_SIZE[0] - render_text.get_width()) // 2, height))

def draw_text_right(text, height):
    """
    Draws text on the right of the screen at specified height
    """
    render_text = font20.render(text, True, (0, 0, 0))
    screen.blit(render_text, (WINDOW_SIZE[0] - 350, height))

def draw_screen():
    """
    Draws text on the the screen
    """
    draw_text_centre("Generation: {}".format(cons.GENERATION), 10)
    draw_text_centre("Use up/down arrows to increase/decrease speed of the simulation - {}x".format(fps / 20), 670)
    draw_text_right("\tLast Gen Statistics\t", 10)
    draw_text_right("Cars that reached finish: {}".format(cons.REACHED_FINISH_LAST), 40)

def run(genomes, config):
    global track, fps, current_track_ind
    cons.GENERATION += 1
    nets = []
    ge = []
    cars = []

    for _, g in genomes:
        net = neat.nn.FeedForwardNetwork.create(g, config)
        nets.append(net)
        cars.append(Car(track.copy, track.car_degree, track.start))
        g.fitness = 0
        ge.append(g)

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key in [pygame.K_s, pygame.K_DOWN]:
                    fps -= 5
                    fps = max(fps, 10)
                elif event.key in [pygame.K_w, pygame.K_UP]:
                    fps += 5
                    fps = min(fps, 60)
                elif event.key in [pygame.K_a, pygame.K_LEFT]:
                    current_track_ind -= 1
                    current_track_ind = 4 if current_track_ind == -1 else current_track_ind
                    track = tracks[current_track_ind]
                    # run_track()
                elif event.key in [pygame.K_d, pygame.K_RIGHT]:
                    current_track_ind += 1
                    current_track_ind = 0 if current_track_ind == 5 else current_track_ind
                    track = tracks[current_track_ind]
                    # run_track()

        running_cars = 0
        screen.blit(track.surface, (0, 0))

        for i, car in enumerate(cars):
            if not car.crashed:
                running_cars += 1
                car.distances.append(car.get_distance_from_end(track.end))
                output = nets[i].activate(car.distances)
                choice = output.index(max(output))
                # for turning left or right, predicted by genomes
                car.move(choice, track)
                car.update(track.copy)
                car.draw(screen)
                genomes[i][1].fitness += car.get_fitness(track.end)

        if running_cars == 0:
            cons.reached_finish.append(cons.REACHED_FINISH_LAST)
            cons.REACHED_FINISH_LAST = 0
            for car in cars:
                if car.reached_finish:
                    cons.REACHED_FINISH_LAST += 1
            return

        draw_screen()
        pygame.display.update()
        clock.tick(fps)


def run_track():
    cons.GENERATION = 0
    cons.REACHED_FINISH_LAST = 0
    cons.reached_finish = []
    neat_config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                     neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                     "config.txt")

    population = neat.Population(neat_config)
    population.add_reporter(neat.StdOutReporter(False))
    population.add_reporter(neat.StatisticsReporter())
    population.add_reporter(stats)
    population.run(run, MAX_GENERATIONS)

def main():
    global track, current_track_ind

    for i, loc_track in enumerate(tracks):
        current_track_ind = i
        track = loc_track
        run_track()

        plotter = PlotResults()
        plotter.plot_results([i for i in range(1, len(reached_finish) + 1, 1)], reached_finish,
             "Generation (i)",
             "No. of finishing cars",
             "plots/finishing_cars_{}".format(i+1))

        mean_fitness = stats.get_fitness_mean()
        best_fitness = stats.best_genome()
        print(best_fitness)
        plotter.plot_results([i for i in range(1, len(mean_fitness) + 1, 1)], mean_fitness,
                             "Generation (i)",
                             "Mean Fitness",
                             "plots/mean_fitness_{}".format(i + 1))



if __name__ == '__main__':
    main()