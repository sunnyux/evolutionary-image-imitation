from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
from ga.fitness import draw_voronoi_image, drawing_fitness_function

genotype_length = 5


def histogram_intersection(hist_1, hist_2):
    minima = np.minimum(hist_1, hist_2)
    intersection = np.true_divide(np.sum(minima), np.sum(hist_2))
    return intersection

def get_averaged_channel_histogram_intersection(hist_1, hist_2):
    avg_intersections_over_channels = 0
    for c in range(3):
        intersection = histogram_intersection(hist_1[c], hist_2[c])
        avg_intersections_over_channels += intersection
    avg_intersections_over_channels = avg_intersections_over_channels / 3
    return avg_intersections_over_channels


def select_blocks(image: Image, block_size: int, nr_offspring, selection, generations):
    width, height = image.size
    num_blocks_width = int(width / block_size)
    num_blocks_height = int(height / block_size)

    num_blocks_total = num_blocks_width * num_blocks_height
    weights = np.full(num_blocks_total, 1 / num_blocks_total)

    blocks = np.arange(num_blocks_total)
    block_colors = []
    block_coordinates = []
    for i in range(num_blocks_width):
        for j in range(num_blocks_height):
            color = np.array([0,0,0]).astype(int)
            size = 0
            block_center_x = i * block_size + (block_size // 2)
            block_center_y = j*block_size + (block_size // 2)

            block_coordinates.append(np.array([block_center_x, block_center_y]).astype(int))
            for k in range(-block_size // 2, block_size // 2):
                for l in range (-block_size // 2, block_size // 2):
                    pixel_x = block_center_x + k
                    pixel_y = block_center_y + l
                    
                    if pixel_x < 0 or pixel_x >= width or pixel_y < 0 or pixel_y >= height:
                        continue
                    r, g, b = image.getpixel((pixel_x, pixel_y))

                    color = np.add(color, np.array([r,g,b]).astype(int))
                    size += 1
            color = np.divide(color, size)
            block_colors.append(color)
    block_colors = np.array(block_colors).astype(int)
    block_coordinates  = np.array(block_coordinates).astype(int)

    fitness_scores_over_generations = []
    all_time_best_offspring = []
    histogram_intersections = []
    
    expected_hist = np.array(image.histogram()).reshape((3,256))
    for g in range(generations):
        offspring = []
        offspring_blocks = []
        for i in range(nr_offspring):
            chosen_blocks = np.random.choice(blocks, 100, replace=False, p=weights)
            offspring_blocks.append(chosen_blocks)

            individual = []
            for chosen_block in chosen_blocks:
                block_color = block_colors[chosen_block]
                block_coordinate = block_coordinates[chosen_block]
                for coordinate in block_coordinate:
                    individual.append(coordinate)
                
                for color in block_color:
                    individual.append(color)
            offspring.append(individual)

        offspring = np.array(offspring)
        fitness_values = drawing_fitness_function(offspring, image)
        zipped = zip(offspring, offspring_blocks, fitness_values)
        sorted_offsprings = sorted(zipped, reverse=False, key=lambda x: x[2])
        picked_offspring = sorted_offsprings[:selection]
        best_offspring = picked_offspring[0]

        if (len(all_time_best_offspring) == 0) or (best_offspring[2] >= all_time_best_offspring[2]):
            all_time_best_offspring = best_offspring
        fitness_scores_over_generations.append(best_offspring[2])
        
        
        img = draw_voronoi_image(best_offspring[0], width, height)

        actual_hist = np.array(img.histogram()).reshape((3,256))
        average_intersection_van_gogh = get_averaged_channel_histogram_intersection(expected_hist, actual_hist)

        histogram_intersections.append(average_intersection_van_gogh)

        
        counts = np.zeros(num_blocks_total)
        for o in picked_offspring:
            for b in o[1]:
                counts[b] += 1
        weights = np.divide(counts, selection * 100)
        

    fitness_scores_over_generations = np.array(fitness_scores_over_generations)
    histogram_intersections = np.array(histogram_intersections)
    return fitness_scores_over_generations, histogram_intersections, all_time_best_offspring


