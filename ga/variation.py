import numpy as np
from ga.util import *
from ga.fitness import compute_difference


def crossover(genes, method="ONE_POINT"):
    parents_1 = np.vstack((genes[:len(genes) // 2], genes[:len(genes) // 2]))
    parents_2 = np.vstack((genes[len(genes) // 2:], genes[len(genes) // 2:]))

    if method == "ONE_POINT":
        crossover_points = np.random.randint(0, genes.shape[1], size=genes.shape[0])
        offspring = np.zeros(shape=genes.shape, dtype=int)

        for i in range(len(genes)):
            offspring[i,:] = np.where(np.arange(genes.shape[1]) <= crossover_points[i], parents_1[i,:], parents_2[i,:])
    elif method == "TWO_POINT":
        l = genes.shape[0]
        crossover_point_1 = np.random.randint(0, l)
        crossover_point_2 = np.random.randint(crossover_point_1, l)

        offspring = np.zeros(shape=genes.shape, dtype=int)

        for i in range(l):
            if i <= crossover_point_1:
                offspring[i, :] = parents_1[i, :]
            elif i > crossover_point_1 and i  <= crossover_point_2:
                offspring[i, :] = parents_2[i, :]
            else:
                offspring[i, :] = parents_1[i, :]
    else:
        raise Exception("Unknown crossover method")

    return offspring


def mutate(genes, feature_intervals, reference_image_array, reference_image, method="DEFAULT", 
           mutation_probability=0.1, num_features_mutation_strength=0.05, 
           mutation_candidate_size=5):
    
    offspring = genes.copy()

    if method == "DEFAULT":
        # mask_mut has a shape of (100, 500) mutatuon_probability of 1 sets all to True and 0 to all False
        mask_mut = np.random.choice([True, False], size=genes.shape,
                                    p=[mutation_probability, 1 - mutation_probability])

        mutations = generate_plausible_mutations(genes, feature_intervals,
                                                num_features_mutation_strength)
                                                
        offspring = np.where(mask_mut, mutations, genes)

    elif method == "IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR" or method == "IMPROVED_REPLACEMENT_DOMINANT_CELL_COLOR": # assuming best_color=True since only replacing x and y coords
        # Remove some point based on mutation_probability and tests a set of random candidate points to replace it. 
        # The one that yields the best fitness of the bunch is accepted and kept.        
        choose_color_method = "AVERAGE"
        if method == "IMPROVED_REPLACEMENT_DOMINANT_CELL_COLOR":
            choose_color_method = "DOMINANT"
            
        offspring = genes.copy()

        n = genes.shape[0] # popilation_size
        l = genes.shape[1] # num_points * NUM_VARIABLES_PER_POINT, genotype_length
        num_points = l // NUM_VARIABLES_PER_POINT

        width = len(reference_image_array[0]) # 50
        height = len(reference_image_array) # 39

        for i in range(n): # for each individual
            individual = genes[i]

            individual_coords = get_coordinates_only(individual, width, height)

            ind_mask = np.random.choice([True, False], size=(len(individual)//5), p=[mutation_probability, 1 - mutation_probability])

            for g in range(0, l//NUM_VARIABLES_PER_POINT): # for each coordinte in individual
                best_replacement = individual
                if ind_mask[g]:
                    best_replacement_fitness = compute_difference(best_replacement, reference_image)

                    all_coords = individual_coords.copy()
                    while len(all_coords) < len(individual_coords) + mutation_candidate_size:
                        rng = np.random.default_rng()
                        x = rng.integers(low=0, high=width)
                        y = rng.integers(low=0, high=height)
                        # x = np.random.randint(0, width)
                        # y = np.random.randint(0, height)
                        all_coords.add((x, y))
                    candidate_replacements = list(all_coords.difference(individual_coords))
                    
                    for c in candidate_replacements:
                        new_individual_coords = list(individual_coords.copy())
                        new_individual_coords[g] = c
                        labels = create_voronoi_site_labels(new_individual_coords, width, height)
                        replaced_individual = labels_to_optimal_colored_genes(labels, new_individual_coords, num_points, reference_image_array, method=choose_color_method)

                        fitness = compute_difference(replaced_individual, reference_image)
                        if fitness <= best_replacement_fitness:
                            best_replacement = replaced_individual
                            best_replacement_fitness = fitness
                    
                offspring[i] = best_replacement
                
    else:
        raise Exception("Unknown mutation method")
        
    return offspring


def change_to_original_color(genes, reference_image):
    mutations = np.zeros(shape=genes.shape)
    n = genes.shape[0] # 100, popilation_size
    l = genes.shape[1] # 500, genotype_length
    for i in range(n):
        for j in range(0, l, 5):
            x = genes[i][j]
            y = genes[i][j+1]
            color = get_coord_color_from_image(reference_image, x, y)
            mutations[i, j : j+5] = np.concatenate(([x, y], color))

    mutations = mutations.astype(int)
    return mutations


def generate_plausible_mutations(genes, feature_intervals,
                                 num_features_mutation_strength=0.25):
    mutations = np.zeros(shape=genes.shape)

    for i in range(genes.shape[1]):
        range_num = feature_intervals[i][1] - feature_intervals[i][0]
        low = -num_features_mutation_strength / 2
        high = +num_features_mutation_strength / 2

        mutations[:, i] = range_num * np.random.uniform(low=low, high=high,
                                                        size=mutations.shape[0])
        mutations[:, i] += genes[:, i]

        # Fix out-of-range
        mutations[:, i] = np.where(mutations[:, i] > feature_intervals[i][1],
                                   feature_intervals[i][1] - 1, mutations[:, i])
        mutations[:, i] = np.where(mutations[:, i] < feature_intervals[i][0],
                                   feature_intervals[i][0] - 1, mutations[:, i])

    mutations = mutations.astype(int)
    return mutations
