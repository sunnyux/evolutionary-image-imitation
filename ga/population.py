import numpy as np
from ga.util import *
from PIL import Image


class Population:
    def __init__(self, population_size, genotype_length, initialization):
        self.genes = np.empty(shape=(population_size, genotype_length), dtype=int)
        self.fitnesses = np.zeros(shape=(population_size,))
        self.initialization = initialization

    def initialize(self, feature_intervals, reference_image_array):
        # feature_intervals: lenth of genotype, 500; 
        # containing repeating values of [0 50], [0 39], [0 256], [0 256], [0 256]
        n = self.genes.shape[0] # 100, popilation_size
        l = self.genes.shape[1] # 500, genotype_length

        if self.initialization == "RANDOM":
            for i in range(l):
                init_feat_i = np.random.randint(low=feature_intervals[i][0],
                                                        high=feature_intervals[i][1], size=n)
                self.genes[:, i] = init_feat_i # filling n columns
        
        # assign the randomly generated points the color of the original image
        elif self.initialization == "PICK_SINGLE_COORD_COLOR":
            for i in range(n):
                for j in range(0, l, 5):
                    rng = np.random.default_rng()
                    # xs = rng.integers(low=0, high=width, size=num_points - len(feature_coords))
                    # ys = rng.integers(low=0, high=height, size=num_points - len(feature_coords))
                    # cs = set(zip(xs, ys))    
                    x = rng.integers(low=feature_intervals[j][0], high=feature_intervals[j][1])
                    y = rng.integers(low=feature_intervals[j+1][0], high=feature_intervals[j+1][1])
                    color = get_coord_color_from_image(reference_image_array, x, y)
                    self.genes[i, j : j+5] = np.concatenate(([x, y], color))
            # np.set_printoptions(threshold=np.inf)
            
        elif self.initialization == "PICK_AVERAGE_CELL_COLOR":
            for i in range(n):
                self.genes[i] = initialize_one_with_best_voronoi_color(feature_intervals, reference_image_array, "AVERAGE")

        elif self.initialization == "PICK_DOMINANT_CELL_COLOR":
            for i in range(n):
                self.genes[i] = initialize_one_with_best_voronoi_color(feature_intervals, reference_image_array, "DOMINANT")
                
        else:
            raise Exception("Unknown initialization method ", self.initialization)

    def stack(self, other):
        self.genes = np.vstack((self.genes, other.genes))
        self.fitnesses = np.concatenate((self.fitnesses, other.fitnesses))

    def shuffle(self):
        random_order = np.random.permutation(self.genes.shape[0])
        self.genes = self.genes[random_order, :]
        self.fitnesses = self.fitnesses[random_order]

    def is_converged(self):
        return len(np.unique(self.genes, axis=0)) < 2

    def delete(self, indices):
        self.genes = np.delete(self.genes, indices, axis=0)
        self.fitnesses = np.delete(self.fitnesses, indices)
