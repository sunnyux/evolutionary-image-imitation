import time

import numpy as np
from PIL import Image
from multiprocess import Pool, cpu_count

from ga import selection, variation
from ga.fitness import drawing_fitness_function, draw_voronoi_image, draw_voronoi_matrix
from ga.population import Population
from ga.util import NUM_VARIABLES_PER_POINT, IMAGE_SHRINK_SCALE, REFERENCE_IMAGE, optimize_individual_color


class Evolution:
    def __init__(self,
                 num_points,
                 reference_image: Image,
                #  image_name: "van_gogh",
                 evolution_type='p+o',
                 population_size=200,
                 generation_budget=-1,
                 evaluation_budget=-1,
                 crossover_method="ONE_POINT",
                 mutation_method="DEFAULT",
                 mutation_probability='inv_mutable_genotype_length',
                 num_features_mutation_strength=.25,
                 num_features_mutation_strength_decay=None,
                 num_features_mutation_strength_decay_generations=None,
                 mutation_candidate_size=5, 
                 optimize_color=False,
                 optimize_color_method="AVERAGE", # or "DOMINANT", or "COORD"
                 selection_name='tournament_2',
                 initialization='RANDOM',
                 noisy_evaluations=False,
                 verbose=False,
                 generation_reporter=None,
                 seed=0):

        self.reference_image: Image = reference_image.copy()
        self.reference_image.thumbnail((int(self.reference_image.width / IMAGE_SHRINK_SCALE),
                                        int(self.reference_image.height / IMAGE_SHRINK_SCALE)),
                                       Image.ANTIALIAS)
        self.reference_image_array = np.asarray(self.reference_image)

        num_variables = num_points * NUM_VARIABLES_PER_POINT # 100 * 5 = 500
        feature_intervals = []
        for i in range(num_variables):
            if i % NUM_VARIABLES_PER_POINT == 0:  # X
                feature_intervals.append([0, self.reference_image.width]) # self.reference_image.width: 50
            elif i % NUM_VARIABLES_PER_POINT == 1:  # Y
                feature_intervals.append([0, self.reference_image.height]) # self.reference_image.height: 39
            else:  # color (RGB)
                feature_intervals.append([0, 256])

        self.num_points = num_points
        self.feature_intervals = feature_intervals
        self.evolution_type = evolution_type
        self.population_size = population_size
        self.generation_budget = generation_budget
        self.evaluation_budget = evaluation_budget
        self.mutation_method = mutation_method
        self.mutation_probability = mutation_probability
        self.num_features_mutation_strength = num_features_mutation_strength
        self.num_features_mutation_strength_decay = num_features_mutation_strength_decay
        self.num_features_mutation_strength_decay_generations = num_features_mutation_strength_decay_generations
        self.mutation_candidate_size = mutation_candidate_size
        self.optimize_color = optimize_color
        self.optimize_color_method = optimize_color_method
        self.selection_name = selection_name
        self.noisy_evaluations = noisy_evaluations
        self.verbose = verbose
        self.generation_reporter = generation_reporter
        self.crossover_method = crossover_method
        self.num_evaluations = 0
        self.initialization = initialization
        # self.image_name = image_name

        seed = int(time.time())
        np.random.seed(seed)
        self.seed = seed

        # set feature intervals to be a np.array
        if type(feature_intervals) != np.array:
            self.feature_intervals = np.array(feature_intervals, dtype=object)

        # check that tournament size is compatible
        if 'tournament' in selection_name:
            self.tournament_size = int(selection_name.split('_')[-1])
            if self.population_size % self.tournament_size != 0:
                raise ValueError('The population size must be a multiple of the tournament size')

        # set up population and elite
        self.genotype_length = len(feature_intervals)
        self.population = Population(self.population_size, self.genotype_length, self.initialization)
        self.elite = None
        self.elite_fitness = np.inf

        # attributes added for experiments
        self.std = None
        self.avg = None

        # set up mutation probability if set to default "inv_mutable_genotype_length"
        if mutation_probability == 'inv_genotype_length':
            self.mutation_probability = 1 / self.genotype_length
        elif mutation_probability == "inv_mutable_genotype_length":
            num_unmutable_features = 0
            self.mutation_probability = 1 / (self.genotype_length - num_unmutable_features)

        # incompatibilities
        if self.evolution_type == 'p+o' and self.noisy_evaluations:
            raise ValueError(
                "Using P+O is not compatible with noisy evaluations (you would need to re-evaluate the parents every generation, which is expensive)")
        elif 'age_reg' in self.evolution_type:
            print(
                "Warning: using noisy evaluations but age regularized evolution does not re-evaluate the entire population every generation")

    def __update_elite(self, population):
        best_fitness_idx = np.argmin(population.fitnesses)
        best_fitness = population.fitnesses[best_fitness_idx]
        if self.noisy_evaluations or best_fitness < self.elite_fitness:
            self.elite = population.genes[best_fitness_idx, :].copy()
            self.elite_fitness = best_fitness

    def __classic_generation(self, merge_parent_offspring=False):
        # create offspring population
        offspring = Population(self.population_size, self.genotype_length, self.initialization)
        offspring.genes[:] = self.population.genes[:]
        offspring.shuffle()
        # variation
        offspring.genes = variation.crossover(offspring.genes, self.crossover_method)
        offspring.genes = variation.mutate(offspring.genes, self.feature_intervals, self.reference_image_array,
                                           self.reference_image, method=self.mutation_method,
                                           mutation_probability=self.mutation_probability,
                                           num_features_mutation_strength=self.num_features_mutation_strength,
                                           mutation_candidate_size=self.mutation_candidate_size)
        if self.optimize_color:
            for i in range(self.population_size):
                offspring.genes[i] = optimize_individual_color(offspring.genes[i], self.num_points, 
                                                               self.reference_image_array, self.optimize_color_method)

        # evaluate offspring
        offspring.fitnesses = drawing_fitness_function(offspring.genes,
                                                       self.reference_image)
        self.num_evaluations += len(offspring.genes)

        self.__update_elite(offspring)

        # calculate the std and avg of this population
        self.std = np.std(offspring.fitnesses)
        self.avg = np.mean(self.population.fitnesses)

        # selection
        if merge_parent_offspring:
            # p+o mode
            self.population.stack(offspring)
        else:
            # just replace the entire thing
            self.population = offspring

        self.population = selection.select(self.population, self.population_size,
                                           selection_name=self.selection_name)

    def run(self, run):
        data = []

        print("Initializing population")
        self.population.initialize(self.feature_intervals, self.reference_image_array)
        print("Done initializing population.")

        self.population.fitnesses = drawing_fitness_function(self.population.genes,
                                                             self.reference_image)
        self.num_evaluations = len(self.population.genes)

        best_fitness_idx = np.argmin(self.population.fitnesses)
        best_fitness = self.population.fitnesses[best_fitness_idx]
        # if best_fitness > self.elite_fitness: # original
        if best_fitness < self.elite_fitness:
            self.elite = self.population.genes[best_fitness_idx, :].copy()
            self.elite_fitness = best_fitness

        # calculate the std and avg of this population
        self.std = np.std(self.population.fitnesses)
        self.avg = np.mean(self.population.fitnesses)

        start_time_seconds = time.time()

        # run generation_budget
        i_gen = 0
        while True:
            if self.num_features_mutation_strength_decay_generations is not None:
                if i_gen in self.num_features_mutation_strength_decay_generations:
                    self.num_features_mutation_strength *= self.num_features_mutation_strength_decay

            if self.evolution_type == 'classic':
                self.__classic_generation(merge_parent_offspring=False)
            elif self.evolution_type == 'p+o':
                self.__classic_generation(merge_parent_offspring=True)
            else:
                raise ValueError('unknown evolution type:', self.evolution_type)

            # generation terminated
            i_gen += 1

            if self.verbose:
                print('generation:', i_gen, 'best fitness:', self.elite_fitness, 'avg. fitness:', self.avg)

            # the data gathered depends on the experiment, so running the notebook as is will not work
            # data.append([self.avg, self.elite_fitness])

            data.append({"num-generations": i_gen,
                         "num-evaluations": self.num_evaluations,
                         "time-elapsed": time.time() - start_time_seconds,
                         "best-fitness": self.elite_fitness,
                         "fitness-std": self.std,
                         "fitness-avg": self.avg,
                         "crossover-method": self.crossover_method,
                         "population-size": self.population_size, "num-points": self.num_points,
                         "initialization": self.initialization,
                         "optimize_color": self.optimize_color,
                         "optimize_color_method": self.optimize_color_method,
                         "seed": self.seed,
                         "run": run})
            if self.generation_reporter is not None:
                self.generation_reporter(
                    {"num-generations": i_gen, "num-evaluations": self.num_evaluations,
                     "time-elapsed": time.time() - start_time_seconds}, self)

            if 0 < self.generation_budget <= i_gen:
                print("generation budget reached")
                break
            if 0 < self.evaluation_budget <= self.num_evaluations:
                print("evaluation budget reached")
                break

            # check if evolution should terminate because optimum reached or population converged
            if self.population.is_converged():
                print("population converged")
                break

        draw_voronoi_image(self.elite, self.reference_image.width, self.reference_image.height,
                           scale=IMAGE_SHRINK_SCALE) \
            .save(
            f"./img/final/final_{self.seed}_{run}_{self.population_size}_{self.crossover_method}_{self.num_points}_{self.initialization}_{self.optimize_color_method}_{self.generation_budget}.png")
        return data


if __name__ == '__main__':
    evo = Evolution(100,
                    REFERENCE_IMAGE,
                    evolution_type='p+o',
                    population_size=100,
                    generation_budget=300,
                    crossover_method='ONE_POINT',
                    initialization='RANDOM',
                    num_features_mutation_strength=.25,
                    num_features_mutation_strength_decay=None,
                    num_features_mutation_strength_decay_generations=None,
                    selection_name='tournament_4',
                    noisy_evaluations=False,
                    verbose=True)
    evo.run()
