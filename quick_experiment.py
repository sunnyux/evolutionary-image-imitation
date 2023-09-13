from time import time
from ga.evolution import Evolution
from ga.fitness import draw_voronoi_image
from ga.util import IMAGE_SHRINK_SCALE, REFERENCE_IMAGE
import sys

# This file is for quick experiments with the new funcions 
# such that I don't have to go through all notebook settings.
# It is not part of the main codebase.

def reporter(time, evo):
    return

def run_algorithm(settings):
    population_size, generation_budget, crossover_method, mutation_method, mutation_probability, \
        mutation_candidate_size, optimize_color, optimize_color_method, selection_name, initialization = settings

    output_file = f"./output/{population_size}_{generation_budget}_{crossover_method}_{mutation_method}_{mutation_probability}_{mutation_candidate_size}_{optimize_color}_{optimize_color_method}_{selection_name}_{initialization}.txt"
    with open(output_file, "w") as f:
        original_stdout = sys.stdout
        sys.stdout = f

        print("population_size, generation_budget, crossover_method, mutation_method, mutation_probability, mutation_candidate_size, optimize_color, optimize_color_method, selection_name, initialization")
        print(settings)
    
        start = time()
        data = []
        evo = Evolution(100, # num_points, # use only 100 or 50, use 50 will make our life easier
                        REFERENCE_IMAGE,
                        # evolution_type='p+o',
                        population_size=population_size,
                        generation_budget=generation_budget,
                        # evaluation_budget=-1,
                        crossover_method=crossover_method,
                        mutation_method=mutation_method,
                        mutation_probability=mutation_probability, # default 'inv_genotype_length'
                        num_features_mutation_strength=.25, # only in DEFAULT mutation_method
                        # num_features_mutation_strength_decay=None,
                        # num_features_mutation_strength_decay_generations=None,
                        mutation_candidate_size=mutation_candidate_size,  # doesn't matter for DEFAULT mutation_method
                        optimize_color=optimize_color,
                        optimize_color_method=optimize_color_method, # "AVERAGE", "DOMINANT", or "COORD"; whatever when optimize_color==False
                        selection_name=selection_name, # tournament_4 was the original
                        initialization=initialization, # initialization method
                        # noisy_evaluations=False,
                        verbose=True,
                        generation_reporter=reporter,
                        seed=0)
        data = evo.run(9)
        time_spent = time() - start
        print(f"Time spent: {time_spent} seconds")

        sys.stdout = original_stdout

    

'''
Initialization methods: RANDOM, PICK_SINGLE_COORD_COLOR, PICK_AVERAGE_CELL_COLOR, PICK_DOMINANT_CELL_COLOR
Crossover methods: ONE_POINT, TWO_POINT
Mutation methods: DEFAULT, IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR, IMPROVED_REPLACEMENT_DOMINANT_CELL_COLOR
Color optimization methods: AVERAGE, DOMINANT, COORD

Thinking process: 
1. Find the best method to assign color to each cell, this reduces the problem to finding the best coordinates
2. Start with using the exact color of the coordinate as the color of the cell 
    (thus PICK_SINGLE_COORD_COLOR, or get_coord_color_from_image)
3. Colors are calculated on pixel comparison, thus knowing all the coordinates of the pixels in the cell and 
    the color of these pixels in the original image is important
    (thus create_voronoi_site_labels, each cell in the voronoi diagram is assigned a label)
4. Knowing the original colors in each cell, we can optimize the color of the cell
    (thus "AVERAGE" and "DOMINANT" methods for color optimization)
5. Using a greedy approach to always find a better solution (thus IMPROVED_REPLACEMENT mutation methods)

TODO (high priority): define the baseline settings for the experiment, in other words, 
    find the set of values for all input variables of the class Evolution
'''

# population_size, generation_budget, crossover_method, mutation_method, mutation_probability, \
#         mutation_candidate_size, optimize_color, optimize_color_method, initialization
def get_settings(population_size=100, generation_budget=500, crossover_method="ONE_POINT", 
                 mutation_method="DEFAULT", mutation_probability='inv_genotype_length',
                 mutation_candidate_size=5, optimize_color=False, optimize_color_method="AVERAGE", 
                 selection_name='tournament_4', initialization="RANDOM"):
    
    print(f"population_size={population_size}, generation_budget={generation_budget}, crossover_method={crossover_method}, \
          mutation_method={mutation_method}, mutation_probability={mutation_probability}, mutation_candidate_size={mutation_candidate_size}, \
          optimize_color={optimize_color}, optimize_color_method={optimize_color_method}, \
          selection_name={selection_name}, initialization={initialization}")
    
    return (population_size, generation_budget, crossover_method, mutation_method, mutation_probability, 
            mutation_candidate_size, optimize_color, optimize_color_method, selection_name, initialization)
    

'''
hypothesis 1 (small and quick): initializing by picking the average cell color represented by
    the coordinate, the generated image will have a much better fitness compared to randomly assigning the colors and by picking
    the dominant color of the cell.
EXPERIMENT SETTINGS: all 4 initialization methods, baseline setting for everything other than initialization
    change the code in evolution.py such that no variation is performed, only the initialization method is performed
Result and discussion: from best to worst fitness: PICK_AVERAGE_CELL_COLOR, PICK_DOMINANT_CELL_COLOR, PICK_SINGLE_COORD_COLOR, RANDOM
'''
data_RANDOM = []
for i in range(10):
    data = run_algorithm(get_settings(generation_budget=1, initialization="RANDOM"))
    data_RANDOM.append(data)
    
# df = pd.DataFrame(data)
# df["time-elapsed"] = df["time-elapsed"].round(0)
# df.to_csv("testing.csv")

    # data = run_algorithm(get_settings(generation_budget=1, initialization="PICK_AVERAGE_CELL_COLOR"))
    # data = run_algorithm(get_settings(generation_budget=1, initialization="PICK_DOMINANT_CELL_COLOR"))
    # data = run_algorithm(get_settings(generation_budget=1, initialization="PICK_SINGLE_COORD_COLOR"))



'''
hypothesis 2: ONE_POINT and TWO_POINT crossover methods do not differ much in terms of fitness when the color is optimized 
    (i.e. setting optimize_color=True, and optimize_color_method) because the two crossover methods do not focus on the
    evolvement of the coordinate, in other words, the shape representation of the image.
EXPERIMENT SETTINGS: the best initialization method from hypothesis 1 ("PICK_AVERAGE_CELL_COLOR), 
    all crossover methods, all optimize_color_method, baseline others (should have 6 in total)
Result and discussion: all in one plot, observe the overlaps
'''
# data = run_algorithm(get_settings(initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   crossover_method="ONE_POINT", optimize_color=True, optimize_color_method="COORD"))
# data = run_algorithm(get_settings(initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   crossover_method="ONE_POINT", optimize_color=True, optimize_color_method="DOMINANT")) # doesn't really improve
# data = run_algorithm(get_settings(initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   crossover_method="ONE_POINT", optimize_color=True, optimize_color_method="AVERAGE"))

# data = run_algorithm(get_settings(initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   crossover_method="TWO_POINT", optimize_color=True, optimize_color_method="COORD"))
# data = run_algorithm(get_settings(initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   crossover_method="TWO_POINT", optimize_color=True, optimize_color_method="DOMINANT")) # also doesn't really improve
# data = run_algorithm(get_settings(generation_budget=2, initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   crossover_method="TWO_POINT", optimize_color=True, optimize_color_method="AVERAGE")) # this was pretty good


'''
hypothesis 3: the greedy approach (IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR) will perform better than all the none greedy approaches
    previously tested within 500 generations. 
    IMPROVED_REPLACEMENT_DOMINANT_CELL_COLOR is disregarded because the previous hypotheses have shown that setting the cell color 
    to the dominant color does not yield a better fitness compared to setting the cell color to the average color.
    `mutation_probability` and `mutation_candidate_size` is hypothesized to have a great effect on the solutions generated by the 
    greedy mutation approach (IMPROVED_REPLACEMENTS)
EXPERIMENT SETTINGS: best performers from the previous experiments, mix those settings with IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR, 
    a high mutation_probability is required, otherwise there will be no mutation,
    but this also causes the run to be insanely slow, 
    needs to try different combinations of `mutation_probability` and `mutation_candidate_size`
    Just get a table of variables experimented, the number of generations, and the final fitness
Result and discussion: so inffecient, basically brute force lol
'''

# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="RANDOM", 
#                                   population_size=24, mutation_probability=0.4, mutation_candidate_size=5, selection_name='tournament_2'))

# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   population_size=24, mutation_probability=0.4, mutation_candidate_size=10, selection_name='tournament_2'))

# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   population_size=100, mutation_probability=0.2, mutation_candidate_size=5, selection_name='tournament_2'))

# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   population_size=100, mutation_probability=0.4, mutation_candidate_size=5, selection_name='tournament_2'))

# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   population_size=100, mutation_probability=0.2, mutation_candidate_size=5, selection_name='tournament_4'))


# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   population_size=24, mutation_probability=0.4, mutation_candidate_size=5, selection_name='tournament_4'))

# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   population_size=24, mutation_probability=0.4, mutation_candidate_size=10, selection_name='tournament_4'))

# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   population_size=24, mutation_probability=0.2, mutation_candidate_size=10, selection_name='tournament_4'))

# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   population_size=24, mutation_probability=0.2, mutation_candidate_size=5, selection_name='tournament_4'))



# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   population_size=24, mutation_probability=0.2, mutation_candidate_size=5, selection_name='tournament_2'))

# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   population_size=24, mutation_probability=0.2, mutation_candidate_size=10, selection_name='tournament_2'))

# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   population_size=100, mutation_probability=0.2, mutation_candidate_size=5, selection_name='tournament_4'))

# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   population_size=24, mutation_probability=0.6, mutation_candidate_size=5, selection_name='tournament_2'))

# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   population_size=24, mutation_probability=0.6, mutation_candidate_size=5, selection_name='tournament_4'))

# data = run_algorithm(get_settings(mutation_method="IMPROVED_REPLACEMENT_AVERAGE_CELL_COLOR", initialization="PICK_AVERAGE_CELL_COLOR", 
#                                   population_size=24, mutation_probability=0.6, mutation_candidate_size=10, selection_name='tournament_4'))



''' General idea for the IMPROVED_REPLACEMENT mutation method

coord = [(1, 1), (2, 2), (3, 3), (4, 4), (5, 5)] # example
old_fitness = compute_fitness_for(gene)
best_replacement = coord
best_fitness = old_fitness
mask = [1, 0, 1, 0, 1] #example

for i in coord:
    if mask[i] == 1:  # only (1,1), (3,3) and (5,5) will be considered
        randomy generate mutation candidate of size=2 
        candidate_points = [(11, 11) (111, 111)] #example
        for c in candidate_points:
            new_gene = [c, (2, 2), (3, 3), (4, 4), (5, 5)]
            new_fitness = compute_fitness_for(new_gene)
            if new_fitness < best_fitness: #minizing
                best_replacement = new_gene
                best_fitness = new_fitness
        
        gene = best_replacement
'''
        



'''
hypothesis 4 (theoretically infinity amount computation required, so don't know if we should do it): 
    if the previously hypothesis is right for 500 generations,
    the none greedy aproach (or settings with greater variation) will generate better fitness 
    if allowed to run for more generations because the solution in the greedy approach will converge too soon
    (the greedy aproach already converge too soon lmao).
'''




# initialization: 150698
# generation: 50 best fitness: 38779 avg. fitness: 39356.69
# data = run_algorithm((0, 1, "ONE_POINT", "DEFAULT", 100, "RANDOM", generation_budget))

# initialization: 51237
# generation: 50 best fitness: 35926 avg. fitness: 36049.75
# data = run_algorithm((0, 48, "ONE_POINT", "IMPROVED_REPLACEMENT", 100, "PICK_SINGLE_POINT_COLOR", generation_budget))

# initialization, pick_average_voronoi_color: 43883
# data = run_algorithm((0, 48, "ONE_POINT", "DEFAULT", 100, "PICK_AVERAGE_VORNOI_COLOR", generation_budget))

# data = run_algorithm((0, 48, "ONE_POINT", "IMPROVED_REPLACEMENT", 100, "PICK_AVERAGE_VORNOI_COLOR", generation_budget))