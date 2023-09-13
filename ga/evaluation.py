import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from textwrap import wrap

WIDTH = 2
FONT_SIZE = 12
FIG_SIZE = (5, 4)
ALL_LINESTYLES = ["solid", "dashed", "dashdot"]

def get_cmap(n, name='hsv'):
    '''Returns a function that maps each index in 0, 1, ..., n-1 to a distinct 
    RGB color; the keyword argument name must be a standard mpl colormap name.'''
    return plt.cm.get_cmap(name, n)

def get_linestyles(n):
    return ALL_LINESTYLES*(n//len(ALL_LINESTYLES) + 1)

def compare_fitness_curves(evolution_run_results: list):
    """
    Creates a list with fitness values for each generation for each run in a list of evolution run results.
    
    args:
        evolution_run_results [[]]: A list containing a list with result dictionaries.
    
    returns:
        list: A list containing for each run, a list with the best fitness value per generation.
    """
    print(len(evolution_run_results))
    result = []
    print(result)
    for index, res in enumerate(evolution_run_results):
        result.append([x["best-fitness"] for x in res])
    return result

def plot_fitness_values_over_generations(names, fitness_values: list):
    """
    Creates a line plot according to the list of fitness values for different runs with the given names.
    
    args:
        names (list): name of the run
        fitness_values (list): for each run, the fitness values per generation
        
    returns:
        None
    """
    assert len(names) == len(fitness_values)
    for index, result in enumerate(fitness_values):
        plt.plot(np.arange(len(result)), result, label=names[index], linewidth=WIDTH)
    plt.xlabel("Generation")
    plt.ylabel("Best fitness")
    plt.legend(fontsize=12)

def plot_avg_best_fitness(df: pd.DataFrame, population_size, num_points, oc_method = False):
    """
    Creates a graph for each combination of initialization and crossover method. Each graphs holds plots showing
    the average fitness score for each generation.

    df:
        Dataframe holding all the evaluation data
    population_size:
        int that represent the population size
    num_points:
        int that represent the amount of points

    returns:
        None
    """

    print("Plots showing the average best-fitness over several runs")
    runs = df['run'].max() + 1
    crossover_methods = df['crossover-method'].unique()
    init_methods = df['initialization'].unique()

    if oc_method:
        oc_methods = df['optimize_color_method'].unique()

        n_colours = len(crossover_methods) * len(init_methods) * len(oc_methods)
        cmap = get_cmap(n_colours)
        colours = [cmap(i) for i in range(n_colours)]
        linestyles = get_linestyles(n_colours)
        for init_method in init_methods:
            plt.figure()
            gb_crossovers = df[df['initialization'] == init_method].groupby(['num-generations', 'crossover-method', 'optimize_color_method'])[
                f'best-fitness'].mean().astype(int)
            for method in oc_methods:
                for c_method in crossover_methods:
                    data = gb_crossovers.loc[:, c_method, method]
                    plt.plot(data.index.get_level_values('num-generations'), data.values, label=f'{c_method}, {method}', linewidth=WIDTH, color=colours.pop(0), linestyle=linestyles.pop(0))
        plt.legend(title='Combinations', fontsize=FONT_SIZE)

    else:
        for init_method in init_methods:
            plt.figure()
            gb_crossovers = df[df['initialization'] == init_method].groupby(['num-generations', 'crossover-method'])[f'best-fitness'].mean().astype(int)
            for method in crossover_methods:
                data = gb_crossovers.loc[:, method]
                plt.plot(data.index.get_level_values('num-generations'), data.values, label=method, linewidth=WIDTH)
        plt.legend(title='Crossover methods', fontsize=FONT_SIZE)

    plt.xlabel('Number of generations')
    plt.ylabel(f'Best fitness')
    plt.title(f'Best fitness per generation averaged over {runs} runs', fontsize=FONT_SIZE)
    plt.suptitle(f' Population size: {population_size}, Points: {num_points}, Initialization method: {init_method}', fontsize=FONT_SIZE)

    plt.show()

    # compare crossovers given different initializations: std

def plot_std_best_fitness(df: pd.DataFrame, population_size, num_points):
    """
    Creates a bar plot for each combination of initialization and crossover methods. Each bar represents the standard
    deviation of final fitnesses for each configuration.

    df:
        Dataframe holding all the evaluation data
    population_size:
        int that represent the population size
    num_points:
        int that represent the amount of points

    returns:
        None
    """

    print("Bar plots showing the standard deviation of the final best-fitness")
    runs = df['run'].max() + 1
    crossover_methods = df['crossover-method'].unique()
    init_methods = df['initialization'].unique()

    std_values = []
    x_labels = []
    for init_method in init_methods:
        for method in crossover_methods:
            gb_seed = (df[(df['initialization'] == init_method) & (df['crossover-method'] == method)]\
                       .groupby(['seed']))
            std = gb_seed['best-fitness'].min().std()
            std_values.append(std)
            x_labels.append(f"{init_method}, {method}")

    # Bar plot for each std between the final best values of each run
    plt.bar(range(len(x_labels)), std_values)
    plt.xticks(range(len(x_labels)), x_labels, rotation=45, ha='right')
    plt.ylabel('Standard deviation')
    plt.title(f'The standard deviation of the best-fitness achieved in {runs} runs', fontsize=FONT_SIZE)
    plt.suptitle(f'Population size: {population_size}, Points: {num_points}', fontsize=FONT_SIZE)

    plt.show()

def plot_avg_gens(df: pd.DataFrame, population_size, num_points, metric, oc_method = False):
    """
    Creates a graph for each combination of initialization and crossover method. Each graphs holds plots showing
    the chosen metric for fitness scores in each generation, averaged over different runs.

    df:
        Dataframe holding all the evaluation data
    population_size:
        int that represent the population size
    num_points:
        int that represent the amount of points
    metric:
        string that represents whether the plots should show the std or avg
    returns:
        None
    """

    if metric == 'avg':
        metric_string = "Average"
    elif metric == 'std':
        metric_string = "Standard deviation of"
    else:
        raise ValueError("metric should be either 'avg' or 'std'. ")

    print("Plots showing the best-fitness per generation, averaged over several runs")
    runs = df['run'].max() + 1
    crossover_methods = df['crossover-method'].unique()
    init_methods = df['initialization'].unique()

    if oc_method:
        oc_methods = df['optimize_color_method'].unique()
        for init_method in init_methods:
            n_colours = len(crossover_methods) * len(init_methods) * len(oc_methods)
            cmap = get_cmap(n_colours)
            colours = [cmap(i) for i in range(n_colours)]
            linestyles = get_linestyles(n_colours)

            plt.figure()
            gb_crossovers = df[df['initialization'] == init_method].groupby(['num-generations', 'crossover-method', 'optimize_color_method'])[
                f'fitness-{metric}'].mean().astype(int)
            for method in oc_methods:
                for c_method in crossover_methods:
                    data = gb_crossovers.loc[:, c_method, method]
                    plt.plot(data.index.get_level_values('num-generations'), data.values, label=f'{c_method}, {method}', linewidth=WIDTH, color=colours.pop(0), linestyle=linestyles.pop(0))
        plt.legend(title='Combinations', fontsize=FONT_SIZE)

    else:
        for init_method in init_methods:
            plt.figure()
            gb_crossovers = df[df['initialization'] == init_method].groupby(['num-generations', 'crossover-method'])[f'fitness-{metric}'].mean().astype(int)
            for method in crossover_methods:
                data = gb_crossovers.loc[:, method]
                plt.plot(data.index.get_level_values('num-generations'), data.values, label=method, linewidth=WIDTH)
        plt.legend(title='Crossover methods', fontsize=FONT_SIZE)

    plt.xlabel('Number of generations')
    plt.ylabel(f"{metric_string.replace('of', '')} fitness")
    plt.title(f'{metric_string} fitness per generation, averaged over {runs} runs', fontsize=FONT_SIZE)
    plt.suptitle(f'Population size: {population_size}, Points: {num_points}, Initialization method: {init_method}', fontsize=FONT_SIZE)


    plt.show()

def plot_avg_time(df: pd.DataFrame, population_size, num_points, metric, oc_method = False):
    """
    Creates a graph for each combination of initialization and crossover method. Each graphs holds plots showing
    the chosen metric for fitness scores in each generation, averaged over different runs.

    df:
        Dataframe holding all the evaluation data
    population_size:
        int that represent the population size
    num_points:
        int that represent the amount of points
    metric:
        string that represents whether the plots should show the std or avg
    returns:
        None
    """

    if metric == 'avg':
        metric_string = "Average"
    elif metric == 'std':
        metric_string = "Standard deviation of"
    else:
        raise ValueError("metric should be either 'avg' or 'std'. ")

    print("Plots showing the best-fitness per generation, averaged over several runs")
    runs = df['run'].max() + 1
    crossover_methods = df['crossover-method'].unique()
    init_methods = df['initialization'].unique()

    if oc_method:
        oc_methods = df['optimize_color_method'].unique()
        for init_method in init_methods:
            n_colours = len(crossover_methods) * len(init_methods) * len(oc_methods)
            cmap = get_cmap(n_colours)
            colours = [cmap(i) for i in range(n_colours)]
            linestyles = get_linestyles(n_colours)
            plt.figure()
            gb_crossovers = df[df['initialization'] == init_method].groupby(['time-elapsed', 'crossover-method', 'optimize_color_method'])[
                f'fitness-{metric}'].mean().astype(int)
            for method in oc_methods:
                for c_method in crossover_methods:
                    data = gb_crossovers.loc[:, c_method, method]
                    plt.plot(data.index.get_level_values('time-elapsed'), data.values, label=f'{c_method}, {method}', linewidth=WIDTH, color=colours.pop(0), linestyle=linestyles.pop(0))
        plt.legend(title='Combinations', fontsize=FONT_SIZE)

    else:
        for init_method in init_methods:
            plt.figure()
            gb_crossovers = df[df['initialization'] == init_method].groupby(['time-elapsed', 'crossover-method'])[f'fitness-{metric}'].mean().astype(int)
            for method in crossover_methods:
                data = gb_crossovers.loc[:, method]
                plt.plot(data.index.get_level_values('time-elapsed'), data.values, label=method)
        plt.legend(title='Crossover methods', fontsize=FONT_SIZE)

    plt.xlabel('Time')
    plt.ylabel(f"{metric_string.replace('of', '')} fitness")
    plt.title(f'{metric_string} generational fitness over time, averaged over {runs} runs', fontsize=FONT_SIZE)
    plt.suptitle(f'Population size: {population_size}, Points: {num_points}, Initialization method: {init_method}', fontsize=FONT_SIZE)


    plt.show()    

def plot_best_fitness_time(df: pd.DataFrame, population_size, num_points, oc_method = False):
    """
    Creates a graph for each combination of initialization and crossover method. Each graphs holds plots showing
    the best fitness over time, averaged over different runs.

    df:
        Dataframe holding all the evaluation data
    population_size:
        int that represent the population size
    num_points:
        int that represent the amount of points
    oc_method:
        boolean to set whether different optimize_color_methods are used
    returns:
        None
    """

    print("Plots showing the best-fitness over time, averaged over several runs")
    runs = df['run'].max() + 1
    crossover_methods = df['crossover-method'].unique()
    init_methods = df['initialization'].unique()

    if oc_method:
        oc_methods = df['optimize_color_method'].unique()
        for init_method in init_methods:
            n_colours = len(crossover_methods) * len(init_methods) * len(oc_methods)
            cmap = get_cmap(n_colours)
            colours = [cmap(i) for i in range(n_colours)]
            linestyles = get_linestyles(n_colours)
            plt.figure()
            gb_crossovers = df[df['initialization'] == init_method].groupby(['time-elapsed', 'crossover-method', 'optimize_color_method'])['best-fitness'].mean().astype(int)
            for method in oc_methods:
                for c_method in crossover_methods:
                    data = gb_crossovers.loc[:, c_method, method]
                    plt.plot(data.index.get_level_values('time-elapsed'), data.values, label=f'{c_method}, {method}', linewidth=WIDTH, color=colours.pop(0), linestyle=linestyles.pop(0))
        plt.legend(title='Combinations', fontsize=FONT_SIZE)

    else:
        pass

    plt.xlabel('Time in seconds')
    plt.ylabel(f"Best fitness")
    plt.title(f'Best fitness over time, averaged over {runs} runs', fontsize=FONT_SIZE)
    plt.suptitle(f'Population size: {population_size}, Points: {num_points}, Initialization method: {init_method}', fontsize=FONT_SIZE)
    plt.legend(title='Combinations', fontsize=FONT_SIZE)


    plt.show()


def plot_avg_gens_mult(df: pd.DataFrame, population_size, num_points, oc_method=False):
    """
    Creates a graph for each combination of initialization and crossover method. Each graphs holds plots showing
    the chosen metric for fitness scores in each generation, averaged over different runs.

    df:
        Dataframe holding all the evaluation data
    population_size:
        int that represent the population size
    num_points:
        int that represent the amount of points
    returns:
        None
    """


    print("Plots showing the best-fitness per generation, averaged over several runs")
    runs = df['run'].max() + 1
    crossover_methods = df['crossover-method'].unique()
    init_methods = df['initialization'].unique()
    images = df['image_name'].unique()

    for image in images:

        if oc_method:
            oc_methods = df['optimize_color_method'].unique()
            for init_method in init_methods:
                plt.figure()
                gb_crossovers = df[df['image_name'] == image].groupby(
                    ['num-generations', 'crossover-method', 'optimize_color_method'])[
                    f'fitness-avg'].mean().astype(int)
                for method in oc_methods:
                    for c_method in crossover_methods:
                        data = gb_crossovers.loc[:, c_method, method]
                        plt.plot(data.index.get_level_values('num-generations'), data.values, label=f'{c_method}, {method}', linewidth=WIDTH)
            plt.legend(title='Combinations', fontsize=FONT_SIZE)

            plt.xlabel('Number of generations')
            plt.ylabel(f"Average fitness")
            plt.title(f'Average fitness per generation on image {image}, averaged over {runs} runs', fontsize=FONT_SIZE)
            plt.suptitle(f'Population size: {population_size}, Points: {num_points}, Initialization method: {init_method}', fontsize=FONT_SIZE)

            plt.show()

        else:
            pass

def plot_best_fitness_time_mult(df: pd.DataFrame, population_size, num_points, oc_method = False):
    """
    Creates a graph for each combination of initialization and crossover method. Each graphs holds plots showing
    the best fitness over time, averaged over different runs.

    df:
        Dataframe holding all the evaluation data
    population_size:
        int that represent the population size
    num_points:
        int that represent the amount of points
    oc_method:
        boolean to set whether different optimize_color_methods are used
    returns:
        None
    """

    print("Plots showing the best-fitness over time, averaged over several runs")
    runs = df['run'].max() + 1
    crossover_methods = df['crossover-method'].unique()
    init_methods = df['initialization'].unique()

    if oc_method:
        oc_methods = df['optimize_color_method'].unique()
        for init_method in init_methods:
            plt.figure()
            gb_crossovers = df[df['initialization'] == init_method].groupby(['time-elapsed', 'crossover-method', 'optimize_color_method'])['best-fitness'].mean().astype(int)
            for method in oc_methods:
                for c_method in crossover_methods:
                    data = gb_crossovers.loc[:, c_method, method]
                    plt.plot(data.index.get_level_values('time-elapsed'), data.values, label=f'{c_method}, {method}', linewidth=WIDTH)
        plt.legend(title='Combinations', fontsize=FONT_SIZE)

    else:
        pass

    plt.xlabel('Time in seconds')
    plt.ylabel(f"Average best fitness")
    plt.title(f'Average best fitness over time, averaged over {runs} runs', fontsize=FONT_SIZE)
    plt.suptitle(f'Population size: {population_size}, Points: {num_points}, Initialization method: {init_method}', fontsize=FONT_SIZE)
    plt.legend(title='Combinations', fontsize=FONT_SIZE)


    plt.show()