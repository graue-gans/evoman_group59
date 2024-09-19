###############################################################################
# EvoMan FrameWork - V1.0 2016  			                                  #
# DEMO : Neuroevolution - Genetic Algorithm  neural network.                  #
# Author: Karine Miras        			                                      #
# karine.smiras@gmail.com     				                                  #
###############################################################################

# imports framework
import sys

from evoman.environment import Environment
from demo_controller import player_controller

# imports other libs
import numpy as np
import os

#import deap and random
import random
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'optimization_test'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10

    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[2],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=True)

    env.state_to_log()



    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors()+1)*n_hidden_neurons + (n_hidden_neurons+1)*5

    # Parameters for evolutionary algorithm

    dom_l = -1
    dom_u = 1
    npop = 20
    cx_prob = 0.8  # Probability of mating (crossover)
    mut_prob = 0.02  # Probability of mutating
    n_generations = 10  # Number of generations
    tournsize = 5


    pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))

    # Create Fitness and Individual classes using DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize fitness
    creator.create("Individual", list, fitness=creator.FitnessMax)

    # Create DEAP toolbox
    toolbox = base.Toolbox()

    # Attribute generator: Creates random values for each individual in the population
    toolbox.register("attr_float", random.uniform, -1, 1)

    # Structure initializers: Define the individual and population structure
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # Fitness function: Evaluate the individual in the environment
    def evaluate_individual(individual):
        fitness = env.play(pcont=np.array(individual))[0]  # Fitness is the first return value from play
        return fitness,

    # Register the fitness evaluation function
    toolbox.register("evaluate", evaluate_individual)

    # Register the crossover and mutation functions
    toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=1, indpb=0.05)  # Gaussian mutation
    toolbox.register("select", tools.selTournament, tournsize=tournsize)  # Tournament selec

    random.seed(43) #43 shows a nice graph

    # Add statistics
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    # Logbook to track the stats
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    # Create initial population
    population = toolbox.population(npop)

    # Evaluate the entire population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # Record statistics for the initial population
    record = stats.compile(population)
    logbook.record(gen=0, nevals=len(population), **record)
    print(logbook.stream)


    # Evolutionary algorithm
    for gen in range(n_generations):
#        print(f"Generation {gen}")

        # Select individuals for the next generation
        offspring = toolbox.select(population, len(population)-1)
        offspring = list(map(toolbox.clone, offspring))


        # ** Elitism **: Keep the best individual from the current population
        best_individual = tools.selBest(population, k=1)[0]  # Select the best individual
        offspring = list(map(toolbox.clone, offspring))

        # Apply crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # Evaluate the new individuals (only those with no fitness values)
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        #elitesism check spelling
        offspring.append(best_individual)  # Add the best individual to the offspring

        # Replace the old population with the new one
        population[:] = offspring

        # Gather and print the best fitness in the population
        fits = [ind.fitness.values[0] for ind in population]
        best_fitness = max(fits)
#        print(f"Best fitness: {best_fitness}")

        # Compile the statistics for this generation and record them
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        current_best_individual = population[fits.index(best_fitness)]

    # Plot the fitness values over generations
    gen = logbook.select("gen")
    avg = logbook.select("avg")
    std = logbook.select("std")
    min_fitness = logbook.select("min")
    max_fitness = logbook.select("max")

    plt.figure(figsize=(10, 6))
    plt.plot(gen, avg, label="Average Fitness")
    plt.fill_between(gen, np.array(avg) - np.array(std), np.array(avg) + np.array(std), alpha=0.2, label="Standard Deviation")
    plt.plot(gen, min_fitness, label="Minimum Fitness", color="red")
    plt.plot(gen, max_fitness, label="Maximum Fitness", color="green")
    plt.xlabel("Generation")
    plt.ylabel("Fitness")
    plt.title("Fitness Evolution Over Generations")
    plt.legend(loc="best")
    plt.grid(True)
    plt.show()




    # start writing your own code from here


if __name__ == '__main__':
    main()