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
import random
import matplotlib.pyplot as plt

from deap import base, creator, tools, algorithms

# runs simulation
def simulation(env,x):
    f,p,e,t = env.play(pcont=x)
    return f
# evaluation for the induvidual because of deap
def simulation_indu(env,x):
    f,p,e,t = env.play(pcont=x)
    return f,p,e,t

# evaluation
def evaluate(env, x):
    return np.array(list(map(lambda y: simulation(env,y), x)))


def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'blend_7'
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Parameters for EA
    n_hidden_neurons = 10
    dom_l = -1
    dom_u = 1
    npop = 100 #50
    cx_prob = 0.7  # Probability of mating (crossover)
    mut_prob = 0.05  # Probability of mutating
    n_generations = 50  # Number of generations 50
    tournsize = 3
    sigma_gausian = 0.3

    # program options
    run_times = 10
    program_name = "run_experiment"

 #   random.seed(43) #43 shows a nice graph


    # initializes simulation in individual evolution mode, for single static enemy.
    env = Environment(experiment_name=experiment_name,
                    enemies=[7],
                    playermode="ai",
                    player_controller=player_controller(n_hidden_neurons), # you  can insert your own controller here
                    enemymode="static",
                    level=2,
                    speed="fastest",
                    visuals=False,
                    randomini="yes")


    env.state_to_log()

    # running the ea one time with graphs
    if program_name == "test_single":
        logbook = run_ea(env, n_hidden_neurons, dom_l, dom_u, npop, cx_prob, mut_prob, n_generations, tournsize, sigma_gausian)

        # plot the fitness  over generations
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

    # running the ea run_times and mapping the avg en std in a graph
    elif program_name == "run_experiment":

        # make empty arrays
        all_avg_fitness = np.zeros((run_times, n_generations))
        all_std_fitness = np.zeros((run_times, n_generations))
        all_max_fitness = np.zeros((run_times, n_generations))
        best_fitness_all = 0


        # runs ea run_times times
        for run in range(run_times):
            print(f"Running simulation {run + 1}/{run_times}")
            logbook, current_best_individual, best_fitness= run_ea(env, n_hidden_neurons, dom_l, dom_u, npop, cx_prob, mut_prob, n_generations, tournsize, sigma_gausian)
            if best_fitness > best_fitness_all:
                best_fitness_all = best_fitness
                best_individual = current_best_individual

            all_avg_fitness[run, :] = logbook.select("avg")
            all_std_fitness[run, :] = logbook.select("std")
            all_max_fitness[run, :] = logbook.select("max")

        # saves file with the best solution
        np.savetxt(experiment_name + '/best.txt', best_individual)
        print(f"The best solutions fitness is {best_fitness_all} and has been saved")

        # calculates the total avg and std
        std_mac_fitness = np.std(all_max_fitness, axis=0)
        max_fitness_across_runs = np.mean(all_max_fitness, axis=0)
        avg_fitness_across_runs = np.mean(all_avg_fitness, axis=0)
        std_fitness_across_runs = np.mean(all_std_fitness, axis=0)

        #logs files
        np.savez(experiment_name + '/logs1.npz', std_fitness=std_mac_fitness, max_fitness=max_fitness_across_runs, avg_fitness=avg_fitness_across_runs, std_avg_fitness=std_fitness_across_runs)

        # plot the avg and std
        generations = list(range(n_generations))
        plt.figure(figsize=(10, 6))
        plt.plot(generations, avg_fitness_across_runs, label="Average Fitness")
        plt.fill_between(generations, avg_fitness_across_runs - std_fitness_across_runs, avg_fitness_across_runs + std_fitness_across_runs, alpha=0.2, label="Standard Deviation")
        plt.plot(generations, max_fitness_across_runs, label="Max Fitness")
        plt.fill_between(generations, max_fitness_across_runs - std_mac_fitness, max_fitness_across_runs + std_mac_fitness, alpha=0.2, label="Standard Deviation Fitness")
        plt.xlabel("Generation")
        plt.ylabel("Fitness")
        plt.title(f"Fitness over {run_times} runs")
        plt.legend(loc="best")
        plt.grid(True)
        plt.show()


    # run the best found solution in the previous experiment
    # IMPORTANT to run graphics set HEADLESS to FALSE
    elif program_name == "run_solution":
        bsol = np.loadtxt(experiment_name + '/best.txt')
        print('\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed', 'normal')
        fitness, playerlife, enemylife, gameruntime = simulation_indu(env, bsol)

        print(f"fitness: {fitness} \nplayerlife: {playerlife}\nenemylife: {enemylife}\ngameruntime: {gameruntime}")

        sys.exit(0)

    elif program_name == "run_solution_5_times":
        bsol = np.loadtxt(experiment_name + '/best.txt')
        print('\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('randomini', 'yes')

        for i in range(5):
            fitness, playerlife, enemylife, gameruntime = simulation_indu(env, bsol)
            print(f"individual gain is:{playerlife-enemylife}")

        sys.exit(0)

    # run the best found solution in the previous experiment, for 3 games
    # IMPORTANT to run graphics set HEADLESS to FALSE
    elif program_name == "run_solution_3":
        bsol = np.loadtxt(experiment_name + '/best.txt')
        print('\n RUNNING SAVED BEST SOLUTION 3 EXPERIMENTS\n')
        env.update_parameter('speed', 'normal')
        levellist = [1,2,3,4,5,6,7,8]
        for i in levellist:
            env.update_parameter('enemies', [i])

            fitness, playerlife, enemylife, gameruntime = simulation_indu(env, bsol)

            print(f"level: {i} \nfitness: {fitness} \nplayerlife: {playerlife}\nenemylife: {enemylife}\ngameruntime: {gameruntime}")

        sys.exit(0)



    else:
        print(f"Program '{program_name}' not found")


def run_ea(env, n_hidden_neurons, dom_l, dom_u, npop, cx_prob, mut_prob, n_generations, tournsize, sigma_gausian):

    # deap only works with evaluation of an individual
    # cant putt this between other functions because local variable env cant be passed with Deap
    def evaluate_individual(individual):
        fitness = env.play(pcont=np.array(individual))[0]  # Fitness is the first return value of play
        return fitness,

    # number of weights for multilayer with 10 hidden neurons
    n_vars = (env.get_num_sensors() + 1) * n_hidden_neurons + (n_hidden_neurons + 1) * 5

    #   pop = np.random.uniform(dom_l, dom_u, (npop, n_vars))

    # Create Fitness and Individual classes using DEAP
    creator.create("FitnessMax", base.Fitness, weights=(1.0,))  # Maximize fitness
    creator.create("Individual", list, fitness=creator.FitnessMax)

    toolbox = base.Toolbox()

    # make population
    toolbox.register("attr_float", random.uniform, dom_l, dom_u)
    toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_float, n=n_vars)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    # register evaluation function
    toolbox.register("evaluate", evaluate_individual)

    # register the crossover and mutation functions
   # toolbox.register("mate", tools.cxTwoPoint)  # Two-point crossover
    toolbox.register("mate", tools.cxBlend, alpha=0.8)  # Two-point crossover

    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=sigma_gausian, indpb=mut_prob)  # Gaussian mutation
  #  toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  # Gaussian mutation

    toolbox.register("select", tools.selTournament, tournsize=tournsize)  # Tournament selec

    # add logging
    stats = tools.Statistics(key=lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + stats.fields

    # create initial population
    population = toolbox.population(npop)

    # evaluate the initial population
    fitnesses = list(map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # log
    record = stats.compile(population)
    logbook.record(gen=0, nevals=len(population), **record)
    print(logbook.stream)

    # evolutionary algorithm
    for gen in range(1, n_generations):
#        print(f"Generation {gen}")

        # Select offspring
        offspring = toolbox.select(population, len(population)-1)
        offspring = list(map(toolbox.clone, offspring))


        # elitism
        best_individual = tools.selBest(population, k=1)[0]  # Select the best individual
        offspring = list(map(toolbox.clone, offspring))

        # crossover and mutation
        for child1, child2 in zip(offspring[::2], offspring[1::2]):
            if random.random() < cx_prob:
                toolbox.mate(child1, child2)
                del child1.fitness.values
                del child2.fitness.values

        for mutant in offspring:
            if random.random() < mut_prob:
                toolbox.mutate(mutant)
                del mutant.fitness.values

        # evaluate if no fitness
        invalid_ind = [ind for ind   in offspring if not ind.fitness.valid]
        fitnesses = map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # elitism
        offspring.append(best_individual)  # Add the best individual to the offspring

        # replace the old population with new
        population[:] = offspring

        # gather and print the best fitness in the population
        fits = [ind.fitness.values[0] for ind in population]
        best_fitness = max(fits)
#        print(f"Best fitness: {best_fitness}")

        # log
        record = stats.compile(population)
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        print(logbook.stream)

        current_best_individual = population[fits.index(best_fitness)]

    return logbook, current_best_individual, best_fitness




if __name__ == '__main__':
    main()