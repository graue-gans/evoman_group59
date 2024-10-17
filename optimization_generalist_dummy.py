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
import multiprocessing
import optuna



def create_environment():
    # this has to be a function because multicore processes cant share the enviroment so each evaluation needs their own.
    experiment_name = 'Test' # doesnt matter
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    n_hidden_neurons = 10
    return Environment(experiment_name=experiment_name,
                       logs="off",
                       enemies=[1,2,3,4,5,6,7,8],
                       multiplemode="yes",
                       playermode="ai",
                       player_controller=player_controller(n_hidden_neurons),  # Insert your own controller here
                       enemymode="static",
                       level=2,
                       speed="fastest",
                       visuals=False,
                       randomini="no")
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

def evaluate_individual(individual):
    # multiprocessing can share memory. So a new environment is created for every evaluation
    env = create_environment()

    fitness = env.play(pcont=np.array(individual))[0]  # Fitness is the first return value of play
    return fitness,


def main():
    # choose this for not using visuals and thus making experiments faster
    headless = True
    if headless:
        os.environ["SDL_VIDEODRIVER"] = "dummy"


    experiment_name = 'test_blend_all_mees2' #TODO Change name for your experiment
    if not os.path.exists(experiment_name):
        os.makedirs(experiment_name)

    # Parameters for EA
    n_hidden_neurons = 10
    dom_l = -1
    dom_u = 1
    npop = 200 #153
    cx_prob = 0.885  # Probability of mating (crossover)
    mut_prob = 0.0474  # Probability of mutating
    n_generations = 500  # Number of generations 50
    tournsize = 4
    sigma_gausian = 0.876
    alpha = 0.78

    # program options
    run_times = 5
    program_name = "run_experiment"
    insert_seed = False


 #   random.seed(43)


    env = create_environment()

    # enable multiprocessing
    pool = multiprocessing.Pool()


    env.state_to_log()

    # optimise hyperparameters
    if program_name == "optimize":

        # Define the objective function for Optuna
        def objective(trial):
            # limits hyperparameters
            npop = trial.suggest_int("npop", 50, 200)  # Population size
            cx_prob = trial.suggest_float("cx_prob", 0.5, 0.9)  # Crossover probability
            mut_prob = trial.suggest_float("mut_prob", 0.01, 0.1)  # Mutation probability
            tournsize = trial.suggest_int("tournsize", 2, 6)  # Tournament size
            sigma_gausian = trial.suggest_float("sigma_gausian", 0.1, 1.0)  # Sigma for Gaussian mutation
            alpha = trial.suggest_float("alpha", 0.1, 1.0)

            times = 5
            total_fitness = 0
            overall_best_fitness = 0
            best_solution = 0

            # Run the EA a few times and get the max for the stocastic nature of the EA. Take the maximum fitness and use that
            for i in range(times):
                logbook, current_best_solution, best_fitness = run_ea(
                    env, n_hidden_neurons, dom_l, dom_u, npop, cx_prob, mut_prob, n_generations, tournsize, sigma_gausian,alpha, pool, insert_seed
                )
                if best_fitness >= overall_best_fitness:
                    overall_best_fitness = best_fitness
                    best_solution = current_best_solution

            trial.set_user_attr("best_solution", best_solution)


            return overall_best_fitness  # Return the fitness to maximize


        # create an study and optimize
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=100)

        print("Best hyperparameters:", study.best_params)
        print("Best fitness achieved:", study.best_value)

        # save best hyperparameters
        with open(experiment_name + '/best_hyperparams.txt', 'w') as f:
            f.write(f"Best hyperparameters: {study.best_params}\n")
            f.write(f"Best fitness: {study.best_value}\n")

        best_trial = study.best_trial
        best_solution = best_trial.user_attrs["best_solution"]

        # saves file with the best solution
        np.savetxt(experiment_name + '/best.txt', best_solution)
        print(f"The best solutions fitness is {study.best_value} and has been saved")

        print(best_solution)

    # running the ea one time with graphs
    elif program_name == "run_one_time":
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

        # register the parallelized map function with the toolbox
        toolbox = run_ea(env, n_hidden_neurons, dom_l, dom_u, npop, cx_prob, mut_prob, n_generations, tournsize, sigma_gausian, alpha, pool, insert_seed)

        # runs ea run_times times
        for run in range(run_times):
            print(f"Running simulation {run + 1}/{run_times}")
            logbook, current_best_individual, best_fitness= run_ea(env, n_hidden_neurons, dom_l, dom_u, npop, cx_prob, mut_prob, n_generations, tournsize, sigma_gausian, alpha, pool, insert_seed)
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

        # logs files
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
    # IMPORTANT to run graphics set HEADLESS to FALSE and Visuals to TRUE
    elif program_name == "run_solution":
        bsol = np.loadtxt(experiment_name + '/best.txt')
        print('\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('speed', 'normal')
        fitness, playerlife, enemylife, gameruntime = simulation_indu(env, bsol)

        print(f"fitness: {fitness} \nplayerlife: {playerlife}\nenemylife: {enemylife}\ngameruntime: {gameruntime}")

        sys.exit(0)


    # shows individual gain after each trial 5 times
    elif program_name == "run_solution_5_times":
        bsol = np.loadtxt(experiment_name + '/best.txt')
        print('\n RUNNING SAVED BEST SOLUTION \n')
        env.update_parameter('randomini', 'yes')

        for i in range(5):
            fitness, playerlife, enemylife, gameruntime = simulation_indu(env, bsol)
            print(f"individual gain is:{playerlife-enemylife}")

        sys.exit(0)

    # run the best found solution in the previous experiment, and give the indipendend statistics for all enemies
    elif program_name == "run_solution_8_enemies":

        # don't need multiprocessing for this so close the pool
        pool.close()
        pool.join()

        bsol = np.loadtxt(experiment_name + '/best.txt')
        print('\n RUNNING SAVED BEST SOLUTION 8 ENEMIES\n')

        env.update_parameter('speed', 'fastest')
        env.update_parameter('multiplemode', 'no')
        levellist = [1,2,3,4,5,6,7,8]

        print(f"{'Level':<12} {'Fitness':<10} {'PlayerLife':<12} {'EnemyLife':<10} {'GameRuntime':<12}")
        print("=" * 58)

        for i in levellist:
            env.update_parameter('enemies', [i])
            fitness, playerlife, enemylife, gameruntime = simulation_indu(env, bsol)
            print(f"{i:<12} {fitness:<10.2f} {playerlife:<12.2f} {enemylife:<10.2f} {gameruntime:<12}")

        sys.exit(0)

    else:
        print(f"Program '{program_name}' not found")

    # close the pool at the end
    pool.close()
    pool.join()


def run_ea(env, n_hidden_neurons, dom_l, dom_u, npop, cx_prob, mut_prob, n_generations, tournsize, sigma_gausian, alpha, pool, insert_seed):

    # create enviroment
    env = create_environment()

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
    toolbox.register("mate", tools.cxBlend, alpha=alpha)

    toolbox.register("mutate", tools.mutGaussian, mu=0, sigma=sigma_gausian, indpb=mut_prob)  # Gaussian mutation
  #  toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)  # Gaussian mutation

    toolbox.register("select", tools.selTournament, tournsize=tournsize)  # tournament selection

    # Replace the default map with the multiprocessing version
    toolbox.register("map", pool.map)

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
    fitnesses = list(toolbox.map(toolbox.evaluate, population))
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    # log
    record = stats.compile(population)
    logbook.record(gen=0, nevals=len(population), **record)
    print(logbook.stream)

    # insert a seed to use previously found best solution
    if insert_seed == True:

        seed_solution = np.loadtxt('seed/best.txt')
        seed_individual = creator.Individual(seed_solution)
        seed_individual.fitness.values = toolbox.evaluate(seed_individual)
        population.append(seed_individual)


    # evolutionary algorithm
    for gen in range(1, n_generations):

        # select offspring
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
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
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