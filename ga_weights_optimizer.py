import pickle
import random
import time
from pprint import pprint

import numpy as np
from collections import Counter
from deap import creator, base, tools, algorithms
import config



class WeightsOptimizer:

    def __init__(self, n, model):
        self.n = n
        self.model = model
        self.ngen = config.ga["number_of_generations"]
        self.population_size = config.ga["population_size"]
        self.cxpb = 0.8
        self.mutpb = 0.2
        self.step = config.ga["step_size"]


    def checkBounds(self):
        """
        Decorator for converting weights to the reqired format(steps of given step size, sum = 1)
        :return: Decorated weights
        """

        def decorator(func):
            def wrapper(*args, **kargs):
                offspring = func(*args, **kargs)
                for child in offspring:
                    for i in range(self.n):
                        child[i] = round(child[i], 2)
                        if child[i] % self.step != 0:
                            if child[i] < self.step:
                                child[i] = self.step
                            else:
                                child[i] = round(round(child[i] / self.step) * self.step, 2)
                    diff = round(1 - sum(child), 2)
                    d = abs(round(diff / self.step, 2))
                    if diff >= 0:
                        indexes = np.random.choice(self.n, int(d))
                        for ix in indexes:
                            child[ix] = child[ix] + self.step
                    else:
                        while d > 0:
                            ix = random.randint(0, self.n - 1)
                            if child[ix] < self.step:
                                continue
                            child[ix] -= self.step
                            d -= 1
                    for i in range(self.n):
                        child[i] = round(child[i], 2)
                return offspring
            return wrapper
        return decorator


    def generate_weights(self):
        """
        Generates n weights which are in steps of the paramerter step and sum = 1
        :param step: step size for weights, all weights will be a multiple of the step_size
        :param n: number of weights to be generated
        :return: n weights
        """
        weights = np.round(np.random.dirichlet(np.ones(self.n), size=1)[0], 2)
        weights = np.where(weights % self.step != 0,
                           np.where(weights < self.step, self.step, np.round(np.round(weights / self.step) * self.step, 2)), weights)
        diff = round(1 - sum(weights), 2)
        d = abs(round(diff / self.step, 2))
        if diff >= 0:
            indexes = np.random.choice(self.n, int(d))
            weights[indexes] += self.step
        else:
            while d > 0:
                ix = random.randint(0, self.n - 1)
                if weights[ix] < self.step:
                    continue
                weights[ix] -= self.step
                d -= 1
        weights = np.round(weights, 2)
        return list(weights)

    def ga(self, checkpoint=False):
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Individual", list, fitness=creator.FitnessMax)
        toolbox = base.Toolbox()
        toolbox.register("init_attr", self.generate_weights)
        toolbox.register("x", tools.initIterate, creator.Individual, toolbox.init_attr)
        toolbox.register("population", tools.initRepeat, list, toolbox.x)
        toolbox.register("evaluate", self.model.objective_function)  # opti_obj=opti_obj)
        toolbox.register("mate", tools.cxTwoPoint)
        toolbox.register("mutate", tools.mutShuffleIndexes, indpb=0.05)
        toolbox.register("select", tools.selTournament, tournsize=2)

        toolbox.decorate("mate", self.checkBounds())
        toolbox.decorate("mutate", self.checkBounds())

        generation = 0

        if checkpoint:
            with open(config.checkpoint, "rb") as cp_file:
                cp = pickle.load(cp_file)
            population = cp["population"]
            generation = cp["generation"] + 1
            random.setstate(cp["rndstate"])
        else:
            population = toolbox.population(n=self.population_size)
            print("\nInitial Population: ")
            pprint(population)
            fitnesses = list(map(toolbox.evaluate, population))
            for ind, fit in zip(population, fitnesses):
                ind.fitness.values = (fit[0],)


        for gen in range(generation, self.ngen):
            start_time = time.time()
            print('\nGA Iteration no: ', gen)
            print("\nPopulation:")
            pprint(population)
            c = Counter(map(tuple, population))
            if list(c.items())[0][1] == len(population):
                print("Optimization converged. Exiting")
                break
            offspring = toolbox.select(population, len(population))
            print("\nAfter Selection:")
            pprint(offspring)
            print("\nAfter Applying crossover and mutation:")
            offspring = algorithms.varAnd(offspring, toolbox, cxpb=self.cxpb, mutpb=self.mutpb)
            pprint(offspring)

            print("Calculating fitness")
            fits = list(map(toolbox.evaluate, offspring))
            for fit, ind in zip(fits, offspring):
                ind.fitness.values = (fit[0],)

            population[:] = offspring

            if gen % config.ga["frequency_checkpoints"] == 0:
                cp = dict(population=population, generation=gen, rndstate=random.getstate())
                with open(config.checkpoint, "wb") as cp_file:
                    pickle.dump(cp, cp_file)
                    cp_file.close()

            fits = [ind.fitness.values[0] for ind in population]

            length = len(population)
            mean = sum(fits) / length
            sum2 = sum(x * x for x in fits)
            std = abs(sum2 / length - mean ** 2) ** 0.5

            print("\n")
            print("  Min %s" % min(fits))
            print("  Max %s" % max(fits))
            print("  Avg %s" % mean)
            print("  Std %s" % std)
            print("\nCompleted generation in time ", time.time() - start_time)

        top = tools.selBest(population, k=1)
        weights = top[0]
        fitness = weights.fitness.values[0]
        return weights, fitness


