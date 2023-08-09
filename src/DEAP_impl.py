from deap import algorithms, base, creator, tools
import random
from functools import partial
import numpy as np
import scipy
import pandas as pd
import array
from timeit import default_timer as timer
import GA_utils
import itertools
from concurrent.futures import ThreadPoolExecutor
import os
import datetime
import pickle
import json

class Expression_data:

    def quantilerank(xs):
        ranks = scipy.stats.rankdata(xs, method='average')
        quantile_ranks = [scipy.stats.percentileofscore(ranks, rank, kind='weak') for rank in ranks]
        return np.array(quantile_ranks)

    def __init__(self,expression_data) -> None:
        expression_data["Phylostratum"] = Expression_data.quantilerank(expression_data["Phylostratum"])
        self.full = expression_data
        exps = expression_data.iloc[:, 3:]
        #exps = exps.apply(lambda row: Expression_data.quantilerank(row))
        exps = np.log(exps).apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        self.age_weighted = exps.mul(expression_data["Phylostratum"], axis=0).to_numpy()
        self.expressions_n = exps.to_numpy()
        self.expressions = exps


arr = pd.read_csv("../data/phylo.csv",
                 delimiter=",")
expression_data = Expression_data(arr)
#mean,prec = GA_utils.comp_mean_prec(expression_data,50000)
variances = np.loadtxt("min_max.txt")
ind_length = expression_data.full.shape[0]

population_size = 2300
parents_ratio = 0.25
num_generations = 5
init_num_removed = 100



def get_distance(solution):
    sol = np.array(solution)
    up = sol.dot(expression_data.age_weighted)
    down = sol.dot(expression_data.expressions_n)
    avgs = np.divide(up,down)
    #return scipy.spatial.distance.mahalanobis(avgs,mean,prec)
    return max(avgs) - min(avgs)


max_value = get_distance(np.ones(ind_length))
min_value = 0


def create_individual():
    a =  ind_length//(init_num_removed*3)
    b = ind_length//init_num_removed
    individual = array.array("b",random.choices([0,1], weights=(1, random.randint(a,b)), k=ind_length))
    return creator.Individual(individual)

def get_fit(res):
    r = (res - min_value) / (max_value - min_value)
    p = np.count_nonzero(variances < res)/len(variances)
    r = p + r
    return r if p > 0.5 else 0
    
def evaluate_individual(individual):
    num_not_removed = np.sum(individual)
    len_removed = ind_length - num_not_removed
    distance = get_distance(individual)
    fit = get_fit(distance)
    # Return the fitness values as a tuple
    return len_removed, fit


def mutFlipBit(individual, indpb,mut_zero):
    for i in range(len(individual)):
        if random.random() < indpb:
            individual[i] = type(individual[i])(not individual[i])
    for i in np.where(np.array(individual) == 0)[0]:
        if random.random() < mut_zero:
            individual[i] = type(individual[i])(not individual[i])

    return individual,
    
def mutate(individual, indpb):
    randomlist = random.sample(range(ind_length), k=random.getrandbits(indpb))
    for ind in randomlist:
        individual[ind] = (individual[ind] + 1) % 2
    return individual,


def scattered_crossover(ind1, ind2, indpb:int):
    gene_sources = np.random.randint(0, indpb, size=ind_length)
    for i in range(ind_length):
        if gene_sources[i] == 0:
            ind1[i], ind2[i] = ind2[i], ind1[i]
    return ind1, ind2

creator.create("Fitness", base.Fitness, weights=(-1.0, -10.0))
creator.create("Individual", array.array,typecode='b', fitness=creator.Fitness)

if __name__ == "__main__": 
  
    
    #mutFlipBitp =  partial(mutFlipBit, mut_zero=mut_zero)
    

    toolbox = base.Toolbox()
    toolbox.register("individual", create_individual)
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("evaluate", evaluate_individual)
    toolbox.register("mate", scattered_crossover,indpb=5)
    toolbox.register("mutate", mutate, indpb=5)
    toolbox.register("select", tools.selNSGA2)


    stats = tools.Statistics()
    stats.register("Num removed", lambda x: x[np.argmin([ind.fitness.values[1] for ind in x])].fitness.values[0])
    stats.register("Min max distance", lambda x: np.min([ind.fitness.values[1] for ind in x]))
    population, logbook = GA_utils.eaMuPlusLambda_stop(toolbox.population(n=population_size),toolbox, mu=round(population_size * parents_ratio), lambda_ = population_size,cxpb=0.4, mutpb=0.4, ngen=num_generations,stats=stats, verbose=True)
    pareto_front = tools.sortNondominated(population, k=population_size,first_front_only=True)
    par = np.array([list(x) for x in pareto_front[0]])

    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    results_folder = os.path.join("results", timestamp)
    os.makedirs(results_folder)
    
    params = {"Crossover probability:" : 5,"Mutation probability:" : 5}
    np.savetxt(os.path.join(results_folder, "results.txt"), par, delimiter="\t")
    with open(os.path.join(results_folder,"logbook.pkl"), "wb") as logbook_file:
        pickle.dump(logbook, logbook_file)
    with open(os.path.join(results_folder, "logbook.json"), "w") as logbook_file:
        json.dump(logbook, logbook_file)
    with open(os.path.join(results_folder, "parameters.txt"), "w") as params_file:
        for key, value in params.items():
            params_file.write(f"{key}: {value}\n")
    
 
   

"""
def get_step(rangee,num):
    return (rangee[1]-rangee[0])/num


cross_range = [0.03, 0.7]  # From 0.1 to 1.0 (inclusive)
mut_range = [0.005, 0.5]
zero_range = [0.005,0.1]

# Generate all combinations of the parameters
parameter_combinations = itertools.product(np.arange(cross_range[0], cross_range[1] + get_step(cross_range,5), get_step(cross_range,5)),
                                           np.arange(mut_range[0], mut_range[1] + get_step(mut_range,5), get_step(mut_range,5)),
                                           np.arange(zero_range[0], zero_range[1] + get_step(zero_range,5), get_step(zero_range,5)))

# Create a ThreadPoolExecutor to run the grid search in parallel
with ThreadPoolExecutor() as executor:
    
    # Submit the function with different parameter combinations to the ThreadPoolExecutor
    futures = [executor.submit(run_GA, *params) for params in parameter_combinations]

    # Get the results as they become available
    results = [future.result() for future in futures]
"""