from deap import algorithms, base, creator, tools
import random
from functools import partial
import numpy as np
import scipy
import pandas as pd
import array
import multiprocess
from timeit import default_timer as timer
import GA_utils
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
        #exps = np.log(exps).apply(lambda x: (x - x.mean()) / x.std(), axis=1)
        self.age_weighted = exps.mul(expression_data["Phylostratum"], axis=0).to_numpy()
        self.expressions_n = exps.to_numpy()
        self.expressions = exps


arr = pd.read_csv("../data/phylo.csv",
                 delimiter=",")
expression_data = Expression_data(arr)
#mean,prec = GA_utils.comp_mean_prec(expression_data,50000)
variances = np.loadtxt("min_max_raw.txt")
ind_length = expression_data.full.shape[0]

population_size = 1000
parents_ratio = 0.25
num_generations = 4000
init_num_removed = 100



def get_distance(solution):
    sol = np.array(solution)
    up = sol.dot(expression_data.age_weighted)
    down = sol.dot(expression_data.expressions_n)
    avgs = np.divide(up,down)
    #return scipy.spatial.distance.mahalanobis(avgs,mean,prec)
    return np.max(avgs) - np.min(avgs)


max_value = get_distance(np.ones(ind_length))


def create_individual():
    a =  ind_length//(init_num_removed*3)
    b = ind_length//init_num_removed
    individual = array.array("b",random.choices([0,1], weights=(1, random.randint(a,b)), k=ind_length))
    return creator.Individual(individual)

def get_fit(res):
    p = np.count_nonzero(variances < res)/len(variances)
    r = (res) / (max_value)
    r = r + p
    return r if p > 0.1 else 0
    
def evaluate_individual(individual):
    individual = np.array(individual)
    num_not_removed = np.sum(individual)
    len_removed = ind_length - num_not_removed
    distance = get_distance(individual)
    fit = get_fit(distance)
    # Return the fitness values as a tuple
    return len_removed, fit

def mutate(individual,indpb):
    if random.random() < indpb:
        randomlist = random.sample(range(len(individual)), k=random.getrandbits(5))
        for ind in randomlist:
            individual[ind] = (individual[ind] + 1) % 2
    return individual,

def scattered_crossover(ind1,ind2):
    gene_sources = np.random.randint(0, 5, size=ind_length)
    i1 = ind1.copy()
    i2 = ind2.copy()
    for i,x in enumerate(gene_sources):
        if x > 0:
            ind1[i] = i1[i]
            ind2[i] = i2[i]
        else:
            ind1[i] = i2[i]
            ind2[i] = i1[i]
    return ind1,ind2


mut  = 0.001
cross = 0.02

creator.create("Fitness", base.Fitness, weights=(-1.0, -10.0))
creator.create("Individual", array.array,typecode='b', fitness=creator.Fitness)
toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxUniform,indpb=cross)
toolbox.register("mutate", tools.mutFlipBit, indpb=mut)
toolbox.register("select", tools.selSPEA2)


if __name__ == "__main__":
    #pool = multiprocess.Pool()
    #toolbox.register("map", pool.map)
    #toolbox.register("population", tools.initRepeat, list, toolbox.individual)

    stats = tools.Statistics()
    stats.register("Num removed", lambda x: x[np.argmin([ind.fitness.values[1] for ind in x])].fitness.values[0])
    stats.register("Min max distance", lambda x: np.min([ind.fitness.values[1] for ind in x]))

    start = timer()
    population, logbook = GA_utils.eaMuPlusLambda_stop(toolbox.population(n=population_size),toolbox, mu=round(population_size * parents_ratio), lambda_ = population_size,cxpb=0.45, mutpb=0.45, ngen=num_generations,stats=stats, verbose=True)
    toolbox.register("select", tools.selNSGA2)
    #population, logbook = GA_utils.eaMuPlusLambda_stop(population,toolbox, mu=round(population_size2 * parents_ratio), lambda_ = population_size2,cxpb=0.3, mutpb=0.5, ngen=num_generations2,stats=stats, verbose=True)

    end = timer()
    print(end - start)

    pareto_front = tools.sortNondominated(population, k=population_size,first_front_only=True)
    par = np.array([list(x) for x in pareto_front[0]])
    np.savetxt("../results/best_solutions_NSGA2.csv", par, delimiter="\t")
    print(mut)
    print(cross)