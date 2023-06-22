from deap import algorithms, base, creator, tools
import random
from functools import partial
import numpy as np
import scipy
import pandas as pd

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
gamma = scipy.stats.gamma(0.4206577838478015, scale=3.4008766040577783*15,loc=0.011999243192420739)
ind_length = expression_data.full.shape[0]

population_size = 1800
num_generations = 400
num_parents = round(population_size * 0.15)
init_num_removed = 50 
creator.create("Fitness", base.Fitness, weights=(-1.0, 1.0))
creator.create("Individual", list, fitness=creator.Fitness)

def create_individual():
    a =  ind_length//(init_num_removed*3)
    b = ind_length//init_num_removed
    individual = random.choices([0,1], weights=(1, random.randint(a,b)), k=ind_length) 
    return creator.Individual(individual)

def get_p_value(solution):
    sol = np.array(solution)
    up = sol.dot(expression_data.age_weighted)
    down = sol.dot(expression_data.expressions_n)
    avgs = np.divide(up,down)
    varr = np.var(avgs)
    return 1 - np.array(gamma.cdf(varr))
    
def evaluate_individual(individual):
    num_not_removed = np.sum(individual)
    len_removed = ind_length - num_not_removed
    p_value = get_p_value(individual)

    # Return the fitness values as a tuple
    return len_removed, p_value

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

toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
#toolbox.register("population", tools.initRepeat, list, toolbox.individual)
population = [toolbox.individual() for _ in range(population_size)]
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxUniform,indpb=0.05)
toolbox.register("mutate", tools.mutFlipBit, indpb=0.003)
toolbox.register("select", tools.selSPEA2)

stats = tools.Statistics()
stats.register("Num removed", lambda x: np.max([ind.fitness.values[0] for ind in x]))
stats.register("P-value", lambda x: np.max([ind.fitness.values[1] for ind in x]))


population, logbook = algorithms.eaMuPlusLambda(population,toolbox, mu=num_parents, lambda_ = population_size,cxpb=0.25, mutpb=0.5, ngen=num_generations,stats=stats, verbose=True)



pareto_front = tools.sortNondominated(population, k=population_size)
for individual in pareto_front:
    print(individual, individual.fitness.values)