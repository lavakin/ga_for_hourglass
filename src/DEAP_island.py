from deap import algorithms, base, creator, tools
import random
import numpy as np
import scipy
import pandas as pd
import array
#import multiprocess
from timeit import default_timer as timer
import GA_utils
import argparse
import os

parser = argparse.ArgumentParser()
    
# Add arguments for input and output files
parser.add_argument("input", type=str, help="Input file path")
parser.add_argument("output", type=str, help="Output file path")

parser.add_argument("--save_plot", action="store_true", help="Save pareto plot")
#parser.add_argument("--save_pareto", action="store_true", help="Save Pareto front")
args = parser.parse_args()

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


arr = pd.read_csv(args.input,
                 delimiter=",")
expression_data = Expression_data(arr)
#permuts = GA_utils.comp_min_max(expression_data,1000000)
permuts = np.loadtxt("min_max_raw.txt")
ind_length = expression_data.full.shape[0]

population_size = 400
parents_ratio = 0.2
num_generations = 3000
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
    p = np.count_nonzero(permuts < res)/len(permuts)
    r = (res) / (max_value)
    r = r + p
    return r if p > 0.4 else 0

def get_p(res):
    return np.count_nonzero(permuts < res)/len(permuts)
    
def evaluate_individual(individual,fit_funct = get_fit):
    individual = np.array(individual)
    num_not_removed = np.sum(individual)
    len_removed = ind_length - num_not_removed
    distance = get_distance(individual)
    fit = fit_funct(distance)
    # Return the fitness values as a tuple
    return len_removed, fit

mut  = 0.0005
cross = 0.02

creator.create("Fitness", base.Fitness, weights=(-1.0, -10.0))
creator.create("Individual", array.array,typecode='b', fitness=creator.Fitness)
toolbox = base.Toolbox()
toolbox.register("individual", create_individual)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("evaluate", evaluate_individual)
toolbox.register("mate", tools.cxUniformPartialyMatched,indpb=cross)
toolbox.register("mutate", tools.mutFlipBit, indpb=mut)
toolbox.register("select", tools.selSPEA2)
toolbox.register("migrate",tools.migRing,k=10,selection = toolbox.select)
stats = tools.Statistics()
stats.register("Num removed", lambda x: x[np.argmin([ind.fitness.values[1] for ind in x])].fitness.values[0])
stats.register("Min max distance", lambda x: np.min([ind.fitness.values[1] for ind in x]))
islands = [toolbox.population(n=400) for _ in range(3)]
population, logbook = GA_utils.eaMuPlusLambda_stop_isl(islands,toolbox, mu=round(len(islands[0]) * parents_ratio), lambda_ = len(islands[0]),cxpb=0.45, mutpb=0.45, ngen=num_generations,stats=stats, verbose=True)

pop = np.flatten(np.array(population))
np.savetxt(os.path.join(args.output,"complete.csv"), pop, delimiter="\t")
ress = np.array([evaluate_individual(x,fit_funct=get_p) for x in pop])

pareto_front = tools.sortNondominated(population, k=population_size,first_front_only=True)
par = np.array([list(x) for x in pareto_front[0]])
parr = np.array([evaluate_individual(x,fit_funct=get_p) for x in par])
if not os.path.exists(args.output):
    os.makedirs(args.output)
np.savetxt(os.path.join(args.output,"pareto.csv"), par, delimiter="\t")


if args.save_plot:
    GA_utils.plot_pareto(ress,parr,args.output)
GA_utils.get_results_from_pareto(par,parr,args.output,expression_data.full.GeneID)
print(mut)
print(cross)