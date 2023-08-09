
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import scipy
import tqdm
import copy
import multiprocess as mp
from functools import partial
from timeit import default_timer as timer
from GA_utils import * 
import sys

variances = np.loadtxt("min_max_raw.txt")

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

class GA:
    
    
    @staticmethod
    def create_population(ind_length,pop_size,init_num_removed):
        a =  ind_length//(init_num_removed*3)
        b = ind_length//init_num_removed
        pop = np.array([random.choices([0,1], weights=(1, random.randint(a,b)), k=ind_length) for _  in range(pop_size)],dtype="b")
        
        return pop
    
    def get_distance(self):
        up = self.population.dot(self.expression_data.age_weighted)
        down = self.population.dot(self.expression_data.expressions_n)
        avgs = np.divide(up,down)
        return np.max(avgs, axis=1) - np.min(avgs, axis=1)

    def get_fit_dist(self):
        dists =  self.get_distance()
        r = (dists - self.min_value) / (self.max_value - self.min_value)
        p = np.vectorize(lambda x: np.count_nonzero(variances < x))(dists)/len(variances)
        r = p + r
        r = np.array([r[i] if p[i] > 0.5 else 0 for i in range(len(p))])
        return r
        

    def __init__(self,expression_data,pop_size,num_gen,init_num_removed,mutation_probability,crossover_probability) -> None:
        ind_length = expression_data.full.shape[0]
        num_parents = round(pop_size * 0.15)
        self.pop_size = pop_size
        self.ind_length =ind_length
        self.num_gen = num_gen
        self.population = GA.create_population(ind_length,pop_size,init_num_removed)
        self.fitness = np.zeros(pop_size)
        self.parents = np.ones((num_parents,ind_length),dtype="b")
        self.mutation_probability = mutation_probability
        self.num_parents = num_parents
        self.crossover_probability = crossover_probability
        self.curr_gen = 0
        self.expression_data = expression_data
        self.best_solutions = []
        self.stop = False
        self.prev_fit = 0
        self.p_prev_fit = 0
        self.prev_len = ind_length
        self.p_prev_len = ind_length
        self.min_value = 0
        self.max_value = 0.3

    def get_tai_var(self):
        up = self.population.dot(self.expression_data.age_weighted)
        down = self.population.dot(self.expression_data.expressions_n)
        avgs = np.divide(up,down)
        return np.var(avgs,axis=1)

    def get_var_and_p_single(self,solution):
        up = solution.dot(self.expression_data.age_weighted)
        down = solution.dot(self.expression_data.expressions_n)
        avgs = np.divide(up,down)
        varr = np.var(avgs)
        return np.max(avgs) - np.min(avgs)
    
        
    def fitness_funct(self):
        num_not_removed = np.sum(self.population,axis = 1)
        num_removed = self.ind_length - num_not_removed
        #rem_ratio = 1 - num_removed/self.ind_length
        self.fitness = self.get_fit_dist()
        self.fitness = self.fitness + (num_removed/self.ind_length)
        
    def get_avgs(self,sols):
        up = sols.dot(self.expression_data.age_weighted)
        down = sols.dot(self.expression_data.expressions_n)
        return np.divide(up,down)
        

    def on_gen(self):       
        if self.curr_gen % 10 == 0:
            print("Generation : ", self.curr_gen)
            max_fitness_on = np.argmin(self.fitness)
            best_solution = self.population[max_fitness_on]
            min_max = self.get_var_and_p_single(best_solution)
            print("Fitness of the best solution :", self.fitness[max_fitness_on])
            print("Min max of the best solution :", min_max)
            print("Length of the best solution :", len(best_solution) - sum(best_solution))
            print("Mutation probability :", self.mutation_probability)
        
        if self.curr_gen % 3 == 0:
            max_fitness = np.min(self.fitness)
            if self.prev_fit > self.p_prev_fit and max_fitness > self.prev_fit:
                self.mutation_probability -= 0.03
            self.p_prev_fit = self.prev_fit
            self.prev_fit = max_fitness

        if self.curr_gen % 10 == 0:
            best_solution = self.population[np.argmin(self.fitness)]
            best_sol_len = len(best_solution) - sum(best_solution)
            if self.prev_len == self.p_prev_len and self.prev_len == best_sol_len:
                self.stop = True
            self.p_prev_len = self.prev_len
            self.prev_len = best_sol_len

        """
            max_fitness_on = np.argmax(self.fitness)
            best_solution = self.population[max_fitness_on]
            self.best_solutions.append(best_solution)
            plot(self)
        """


    @staticmethod
    def mutate2(mut_prob,offspring):
        if random.random() < mut_prob:
            randomlist = random.sample(range(len(offspring)), k=random.getrandbits(5))
            for ind in randomlist:
                offspring[ind] = (offspring[ind] + 1) % 2
        return offspring

    def mutation_func2(self):
        mut = partial(GA.mutate2, self.mutation_probability)
        res = np.array(list(map(mut, self.population)))
        self.population = res
    
    
    def steady_state_selection(self):
        f_s = self.fitness.argsort()
        num_good = round(self.num_parents * 0.85)
        num_bad = self.num_parents  - num_good
        bad_inds = np.array(random.sample(range(num_good,self.pop_size),k=num_bad))
        self.parents[:num_good, :] = self.population[f_s[:num_good], :].copy()
        self.parents[num_good:, :] = self.population[f_s[bad_inds], :].copy()


    def scattered_crossover(self):
        for k in range(self.pop_size):
            if not (self.crossover_probability is None):
                probs = np.random.random(size=self.parents.shape[0])
                indices = np.where(probs <= self.crossover_probability)[0]

                # If no parent satisfied the probability, no crossover is applied and a parent is selected.
                if len(indices) == 0:
                    self.population[k, :] = self.parents[k % self.parents.shape[0], :]
                    continue
                elif len(indices) == 1:
                    parent1_idx = indices[0]
                    parent2_idx = parent1_idx
                else:
                    parent1_idx,parent2_idx = random.sample(list(set(indices)), 2)
            else:
                # Index of the first parent to mate.
                parent1_idx = k % self.parents.shape[0]
                # Index of the second parent to mate.
                parent2_idx = (k+1) % self.parents.shape[0]

            # A 0/1 vector where 0 means the gene is taken from the first parent and 1 means the gene is taken from the second parent.
            gene_sources = np.random.randint(0, 5, size=self.ind_length)
            self.population[k, :] =  np.where(gene_sources > 0, self.parents[parent1_idx, :], self.parents[parent2_idx, :])
    
    def __call__(self):
        self.fitness_funct()
        for self.curr_gen in range(self.num_gen):
            self.steady_state_selection()
            self.scattered_crossover()
            self.mutation_func2()
            self.fitness_funct()
            self.on_gen()
            if self.stop:
                break
        return self


file_name = sys.argv[1]
arr = pd.read_csv(file_name,
                 delimiter=",")
expression_data = Expression_data(arr)
start = timer()
solutions = []
for _ in range(10):
    ga = GA(expression_data,4000,600,300,0.3,0.25)
    ga_finish = ga()
    solutions.append(ga_finish.population[np.argmax(ga_finish.fitness)])

np.savetxt("../results/new/phylo.csv",solutions,delimiter="\t" )
end = timer()
print(end - start)
summed = np.array(solutions).sum(axis=0)
sol = get_sol_from_indices(np.where(summed <= 3)[0],ga.ind_length)
genes = get_removed_genes_from_solution(sol,expression_data.full)
genes.to_csv("../results/new/ex_genes.csv")
#np.savetxt("../results/victor/" + file_name.split("/")[-1], solutions, delimiter="\t")





    