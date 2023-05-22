
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random
import seaborn as sns
import scipy
import tqdm
import multiprocess as mp
from functools import partial
from timeit import default_timer as timer

class Expression_data:
    def __init__(self,expression_data) -> None:
        self.full = expression_data
        exps = expression_data.iloc[:, 3:]
        self.age_weighted = exps.mul(expression_data["Phylostratum"], axis=0).to_numpy()
        self.expressions_n = exps.to_numpy()
        self.expressions = exps

class GA:
    
    @staticmethod
    def flat_line_test_g_dist(expression_data,rounds):
        phil = expression_data.full['Phylostratum']
        variances = []
        print("Running permuations")
        for _ in tqdm.trange(rounds):
            perm = np.random.permutation(phil)
            weighted = expression_data.expressions.mul(perm, axis=0)
            avg = weighted.sum(axis=0)/expression_data.expressions.sum(axis=0)
            variances.append(np.var(avg))
        shape, loc, scale = scipy.stats.gamma.fit(variances)
        return scipy.stats.gamma(shape, scale=scale,loc=loc)
    
    @staticmethod
    def create_population(ind_length,pop_size,init_num_removed):
        a =  ind_length//(init_num_removed*3)
        b = ind_length//init_num_removed
        pop = np.array([random.choices([0,1], weights=(1, random.randint(a,b)), k=ind_length) for _  in range(pop_size)],dtype="b")
        
        return pop

    def __init__(self,expression_data,pop_size,num_gen,init_num_removed,mutation_probability,crossover_probability) -> None:
        ind_length = expression_data.full.shape[0]
        num_parents = round(pop_size * 0.15)
        #self.gamma = GA.flat_line_test_g_dist(expression_data,1000)
        self.gamma = scipy.stats.gamma(1.5114422884072107, scale=0.0003905211020188656,loc=5.369238206226383e-06)
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
        return varr,  1 - np.array(self.gamma.cdf(varr))
    
    def get_p_value(self):
        return  1 - np.array(self.gamma.cdf(self.get_tai_var()))
        
    def fitness_funct(self):
        num_not_removed = np.sum(self.population,axis = 1)
        num_removed = self.ind_length - num_not_removed
        self.fitness = self.get_p_value() 
        self.fitness = self.fitness + (self.fitness * (1 - num_removed/self.ind_length))
        

    def on_gen(self):       
        if self.curr_gen % 5 == 0:
            print("Generation : ", self.curr_gen)
            max_fitness_on = np.argmax(self.fitness)
            best_solution = self.population[max_fitness_on]
            varr,p = self.get_var_and_p_single(best_solution)
            print("Fitness of the best solution :", self.fitness[max_fitness_on])
            print("Variance of the best solution :", varr)
            print("P-value of the best solution :", p)
            print("Length of the best solution :", len(best_solution) - sum(best_solution))


    @staticmethod
    def mutate2(mut_prob,offspring):
        if random.random() < mut_prob:
            randomlist = random.sample(range(len(offspring)), k=random.getrandbits(4))
            #randomlist1 = np.random.randint(0,len(offspring), size=random.randint(1,400))
            for ind in randomlist:
                offspring[ind] = (offspring[ind] + 1) % 2
            #offspring[randomlist1] = 1
        return offspring

    def mutation_func2(self):
        mut = partial(GA.mutate2, self.mutation_probability)
        with mp.Pool() as pool2:
            res = np.array(list(map(mut, self.population)))
            self.population = res
    
    
    def steady_state_selection(self):
        f_s = self.fitness.argsort()[::-1]
        num_good = round(self.num_parents * 0.9)
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
            gene_sources = np.random.randint(0, 4, size=self.ind_length)
            self.population[k, :] =  np.where(gene_sources > 0, self.parents[parent1_idx, :], self.parents[parent2_idx, :])
    
    def __call__(self):
        self.fitness_funct()
        for self.curr_gen in range(ga.num_gen):
            self.steady_state_selection()
            self.scattered_crossover()
            self.mutation_func2()
            self.fitness_funct()
            self.on_gen()
        return self


            
arr = pd.read_csv("phylo.csv",
                 delimiter=",")
expression_data = Expression_data(arr)
start = timer()
ga = GA(expression_data,2500,180,3,0.55,0.2)  
"""
solutions = []
for _ in range(10):
    ga_c = copy.deepcopy(ga)
    ga_finish = ga_c()
    solutions.append(ga_finish.population[np.argmax(ga_finish.fitness)])
"""
ga()
end = timer()
print(end - start)
#np.savetxt("solutions.csv", np.array(solutions), delimiter=",")





    