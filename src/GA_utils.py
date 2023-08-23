import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np 
from deap import algorithms, base, creator, tools
import tqdm
import os
import concurrent.futures
import random

class SolutionException(Exception):
    def __init__(self, message):
        super().__init__(message)
        
def varOr(population, toolbox, lambda_, cxpb, mutpb,mutate_funct):
    assert (cxpb + mutpb) <= 1.0, (
        "The sum of the crossover and mutation probabilities must be smaller "
        "or equal to 1.0.")

    offspring = []
    for _ in range(lambda_):
        op_choice = random.random()
        if op_choice < cxpb:            # Apply crossover
            ind1, ind2 = [toolbox.clone(i) for i in random.sample(population, 2)]
            ind1, ind2 = toolbox.mate(ind1, ind2)
            del ind1.fitness.values
            offspring.append(ind1)
        elif op_choice < cxpb + mutpb:  # Apply mutation
            ind = toolbox.clone(random.choice(population))
            ind, = mutate_funct(ind)
            del ind.fitness.values
            offspring.append(ind)
        else:                           # Apply reproduction
            offspring.append(random.choice(population))

    return offspring

def eaMuPlusLambda_stop(population, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)
    prev_max_len = 0
    max_len_counter = 1
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        offspring = algorithms.varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(population + offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)
        
        max_len = max(population, key=lambda ind: ind.fitness.values[0]).fitness.values[0]
        if prev_max_len == max_len:
            max_len_counter += 1
        else:
            prev_max_len = max_len
            max_len_counter = 1
        if max_len_counter > 120:
            break
    return population, logbook


def eaMuPlusLambda_stop_isl(islands, toolbox, mu, lambda_, cxpb, mutpb, ngen,mut_functs_isl: list,
                   stats=None, verbose=__debug__):
    
    def isl_evaluate(invalid_ind):
        return list(toolbox.map(toolbox.evaluate, invalid_ind))
    
    def isl_select(island):
        return toolbox.select(island, mu)
    
    def isl_evolve(island,i):
        return varOr(island, toolbox, lambda_, cxpb, mutpb,mut_functs_isl[i])
    
    def comp_fitness_inv(island):
        inv_ind = [ind for ind in island if not ind.fitness.valid]
        fitnesses = isl_evaluate(inv_ind)
        for ind, fit in zip(inv_ind, fitnesses):
                ind.fitness.values = fit

    
    def island_evolve(island,i):
        offsprings  = isl_evolve(island,i)
        comp_fitness_inv(island)
        comp_fitness_inv(offsprings)
        return isl_select(offsprings + island)
    
    def migrate(islands,gen):
        if min([min(islands[i], key=lambda ind: ind.fitness.values[1]).fitness.values[1] for i in range(len(islands))]) > 0:
            if gen%5 == 0:
                toolbox.migrate(islands)
        else:
            if gen%10 == 0:
                toolbox.migrate(islands)

    def should_stop(islands,prev_max_len,max_len_counter):
        max_len = min([min(islands[i], key=lambda ind: ind.fitness.values[1]).fitness.values[0] for i in range(len(islands))])
        if prev_max_len == max_len:
            max_len_counter += 1
        else:
            prev_max_len = max_len
            max_len_counter = 1
        if max_len_counter > 60:
            return True,prev_max_len,max_len_counter
        return False,prev_max_len,max_len_counter
    
    def log_results(islands,gen):
        for i in range(len(islands)):
            record = stats.compile(islands[i]) if stats is not None else {}
            logbook.record(gen=gen, **record)
            if verbose:
                print(logbook.stream)
                
        print("\n")


    executor = concurrent.futures.ThreadPoolExecutor()

    logbook = tools.Logbook()
    logbook.header = ['gen'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    fitnesses = list(executor.map(comp_fitness_inv, islands))

    record = stats.compile(islands[0]) if stats is not None else {}
    logbook.record(gen=0, **record)
    if verbose:
        print(logbook.stream)

    prev_max_len = 0
    max_len_counter = 1
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        islands = list(executor.map(island_evolve, islands,range(len(islands))))

        # Update the statistics with the new population

        log_results(islands,gen)
        stop,prev_max_len,max_len_counter = should_stop(islands,prev_max_len,max_len_counter)   
        if stop:
            break
        migrate(islands,gen)    

    return islands, logbook


def eaMuPlusLambda_stop_isl_multi(islands1,islands2, toolbox, mu, lambda_, cxpb, mutpb, ngen,
                   stats=None, halloffame=None, verbose=__debug__):
    
    def isl_evaluate(invalid_ind):
        return list(toolbox.map(toolbox.evaluate, invalid_ind))
    
    def isl_select(island):
        return toolbox.select(island, mu)
    
    def isl_evolve(island):
        return algorithms.varOr(island, toolbox, lambda_, cxpb, mutpb)
    
    def comp_fitness_inv(island):
        inv_ind = [ind for ind in island if not ind.fitness.valid]
        fitnesses = isl_evaluate(inv_ind)
        for ind, fit in zip(inv_ind, fitnesses):
                ind.fitness.values = fit

    
    def island_evolve(island):
        offsprings  = isl_evolve(island)
        comp_fitness_inv(island)
        comp_fitness_inv(offsprings)
        return isl_select(offsprings + island)


    executor = concurrent.futures.ThreadPoolExecutor()

    logbook = tools.Logbook()
    logbook.header = ['gen'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    executor.map(comp_fitness_inv, islands1)
    executor.map(comp_fitness_inv, islands2)


    record = stats.compile(islands1[0]) if stats is not None else {}
    logbook.record(gen=0, **record)
    if verbose:
        print(logbook.stream)
    prev_max_len = 0
    max_len_counter = 1
    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Vary the population
        islands1 = list(executor.map(island_evolve, islands1))
        islands2 = list(executor.map(island_evolve, islands2))

        # Update the statistics with the new population
        for i in range(len(islands1)):
            record = stats.compile(islands1[i]) if stats is not None else {}
            logbook.record(gen=gen, **record)
            if verbose:
                print(logbook.stream)
                
        print("\n")
        
        max_len = min([min(islands1[i], key=lambda ind: ind.fitness.values[1]).fitness.values[0] for i in range(len(islands))])
        if prev_max_len == max_len:
            max_len_counter += 1
        else:
            prev_max_len = max_len
            max_len_counter = 1
        if max_len_counter > 60:
            break
        if min([min(islands1[i], key=lambda ind: ind.fitness.values[1]).fitness.values[1] for i in range(len(islands1))]) > 0:
            if gen%5 == 0:
                toolbox.migrate(islands1)
        else:
            if gen%10 == 0:
                toolbox.migrate(islands1)
        

    return islands1, islands2, logbook

def plot(GA):
        fig11 = plt.figure(figsize=(13, 8))
        x = np.linspace (0.0000005, 0.000001, 300) 

        #calculate pdf of Gamma distribution for each x-value
        y = GA.gamma.pdf(x)
        best_inds = np.argsort(GA.fitness)[-15:]
        best = best_inds[-1]
        best_len = GA.ind_length - np.sum(GA.population[best]) 
        fitness = GA.fitness[best_inds]
        best_sols = GA.population[best_inds]
        col = [150 if x == max(fitness) else 40 for x in fitness]

        plt.style.use('seaborn-v0_8-pastel')
        fig =plt.figure(GA.curr_gen,figsize=(13, 8))
        grid = fig.add_gridspec(3, 10, wspace=1.5, hspace=0.5)
        ax1 = plt.subplot(grid[:2, :-1])
        ax2 = plt.subplot(grid[:, -1])
        ax3 = plt.subplot(grid[2, :-1])
        varr,_ = GA.get_var_and_p_single(GA.population[best])

        for sol,f in zip(GA.get_avgs(best_sols),col):
            ax1.plot(["Zygote", "Quadrant","Globular","Heart","Torpedo","Bent","Mature"], sol, lw=3,c=plt.cm.Greens(f))
            ax1.plot(["Zygote", "Quadrant","Globular","Heart","Torpedo","Bent","Mature"], GA.get_avgs(np.ones(GA.ind_length)), lw=3,c="Grey")
            ax1.set_ylim([3, 3.42])
            ax1.set_xlabel("Stage")
            ax1.set_ylabel("TAI")
        
        ax2.bar(["Removed"], best_len,color="Green")
        ax2.set_ylim([0, 900])
        ax2.yaxis.tick_right()
        ax3.plot(x, y)
        ax3.set_xlabel("Variance")
        ax3.set_ylabel("pdf")
        if varr  < 0.00000194:
            plt.vlines(varr, 0, GA.gamma.pdf(varr),colors=["Green"])
        plt.savefig("./best_graphs/best" + str(GA.curr_gen) + ".png")
        plt.close()

def comp_mean_prec(expression_data, permutations):
    Divisor = expression_data.expressions_n.sum(axis=0)
    fMatrix = expression_data.expressions_n / Divisor
    permMatrix = np.array([np.random.permutation(expression_data.full["Phylostratum"]) for _ in tqdm.trange(permutations)])
    bootM = permMatrix @ fMatrix
    bootM = np.unique(bootM, axis=1)
    mean = np.mean(bootM, axis=0)
    cov_matrix = np.cov(bootM, rowvar=False)
    precision_matrix = np.linalg.inv(cov_matrix)
    return mean,precision_matrix

def comp_vars(expression_data,rounds):
    avgs = []
    phil = expression_data.full["Phylostratum"]
    print("Running permuations")
    for _ in tqdm.trange(rounds):
        perm = np.random.permutation(phil)
        weighted = expression_data.expressions.mul(perm, axis=0)
        avg = weighted.sum(axis=0)/expression_data.expressions_n.sum(axis=0)
        avgs.append(avg)
    return np.var(avgs, axis=1)

def comp_min_max(expression_data,rounds):
    avgs = []
    phil = expression_data.full["Phylostratum"]
    print("Running permuations")
    for _ in tqdm.trange(rounds):
        perm = np.random.permutation(phil)
        weighted = expression_data.expressions.mul(perm, axis=0)
        avg = weighted.sum(axis=0)/expression_data.expressions_n.sum(axis=0)
        avgs.append(avg)
    return np.max(avgs, axis=1) - np.min(avgs, axis=1)

def get_sol_from_indices(indices,ind_len):
    ones = np.ones(ind_len)
    ones[indices] = 0
    return ones

def get_removed_genes_from_solution(solution,GeneIds):
    return np.array(GeneIds[np.where(solution == 0)[0]])

def plot_pareto(pareto,par,folder):
    plt.scatter(pareto[:,0],pareto[:,1],s = 1.5,color='blue', marker='o', label='Solution')
    plt.scatter(par[:,0],par[:,1],s = 5,color='red', marker='o', label='Front')
    #plt.gca().invert_yaxis()
    #plt.gca().invert_xaxis()

    plt.xlabel('Number of extracted genes')  # X-axis label
    plt.ylabel('p-value')  # Y-axis label

    plt.title('Solution extraction')  # Title

    plt.legend()  # Show legend

    plt.grid(True, linestyle='--', alpha=0.7)  # Add gridlines

    plt.xticks(fontsize=12)  # Customize tick labels
    plt.yticks(fontsize=12)
    #selected = pareto[np.logical_and(pareto[:,1] < 0.6,pareto[:,1] > 0.4)][:,0]
    selected = pareto
    if len(selected) == 0:
        raise SolutionException("No solution found")
    
    rect_x = np.min(selected[:,0])
    rect_y = 0.4
    rect_width = rect_x*0.1
    rect_height = 0.6 - 0.4

    # Create a Rectangle patch
    rectangle = patches.Rectangle((rect_x, rect_y), rect_width, rect_height,
                                linewidth=2, edgecolor='red', facecolor='none', linestyle='dashed')

    # Add the Rectangle patch to the current plot
    plt.gca().add_patch(rectangle)
    # Save or display the plot
    plt.tight_layout()  # Adjust layout to prevent clipping of labels
    # Save the plot as an image
    
    plt.savefig(os.path.join(folder, "pareto_front.png")) 
    plt.show()

def get_results_from_pareto(solutions,pareto,folder,GeneIds):
    #pareto_filtered= pareto[np.logical_and(pareto[:,1] < 0.6,pareto[:,1] > 0.4)]
    pareto_filtered = pareto
    #solutions = solutions[np.logical_and(pareto[:,1] < 0.6,pareto[:,1] > 0.4)]
    solutions = solutions
    if len(pareto_filtered) == 0:
        raise SolutionException("No solution found")
    min_v = min(pareto[:,0])
    sel_sols  = solutions[pareto_filtered[:,0] < min_v + min_v *0.1]
    
    genes = get_removed_genes_from_solution(get_sol_from_indices(np.where(len(sel_sols) - sel_sols.sum(axis=0) >= round(len(sel_sols)*0.8))[0],sel_sols.shape[1]),GeneIds)
    np.savetxt(os.path.join(folder,"extracted_genes.txt"),genes, fmt="%s")