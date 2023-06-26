import matplotlib.pyplot as plt
import numpy as np 
from deap import algorithms, base, creator, tools

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
        if max_len_counter > 70:
            break
    return population, logbook

def plot(GA):
        fig11 =plt.figure(figsize=(13, 8))
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
