import matplotlib.pyplot as plt
import numpy as np 


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
