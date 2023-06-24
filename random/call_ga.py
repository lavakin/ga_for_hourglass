import pandas as pd
from hourglass import Expression_data, GA
import os
import numpy as np

csv_file_path = PATH
arr = pd.read_csv(csv_file_path,
                delimiter=",")
expression_data = Expression_data(arr)
ga = GA(expression_data,3200,2,50,0.55,0.25)  
ga = ga()
best_solution = ga.population[np.argmax(ga.fitness)]
genes = expression_data.full.iloc[expression_data.full.index.values[np.where(best_solution == 0)[0]],:]["GeneID"]
outfile_path = os.path.join('outputs', 
csv_file_path.split("/")[-1].split(".")[-2] + ".txt")
with open(outfile_path, 'w') as f:
    for g in genes:
        f.write(f"{g}\n")
