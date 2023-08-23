[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https://github.com/lavakin/ga_for_hourglass&label=Visitors&countColor=%23263759&style=flat)
# Hourglass Destroyer
## Description

The Hourglass Destroyer project features an algorithm designed to identify genes contributing to the observed hourglass pattern in developmental gene expression data. The hourglass pattern refers to a characteristic developmental pattern observed in various organisms, where the morphological and genetic similarities between individuals are most pronounced at a specific stage of development.

This algorithm aims to identify a subset of genes that, if removed from the dataset, would significantly reduce the presence of the hourglass pattern. By employing a multi-objective genetic algorithm, it maximizes the removal of the hourglass pattern while minimizing the number of removed genes.

The algorithm utilizes the DEAP (Distributed Evolutionary Algorithms in Python) library, which provides a flexible framework for implementing genetic algorithms. It offers various selection methods, mutation operators, and genetic operators to evolve populations of candidate solutions.

Additionally, to enhance its search capability and avoid being trapped in local optima, the algorithm employs an island model. This approach involves maintaining multiple subpopulations, or "islands," that evolve independently. Periodic migration of individuals between islands allows for sharing of genetic information and prevents the algorithm from converging prematurely to suboptimal solutions. This utilization of the island model enhances the algorithm's ability to explore the solution space and discover more globally optimal solutions.

https://github.com/lavakin/ga_for_hourglass/assets/76702901/5435d04b-151b-46da-b179-d48ca9a7e5ce

## Key Features

- Multi-objective optimization: Identifies a subset of genes that, when removed, minimizes the presence of the hourglass pattern.
- Developmental gene expression analysis: Analyzes gene expression data to identify hourglass pattern genes.
- Genetic algorithm: Utilizes DEAP library for efficient evolutionary computation.
- Customizable: Easily adjustable parameters and fitness functions for different analyses.
- Visualization: Provides tools for visualizing the hourglass pattern and selected gene subset.

## Installation
To use this project, follow these steps:

1. Clone the repository: `git clone https://github.com/lavakin/ga_for_hourglass.git`
2. Install the required dependencies: `bash setup.sh`
3. Run the main script: `python src/DEAP_imp.py input_file output_folder  --save_plot`

## Contributing

Contributions to this project are welcome. If you have any ideas for improvements, new features, or bug fixes, please submit a pull request. For major changes, please open an issue to discuss the proposed modifications.

## License

This project is licensed under the GNU License. Feel free to use and modify the code according to the terms of this license.
