[![Lifecycle: experimental](https://img.shields.io/badge/lifecycle-experimental-orange.svg)](https://lifecycle.r-lib.org/articles/stages.html#experimental)
![Visitors](https://api.visitorbadge.io/api/visitors?path=https://github.com/lavakin/ga_for_hourglass&label=Visitors&countColor=%23263759&style=flat)
# Hourglass destroyer
## Description

The Developmental Gene Analysis project features an algorithm designed to identify genes contributing to the observed hourglass pattern in developmental gene expression data. The hourglass pattern refers to a characteristic developmental pattern observed in various organisms, where the morphological and genetic similarities between individuals are most pronounced at a specific stages of development.

This algorithm aims to identify a subset of genes that, if removed from the dataset, would significantly reduce the presence of the hourglass pattern. By employing a multi-objective genetic algorithm, it maximizes the removal of the hourglass pattern while minimizing the number of removed genes.

The algorithm utilizes the DEAP (Distributed Evolutionary Algorithms in Python) library, which provides a flexible framework for implementing genetic algorithms. It offers various selection methods, mutation operators, and genetic operators to evolve populations of candidate solutions.

https://github.com/lavakin/ga_for_hourglass/assets/76702901/5435d04b-151b-46da-b179-d48ca9a7e5ce

## Key Features

- Multi-objective optimization: Identifies a subset of genes that, when removed, minimizes the presence of the hourglass pattern.
- Developmental gene expression analysis: Analyzes gene expression data to identify hourglass pattern genes.
- Genetic algorithm: Utilizes DEAP library for efficient evolutionary computation.
- Customizable: Easily adjustable parameters and fitness functions for different analyses.
- Visualization: Provides tools for visualizing the hourglass pattern and selected gene subset.

Apologies for the misunderstanding. Here's the revised "About" section that reflects the algorithm's objective of identifying a subset of genes that, when removed from the dataset, minimizes the presence of the hourglass pattern:

markdown

## Project Name

Developmental Gene Analysis: Hourglass Pattern Optimization Algorithm

## Description

The Developmental Gene Analysis project features an algorithm designed to optimize the removal of genes contributing to the observed hourglass pattern in developmental gene expression data. The hourglass pattern refers to a characteristic developmental pattern observed in various organisms, where the morphological and genetic similarities between individuals are most pronounced at a specific stage of development.

This algorithm aims to identify a subset of genes that, if removed from the dataset, would significantly reduce the presence of the hourglass pattern. By employing a multi-objective genetic algorithm, it maximizes the removal of the hourglass pattern while minimizing the disruption to the overall gene expression profile.

The algorithm utilizes the DEAP (Distributed Evolutionary Algorithms in Python) library, which provides a flexible framework for implementing genetic algorithms. It offers various selection methods, mutation operators, and genetic operators to evolve populations of candidate solutions.

## Key Features

- Multi-objective optimization: Identifies a subset of genes that, when removed, minimizes the presence of the hourglass pattern.
- Developmental gene expression analysis: Analyzes gene expression data to identify hourglass pattern genes.
- Genetic algorithm: Utilizes DEAP library for efficient evolutionary computation.
- Customizable: Easily adjustable parameters and fitness functions for different analyses.
- Visualization: Provides tools for visualizing the hourglass pattern and selected gene subset.

## Installation
To use this project, follow these steps:

1. Clone the repository: `git clone https://github.com/lavakin/ga_for_hourglass.git`
2. Install the required dependencies: `pip install -r requirements.txt`
3. Configure the input data and parameters in the provided configuration files.
4. Run the main script: `python main.py`

## Contributing

Contributions to this project are welcome. If you have any ideas for improvements, new features, or bug fixes, please submit a pull request. For major changes, please open an issue to discuss the proposed modifications.

## License

This project is licensed under the [MIT License](https://opensource.org/licenses/MIT). Feel free to use and modify the code according to the terms of this license.

## Example usage
