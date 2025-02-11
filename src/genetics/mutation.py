import numpy as np
import random

"""
### Mutation Functions for TSP Genetic Algorithm

This block contains various mutation functions used in the Genetic Algorithm (GA) for solving the Traveling Salesman Problem (TSP). Each function modifies a tour (array of city indices) to introduce genetic diversity and explore different solutions. The available mutation strategies are:

1. **swap_mutation**: Swaps two randomly selected cities in the tour.
2. **inversion_mutation**: Reverses the order of cities between two randomly selected indices.
3. **scramble_mutation**: Randomly shuffles a subset of cities between two selected indices.
4. **insert_mutation**: Moves a city from one position to another within the tour.
5. **displacement_mutation**: Removes a segment of the tour and reinserts it at a different position.
6. **two_opt_mutation**: Reverses a segment of the tour to optimize local paths.
Each function takes a NumPy array representing a tour and returns the mutated tour.
"""


def swap_mutation(route, mutation_rate=0.01):
    """
    Performs swap mutation on a route with a given mutation rate.
    """
    for i in range(1, len(route) - 1):
        if np.random.rand() < mutation_rate:
            j = np.random.randint(1, len(route) - 1)
            # Swap cities
            route[i], route[j] = route[j], route[i]
    return route


def inversion_mutation(tour: np.ndarray) -> np.ndarray:
    i, j = sorted(random.sample(range(len(tour)), 2))
    tour[i : j + 1] = tour[i : j + 1][::-1]
    return tour


def scramble_mutation(tour: np.ndarray) -> np.ndarray:
    i, j = sorted(random.sample(range(len(tour)), 2))
    sub_tour = tour[i : j + 1]
    random.shuffle(sub_tour)
    tour[i : j + 1] = sub_tour
    return tour


def insert_mutation(tour: np.ndarray) -> np.ndarray:
    i = random.randint(0, len(tour) - 1)
    city = tour[i]
    tour = np.delete(tour, i)  # Remove the city
    j = random.randint(0, len(tour))  # New position can be at the end
    tour = np.insert(tour, j, city)  # Insert the city
    return tour


def displacement_mutation(tour: np.ndarray) -> np.ndarray:
    i, j = sorted(random.sample(range(len(tour)), 2))
    segment = tour[i : j + 1]
    tour = np.delete(tour, slice(i, j + 1))  # Remove the segment
    insert_pos = random.randint(0, len(tour))  # Can be inserted at the end
    tour = np.insert(tour, insert_pos, segment)  # Insert the segment
    return tour


def two_opt_mutation(tour: np.ndarray) -> np.ndarray:
    i, j = sorted(random.sample(range(len(tour)), 2))
    tour[i : j + 1] = reversed(tour[i : j + 1])
    return tour


def mutation(
    gen: np.ndarray, mutation_rate: float = 0.01, mutation_algo: str = "swap"
) -> np.ndarray:
    """
    Perform mutation on multiple TSP routes (2D array) with a given mutation rate and mutation algorithm.

    Parameters:
    - gen (np.ndarray): A 2D numpy array where each row represents a tour (chromosome).
    - mutation_rate (float): Probability of mutation for each tour, default is 0.01 (1%).
    - mutation_algo (str): Mutation algorithm, default is "swap_mutation".

    Returns:
    - np.ndarray: The mutated 2D array of tours.
    """

    mutation_algo_dict = {
        "swap": swap_mutation,
        "inversion": inversion_mutation,
        "scramble": scramble_mutation,
        "insert": insert_mutation,
        "displacement": displacement_mutation,
        "two_opt": two_opt_mutation,
    }

    for idx, tour in enumerate(gen):
        if np.random.rand() < mutation_rate:
            gen[idx] = mutation_algo_dict[mutation_algo](tour)

    return gen
