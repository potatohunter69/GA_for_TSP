import numpy as np


def calculate_fitness(
    population: np.ndarray, distance_matrix: np.ndarray
) -> np.ndarray:
    """
    Calculates the fitness scores for each route in the population.
    Fitness is defined as the inverse of the total distance.
    """
    from_cities = population[:, :-1]  # All but the last city
    to_cities = population[:, 1:]  # All but the first city

    # Compute the distances for all routes in one go
    total_distances = np.sum(distance_matrix[from_cities, to_cities], axis=1)

    # Avoid division by zero (using np.where for safe computation)
    fitness_scores = np.divide(
        1.0,  # Use float division
        total_distances,
        where=total_distances > 0,
        out=np.full(
            total_distances.shape, float("inf"), dtype=np.float64
        ),  # Ensure float output
    )

    return fitness_scores


def tournament_selection(population, fitness_scores, tournament_size, num_parents):
    """
    Selects parents using tournament selection, but picks the best individual from each tournament.

    :param population: The population from which to select parents.
    :param fitness_scores: The fitness scores of the population.
    :param tournament_size: The number of individuals participating in each tournament.
    :param num_parents: The number of parents to select.
    :return: The selected parents in a list.
    """
    population_size = len(population)

    # Randomly select tournament_size individuals for the tournament at once
    tournament_indices = np.random.choice(
        population_size, (num_parents, tournament_size)
    )

    # Get the fitness scores of the participants
    tournament_fitness = fitness_scores[tournament_indices]

    # Sort the fitness scores
    sorted_indices = np.argsort(tournament_fitness, axis=1)[:, ::-1]

    # Get the best indices from the sorted list
    best_indices = sorted_indices[:, 0]

    # Get the second-best individuals in the population
    best_parents_indices = tournament_indices[np.arange(num_parents), best_indices]

    # Select the second-best individuals from the population
    selected_parents = population[best_parents_indices]

    return selected_parents
