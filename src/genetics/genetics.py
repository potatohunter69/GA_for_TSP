import numpy as np

from genetics.initialize import (
    gen_population,
    get_distance_matrix,
    validate_cities,
)
from genetics.crossover import crossover
from genetics.mutation import mutation
from genetics.selection import tournament_selection, calculate_fitness
from genetics.parameters import Params


def evolve_population(
    population: np.ndarray, fitness_scores: np.ndarray, params: Params
):
    # Step 1: Elitism - retain the top elite_size individuals
    elite_indices = fitness_scores.argsort()[-params.elite_size :][::-1]
    elite = population[elite_indices]

    # Step 2: Selection - select parents using tournament selection
    num_parents = len(population) - params.elite_size
    parents = tournament_selection(
        population, fitness_scores, params.tournament_size, num_parents
    )

    # Step 3: Crossover - generate offspring from selected parents
    offspring = np.empty((num_parents, len(parents[0])), dtype=int)
    for i in range(0, num_parents, 2):
        parent1 = parents[i]
        parent2 = parents[i + 1] if i + 1 < num_parents else parents[0]
        child1, child2 = crossover(parent1, parent2, params.crossover_type)
        offspring[i] = child1
        if i + 1 < num_parents:
            offspring[i + 1] = child2

    # Step 4: Mutation - mutate offspring
    mutated_offspring = mutation(offspring, params.mutation_rate)

    # Step 5: Create new population by combining elites and mutated offspring
    new_population = np.vstack((elite, mutated_offspring))
    return new_population


def run_genetic_algorithm(cities: np.ndarray, params: Params):
    # Step 1: Validate city data
    validate_cities(cities)

    # Step 2: Generate distance matrix
    distance_matrix = get_distance_matrix(cities)
    # print("Distance matrix generated.")

    # Step 3: Generate initial population
    population = gen_population(
        params.initial_population, params.population_size, distance_matrix
    )

    # Initialize variables to track progress
    best_fitness_history = []
    best_route_history = []

    for generation in range(params.generations):
        # Step 4: Calculate fitness scores
        fitness_scores = calculate_fitness(population, distance_matrix)

        # Record the best fitness and route
        best_fitness = fitness_scores.max()
        best_index = fitness_scores.argmax()
        best_fitness_history.append(best_fitness)
        best_route_history.append(population[best_index])

        # Step 5: Evolve population
        population = evolve_population(population, fitness_scores, params)

    # After all generations, find the best route
    final_fitness_scores = calculate_fitness(population, distance_matrix)
    best_index = final_fitness_scores.argmax()
    best_route = population[best_index]
    best_fitness = final_fitness_scores[best_index]

    return best_route, best_fitness, best_fitness_history, best_route_history
