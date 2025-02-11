from dataclasses import dataclass


@dataclass
class Params:
    population_size: int
    generations: int
    elite_size: int
    tournament_size: int
    mutation_rate: float
    mutation_type: str = "swap"
    crossover_type: str = "ox"
    initial_population: str = "nn"


from itertools import product
from typing import List, Optional


def generate_param_grid(
    population_sizes: Optional[List[int]] = None,
    generations_list: Optional[List[int]] = None,
    elite_sizes: Optional[List[int]] = None,
    tournament_sizes: Optional[List[int]] = None,
    mutation_rates: Optional[List[float]] = None,
    mutation_types: Optional[List[str]] = None,
    crossover_types: Optional[List[str]] = None,
) -> List[Params]:
    """
    Generate all possible parameter combinations for the genetic algorithm.
    Returns:
        List[Params]: A list of all parameter combinations as Params objects.
    """
    population_sizes = population_sizes or []
    generations_list = generations_list or []
    elite_sizes = elite_sizes or []
    tournament_sizes = tournament_sizes or []
    mutation_rates = mutation_rates or []
    mutation_types = mutation_types or ["swap"]
    crossover_types = crossover_types or ["ox"]

    # Generate parameter combinations using itertools.product
    param_combinations = product(
        population_sizes,
        generations_list,
        elite_sizes,
        tournament_sizes,
        mutation_rates,
        mutation_types,
        crossover_types,
    )

    # Create Params objects for each combination
    return [Params(*params) for params in param_combinations]
