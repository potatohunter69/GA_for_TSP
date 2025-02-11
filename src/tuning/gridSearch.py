import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Callable
from genetics.parameters import Params
from tools.log import print_estimated_time
from tuning.result import Result
from .result import process_results
import numpy as np
import logging


def test_parameter_combination(
    params: Params, cities: np.ndarray, genetic_algorithm: callable
) -> Result:
    """
    Runs the genetic algorithm for a given set of parameters.

    Args:
        params: Parameter set for the genetic algorithm.
        cities: The dataset of cities (e.g., for TSP).
        genetic_algorithm: Function that executes the genetic algorithm.

    Returns:
        dict: A dictionary containing the parameters, the best fitness, and the best route.
    """
    start_time = time.time()

    try:
        best_route, best_fitness, _, _ = genetic_algorithm(cities, params)
        duration = time.time() - start_time
        return Result(
            params=params,
            fitness=best_fitness,
            best_route=best_route,
            duration=duration,
        )

    except Exception as e:
        logging.error(f"Error with params {params}: {e}")
        return Result(
            params=params, fitness=-1.0, best_route=np.zeros(()), duration=-1.0
        )


def gs_multithreading(
    cities: np.ndarray,
    param_combinations: List[Params],
    genetic_algorithm: Callable,
    timeout: int = None,
):
    results = []

    with ThreadPoolExecutor() as executor:
        logging.info(f"Number of available workers: {executor._max_workers}")
        future_to_params = {
            executor.submit(
                test_parameter_combination, params, cities, genetic_algorithm
            ): params
            for params in param_combinations
        }

        for combination_index, future in enumerate(as_completed(future_to_params), 1):
            params = future_to_params[future]

            try:
                result = future.result(timeout=timeout)
                results.append(result)

            except Exception as e:
                logging.error(f"Error processing parameters {params}: {str(e)}")
    return results


def gs_classic(
    cities: np.ndarray,
    param_combinations: List[Params],
    genetic_algorithm: Callable,
    _: int = None,
):
    results = []
    start_time = time.time()
    total_combinations = len(param_combinations)
    intervals_logged = 0
    index = 0

    for params in param_combinations:
        try:
            results.append(
                test_parameter_combination(params, cities, genetic_algorithm)
            )
            index += 1
            intervals_logged = print_estimated_time(
                index, total_combinations, start_time, intervals_logged
            )
        except Exception as e:
            logging.error(f"Error processing parameters {params}: {str(e)}")
    return results


def search_grid(
    cities: np.ndarray,
    param_combinations: List[Params],
    genetic_algorithm: Callable,
    multithreading: bool = False,
    timeout: int = None,
) -> List[Result]:
    """
    Runs parameter tuning using a genetic algorithm over a list of parameter combinations.
    Can run in parallel or sequentially based on the multithreading flag.

    Args:
        cities: The dataset of cities.
        param_combinations: List of possible parameter sets to evaluate.
        genetic_algorithm: Function that runs the genetic algorithm.
        timeout: Optional timeout for each task (in seconds).
        multithreading: Boolean flag to enable multithreading (parallel execution).
                        If False, the function runs sequentially.

    Returns:
        List[Dict]: A list of results sorted by fitness, containing the parameter set,
                    the best fitness, best route, and duration.
    """

    logging.info(
        f"Starting parameter tuning for {len(param_combinations)} combinations..."
    )
    start_time = time.time()
    res = (
        gs_multithreading(cities, param_combinations, genetic_algorithm, timeout)
        if multithreading
        else gs_classic(cities, param_combinations, genetic_algorithm, timeout)
    )
    total_duration = time.time() - start_time
    logging.info(f"Parameter tuning completed in {total_duration:.2f} seconds.")
    return process_results(res)
