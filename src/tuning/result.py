import pickle
from typing import List
from dataclasses import dataclass

import numpy as np

from genetics import Params


@dataclass
class Result:
    params: Params
    fitness: float
    best_route: np.ndarray
    duration: float


def process_results(results: List[Result]) -> List[Result]:
    """
    Sorts the results by fitness in descending order.
    :param results:
    :return List[Dict]: A list of results sorted by fitness.
    """

    results = [r for r in results if r.fitness is not None]
    sorted_results = sorted(results, key=lambda x: x.fitness, reverse=True)

    return sorted_results
