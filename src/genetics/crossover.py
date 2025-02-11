"""
File name: order_crossover.py
Author: Sami Noroozi
Created: 19/9/2024
Version: 1
"""

import random
from typing import List, Tuple

import numpy as np


def order_crossover(
    parent1: np.ndarray, parent2: np.ndarray
) -> (np.ndarray, np.ndarray):
    """
    Performs Order Crossover (OX) between two parents to produce two offspring.
    """
    size = len(parent1)
    start, end = sorted(np.random.choice(range(1, size - 2), 2, replace=False))

    def fill_remaining(p1, p2):
        current_pos = (end + 1) % size

        offspring = np.full(size, -1, dtype=int)
        offspring[start : end + 1] = p1[start : end + 1]

        genes_set = set(offspring[start : end + 1])

        for gene in p2:
            if gene not in genes_set:
                offspring[current_pos] = gene
                genes_set.add(gene)

                current_pos += 1
                if current_pos == size:
                    current_pos = 1

        if offspring[0] == -1:
            offspring[0] = offspring[-1]
        return offspring

    offspring1 = fill_remaining(parent2, parent1)
    offspring2 = fill_remaining(parent1, parent2)

    return offspring1, offspring2


def partially_mapped_crossover(parent1, parent2):
    """
    Performs Partially Mapped Crossover (PMX) between two parents to produce two offspring.
    """
    size = len(parent1) - 1
    a, b = sorted(np.random.choice(range(size), 2, replace=False))

    def pmx_create_child(p1, p2):
        child = [None] * size
        child[a:b] = p1[a:b]

        mapping = {p1[i]: p2[i] for i in range(a, b)}

        for i in range(size):
            if not (a <= i < b):
                city = p2[i]
                while city in mapping:
                    city = mapping[city]
                child[i] = city

        child.append(child[0])
        return child

    child1 = pmx_create_child(parent1, parent2)
    child2 = pmx_create_child(parent2, parent1)

    return child1, child2


def cycle_crossover(parent1, parent2):
    """
    Performs Cycle Crossover (CX) between two parents to produce two offspring.
    """
    size = len(parent1) - 1

    def cx_create_child(p1, p2):
        child = [None] * size
        index = 0
        while None in child:
            if child[index] is None:
                start = index
                while True:
                    child[index] = p1[index]
                    index = p1.index(p2[index])
                    if index == start:
                        break
            index = child.index(None) if None in child else size
        child.append(child[0])
        return child

    child1 = cx_create_child(parent1, parent2)
    child2 = cx_create_child(parent2, parent1)

    return child1, child2


def position_based_crossover(parent1, parent2):
    """
    Performs Position-Based Crossover (PBX) between two parents to produce two offspring.

    A subset of positions is randomly selected from one parent, and those values are placed in the same
    positions in the child. The remaining positions are filled with the other parent's values in the order
    they appear, skipping over the already selected elements.
    """
    size = (
        len(parent1) - 1
    )  # We exclude the last city as it repeats the first (circular TSP)
    positions = sorted(np.random.choice(range(size), size // 2, replace=False))

    def pbx_create_child(p1, p2):
        child = [None] * size
        # Copy elements from parent1 at the selected positions
        for pos in positions:
            child[pos] = p1[pos]

        # Fill remaining positions from parent2 in order
        current_idx = 0
        for city in p2:
            if city not in child:
                while child[current_idx] is not None:
                    current_idx += 1
                child[current_idx] = city

        child.append(child[0])  # Make it circular for the TSP
        return child

    child1 = pbx_create_child(parent1, parent2)
    child2 = pbx_create_child(parent2, parent1)

    return child1, child2


def crossover(parent1, parent2, method="ox"):
    """
    Performs crossover between two parent solutions using the specified crossover method.

    Parameters:
    -----------
    parent1 : list or array
        The first parent, representing a sequence of cities (a potential solution to the TSP).

    parent2 : list or array
        The second parent, representing a sequence of cities (another potential solution to the TSP).

    method : str, optional, default='ox'
        The crossover method to use. Options are:
            - 'ox' : Order Crossover (OX)
            - 'pmx' : Partially Mapped Crossover (PMX)
            - 'cx' : Cycle Crossover (CX)
            - 'pbx' : Position-Based Crossover (PBX)

    Returns:
    --------
    tuple
        Two offspring (child1, child2), each representing a new sequence of cities, produced by combining the parent solutions based on the chosen crossover method.
    """

    if method == "ox":
        return order_crossover(parent1, parent2)
    elif method == "pmx":
        return partially_mapped_crossover(parent1, parent2)
    elif method == "cx":
        return cycle_crossover(parent1, parent2)
    elif method == "pbx":
        return position_based_crossover(parent1, parent2)
    else:
        raise ValueError("Invalid crossover method")
