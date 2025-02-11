import numpy as np


def get_distance_matrix(cities):
    """
    Computes the Euclidean distance matrix for the given cities.
    """
    num_of_cities = cities.shape[0]
    dists = np.zeros((num_of_cities, num_of_cities), dtype=int)

    for i in range(num_of_cities):
        for j in range(num_of_cities):
            dists[i, j] = np.linalg.norm(cities[j, 1:] - cities[i, 1:])

    return dists


def find_next_city(
    current_city: int, dists: np.ndarray, visited_cities: set[int]
) -> int:
    """
    Finds the nearest unvisited city to the current city based on the distance matrix.
    Instead of using infinity masking, it directly skips visited cities.
    """
    # Get distances to all cities from the current city
    min_dist = np.inf
    next_city = -1

    for city in range(dists.shape[0]):
        if city not in visited_cities:
            dist = dists[current_city, city]
            if dist < min_dist:
                min_dist = dist
                next_city = city

    return next_city


def nearest_neighbor(first_city: int, dists: np.ndarray, num_cities: int) -> np.ndarray:
    """
    Generate a route using the nearest neighbor heuristic.
    """
    visited_cities = set()  # Track visited cities using a set for faster lookup
    route = np.zeros(
        num_cities + 1, dtype=int
    )  # Store the route including return to start

    # Start from the first city
    current_city = first_city
    route[0] = current_city
    visited_cities.add(current_city)

    # Loop through and find the nearest neighbor for each step
    for i in range(1, num_cities):
        next_city = find_next_city(current_city, dists, visited_cities)
        route[i] = next_city
        visited_cities.add(next_city)
        current_city = next_city

    # Return to the starting city to complete the cycle
    route[-1] = route[0]

    return route


def gen_population(mode: str, population_size: int, cities: np.ndarray) -> np.ndarray:
    """
    Generates the initial population of routes.
    Each route is a random permutation of city indices forming a cycle.
    and the first one is generated using the nearest neighbor heuristic.
    """
    num_cities = cities.shape[0]  # Number of cities
    population = np.empty((population_size, num_cities + 1), dtype=int)

    for i in range(population_size):
        if mode == "nn" and i % 2 and i < num_cities / 2:
            population[i] = nearest_neighbor(i, cities, num_cities)
        else:
            route = np.random.permutation(num_cities)
            population[i, :-1] = route
            population[i, -1] = route[0]

    np.random.shuffle(population)
    return population


def validate_cities(cities_array):
    """
    Validates that each city has a unique ID within the correct range.
    """
    city_ids = cities_array[:, 0].astype(int)
    num_cities = cities_array.shape[0]

    # Check for uniqueness
    if len(set(city_ids)) != num_cities:
        raise ValueError("City IDs are not unique.")

    # Check that IDs are within 0 to num_cities - 1
    if city_ids.min() < 0 or city_ids.max() >= num_cities:
        raise ValueError("City IDs are out of the valid range.")
