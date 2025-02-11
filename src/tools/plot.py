from typing import List, Callable

import matplotlib.pyplot as plt
import numpy as np

from tuning import Result


def plot_route(route, cities, eu_countries=None, title="Route"):
    """
    Plots a single route.
    """
    plt.figure(figsize=(10, 8))

    # Extract x and y coordinates for all cities in the route
    x_coords = [cities[city_idx][1] for city_idx in route]
    y_coords = [cities[city_idx][2] for city_idx in route]

    # Plot the cities
    plt.scatter(x_coords, y_coords, c="blue")

    # Annotate each city with its name
    if eu_countries is not None:
        for city_idx in route:
            plt.text(
                cities[city_idx][1],
                cities[city_idx][2],
                eu_countries[int(cities[city_idx][0])],
                fontsize=9,
            )

    # Plot the path
    plt.plot(x_coords, y_coords, "r-", linewidth=1)

    plt.title(title)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)
    plt.show()


def plot_routes(
    routes: np.ndarray, cities: np.array, eu_countries: np.array = None, title="Routes"
):
    """
    Plots all routes and prints the city names.
    """
    plt.figure(figsize=(10, 8))

    for i, route in enumerate(routes):
        x_coords = [cities[city_idx][1] for city_idx in route]
        y_coords = [cities[city_idx][2] for city_idx in route]

        if i == 0:
            plt.scatter(x_coords, y_coords, c="blue")
            plt.plot(x_coords, y_coords, "r-", linewidth=2, label=f"Route {i+1} (Best)")
        else:
            plt.plot(x_coords, y_coords, "grey", linewidth=1, alpha=0.7)

    if eu_countries is not None:
        for city_idx in routes[0]:
            plt.text(
                cities[city_idx][1],
                cities[city_idx][2],
                eu_countries[int(cities[city_idx][0])],
                fontsize=9,
            )

    plt.title(title, fontsize=16)
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.grid(True)

    plt.legend()
    plt.show()


def plot_fitness(fitness_values, title="Best Fitness Over Generations"):
    """
    Plots the fitness values over generations.
    """
    plt.figure(figsize=(10, 6))
    plt.plot(fitness_values, "b-", linewidth=2)
    plt.title(title)
    plt.xlabel("Generation")
    plt.ylabel("Fitness (1/Total Distance)")
    plt.grid(True)
    plt.show()


import matplotlib.pyplot as plt


def plot_gridsearch_result(
    results: List[Result],
    param_name="Parameter",
    title="Grid Search Results",
):
    """
    Plots the results of a grid search. The X-axis parameter is chosen using the selector function.
    """
    fig, axs = plt.subplots(2, 4, figsize=(20, 10))

    fitness_scores = [result.fitness for result in results]
    durations = [result.duration for result in results]
    population_sizes = [result.params.population_size for result in results]
    generations = [result.params.generations for result in results]
    elite_sizes = [result.params.elite_size for result in results]
    tournament_sizes = [result.params.tournament_size for result in results]
    mutation_rates = [result.params.mutation_rate for result in results]

    def plot_axs(ax, x_values, y_values, titlelabel):
        ax.scatter(x_values, y_values, c=fitness_scores, cmap="viridis", alpha=0.7)
        ax.set_title(titlelabel)
        ax.set_xlabel(titlelabel)
        ax.set_ylabel(titlelabel.split(" ")[0])

    plot_axs(
        axs[0, 0], population_sizes, fitness_scores, "Population_Size over fitness"
    )
    plot_axs(axs[0, 1], generations, fitness_scores, "Generations over fitness")
    plot_axs(axs[0, 2], elite_sizes, fitness_scores, "Elite_Size over fitness")
    plot_axs(
        axs[0, 3], tournament_sizes, fitness_scores, "Tournament_Size over fitness"
    )
    plot_axs(axs[1, 0], mutation_rates, fitness_scores, "Mutation_Rate over fitness")
    plot_axs(axs[1, 1], durations, fitness_scores, "Duration over fitness")
    plot_axs(axs[1, 2], population_sizes, durations, "Population Size over duration")
    plot_axs(axs[1, 3], generations, durations, "Generations over duration")

    plt.suptitle(title)
    plt.tight_layout()
    plt.show()


def plot_comparison(df1, df2, species, metrics):
    fig, axs = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    plt.subplots_adjust(wspace=0.3)

    x = np.arange(len(species))
    width = 0.3

    for i, metric in enumerate(metrics):
        penguin_means = {
            "Initial GA": df1[metric],
            "Final GA": df2[metric],
        }
        multiplier = 0

        for attribute, measurement in penguin_means.items():
            offset = width * multiplier
            rects = axs[i].bar(x + offset, measurement, width, label=attribute)
            # axs[i].bar_label(rects, padding=3)
            multiplier += 1

        axs[i].set_ylabel(metric.capitalize())
        axs[i].set_title(f"{metric.capitalize()} Comparison")

    axs[0].set_xticks(x + width / 2, species)
    plt.xlabel("Data Sizes")
    fig.suptitle("Comparison of Initial GA and Final GA Performance")

    plt.legend(loc="center", bbox_to_anchor=(1.05, 1), ncols=2)

    plt.show()
