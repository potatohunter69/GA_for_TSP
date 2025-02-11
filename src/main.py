import numpy as np
from genetics.genetics import run_genetic_algorithm
from genetics.parameters import Params, generate_param_grid
from tools import load_csv, plot_route
from tools.log import timing
from tuning import search_grid


if __name__ == "__main__":
    cities = load_csv("../data/cities_50_dataset.csv")

    @timing("run_ga")
    def run_ga():
        np.random.seed(42)

        # Parameters
        params = Params(
            population_size=700,
            generations=1200,
            elite_size=20,
            tournament_size=20,
            mutation_rate=0.03,
            mutation_type="swap",
            crossover_type="ox",
        )

        best_route, best_fitness, rh, fh = run_genetic_algorithm(cities, params)
        plot_route(best_route, cities)
        print(f"Best route: {best_route}")
        print(f"Best fitness: {best_fitness}")

    @timing("grid_search")
    def grid_search():
        param_grid = generate_param_grid(
            population_sizes=[50, 30],
            generations_list=[500],
            elite_sizes=[5],
            tournament_sizes=[10],
            mutation_rates=[0.05],
        )
        results = search_grid(
            cities, param_grid, run_genetic_algorithm, multithreading=False
        )
        print(f"Results: {len(results)}")

    run_ga()
