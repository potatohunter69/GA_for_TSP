import numpy as np
import os
from typing import List


def load_cities_name(filename):
    with open(filename, "r") as file:
        cities = file.read().splitlines()
    return np.array(cities)


def load_dataset(directory: str, endswith="_dataset.csv") -> List:
    """
    Loads all dataset files in the given directory and combines them into a single array.
    """
    combined_data = []

    for filename in os.listdir(directory):
        if filename.endswith(endswith):
            file_path = os.path.join(directory, filename)
            data = load_csv(file_path)
            combined_data.append(data)
            print(
                f"Loaded dataset from {file_path} stored at index {len(combined_data) - 1}."
            )

    if not combined_data:
        raise ValueError("No valid dataset files found in the directory.")

    return combined_data


def load_csv(file_path: str) -> np.ndarray:
    try:
        data = np.genfromtxt(
            file_path, delimiter=",", skip_header=1, dtype=int, encoding="utf-8"
        )

        if data.ndim != 2 or data.shape[1] < 3:
            raise ValueError(
                f"Unexpected CSV format at {file_path}: Expected 2D array with at least 3 columns."
            )

    except Exception as e:
        raise ValueError(f"Error loading CSV file at {file_path}: {e}")

    city_ids = data[:, 0].astype(int)
    num_cities = data.shape[0]
    if len(set(city_ids)) != num_cities:
        raise ValueError("City IDs are not unique.")
    if city_ids.min() < 0 or city_ids.max() >= num_cities:
        raise ValueError("City IDs are out of valid range.")
    return data
