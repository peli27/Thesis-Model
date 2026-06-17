import numpy as np
import csv
import os

from beam_properties import get_beam_properties
from frequency_analysis import natural_frequencies
from multi_mode_response import damped_free_response, reconstruct_displacement_at_point


def generate_random_beam():
    """
    Randomly generate one beam configuration.
    """

    beam = get_beam_properties()

    L = np.random.uniform(1.0, 3.0)          # m
    b = np.random.uniform(0.02, 0.08)        # m
    h = np.random.uniform(0.005, 0.02)       # m
    E = np.random.uniform(180e9, 220e9)      # Pa
    rho = np.random.uniform(7600, 8000)      # kg/m^3
    damping_ratio = np.random.uniform(0.005, 0.05)

    beam["L"] = L
    beam["b"] = b
    beam["h"] = h
    beam["E"] = E
    beam["rho"] = rho

    beam["A"] = beam["b"] * beam["h"]
    beam["I"] = beam["b"] * beam["h"]**3 / 12

    return beam, damping_ratio


def simulate_beam_sample():
    """
    Runs one beam simulation and returns one dataset row.
    """

    beam, damping_ratio = generate_random_beam()

    frequencies = natural_frequencies(beam, modes=4)

    t = np.linspace(0, 2, 5000)

    amplitudes = [0.001, 0.0004, 0.0002, 0.0001]

    modal_responses = damped_free_response(
        frequencies,
        amplitudes,
        damping_ratio,
        t
    )

    x_position = 0.5

    midpoint_displacement = reconstruct_displacement_at_point(
        modal_responses,
        x_position
    )

    peak_displacement = np.max(np.abs(midpoint_displacement))
    rms_displacement = np.sqrt(np.mean(midpoint_displacement**2))

    sample = {
        "L": float(beam["L"]),
        "b": float(beam["b"]),
        "h": float(beam["h"]),
        "E": float(beam["E"]),
        "rho": float(beam["rho"]),
        "damping_ratio": float(damping_ratio),

        "f1": float(frequencies[0]),
        "f2": float(frequencies[1]),
        "f3": float(frequencies[2]),
        "f4": float(frequencies[3]),

        "peak_displacement": float(peak_displacement),
        "rms_displacement": float(rms_displacement)
    }

    return sample


def generate_dataset(number_of_samples=5000):
    """
    Generates the full synthetic dataset.
    """

    data = []

    for i in range(number_of_samples):
        sample = simulate_beam_sample()
        data.append(sample)

        if (i + 1) % 500 == 0:
            print(f"Generated {i + 1} samples")

    return data


def split_dataset(dataset):
    """
    Splits dataset into training, validation and test sets.
    """

    np.random.seed(42)
    np.random.shuffle(dataset)

    n = len(dataset)

    train_end = int(0.70 * n)
    validation_end = int(0.85 * n)

    train_data = dataset[:train_end]
    validation_data = dataset[train_end:validation_end]
    test_data = dataset[validation_end:]

    return train_data, validation_data, test_data


def save_dataset_csv(data, filename):
    """
    Saves dataset to CSV.
    """

    columns = [
        "L", "b", "h", "E", "rho", "damping_ratio",
        "f1", "f2", "f3", "f4",
        "peak_displacement", "rms_displacement"
    ]

    with open(filename, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=columns)

        writer.writeheader()
        writer.writerows(data)


if __name__ == "__main__":

    os.makedirs("dataset", exist_ok=True)

    dataset = generate_dataset(number_of_samples=5000)

    train_data, validation_data, test_data = split_dataset(dataset)

    save_dataset_csv(dataset, "dataset/full_dataset.csv")
    save_dataset_csv(train_data, "dataset/train_dataset.csv")
    save_dataset_csv(validation_data, "dataset/validation_dataset.csv")
    save_dataset_csv(test_data, "dataset/test_dataset.csv")

    print("Dataset generation complete.")
    print(f"Full dataset: {len(dataset)} samples")
    print(f"Training dataset: {len(train_data)} samples")
    print(f"Validation dataset: {len(validation_data)} samples")
    print(f"Test dataset: {len(test_data)} samples")