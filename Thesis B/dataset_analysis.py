import os
import pandas as pd
import matplotlib.pyplot as plt


def load_dataset(filename="dataset/full_dataset.csv"):
    """
    Loads the full generated dataset.
    """
    df = pd.read_csv(filename)
    return df


def check_dataset_quality(df):
    """
    Checks basic dataset quality.
    """
    print("\n--- Dataset Shape ---")
    print(df.shape)
    print("\n--- Missing Values ---")
    print(df.isnull().sum())
    print("\n--- Duplicate Rows ---")
    print(df.duplicated().sum())
    print("\n--- Column Names ---")
    print(df.columns.tolist())


def print_dataset_statistics(df):
    """
    Prints summary statistics for all variables.
    """
    print("\n--- Dataset Statistics ---")
    print(df.describe())


def plot_histogram(df, column, folder):
    """
    Saves a histogram for one selected column.
    """
    plt.figure(figsize=(8, 5))
    plt.hist(df[column], bins=30)

    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column}")

    plt.grid(True)
    plt.tight_layout()

    plt.savefig(f"{folder}/{column}_distribution.png", dpi=300)
    plt.close()


def plot_input_distributions(df, folder):
    """
    Plots histograms of input variables.
    """
    input_columns = [
        "L",
        "b",
        "h",
        "E",
        "rho",
        "damping_ratio"
    ]

    for column in input_columns:
        plot_histogram(df, column, folder)


def plot_output_distributions(df, folder):
    """
    Plots histograms of output variables.
    """
    output_columns = [
        "f1",
        "f2",
        "f3",
        "f4",
        "peak_displacement",
        "rms_displacement"
    ]

    for column in output_columns:
        plot_histogram(df, column, folder)


def plot_correlation_matrix(df, folder):
    """
    Plots and saves the dataset correlation matrix.
    """
    correlation_matrix = df.corr()

    plt.figure(figsize=(10, 8))
    plt.imshow(correlation_matrix)

    plt.colorbar(label="Correlation Coefficient")

    plt.xticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns,
        rotation=90
    )

    plt.yticks(
        range(len(correlation_matrix.columns)),
        correlation_matrix.columns
    )

    plt.title("Dataset Correlation Matrix")
    plt.tight_layout()

    plt.savefig(f"{folder}/correlation_matrix.png", dpi=300)
    plt.close()

    print("\n--- Correlation Matrix ---")
    print(correlation_matrix)


if __name__ == "__main__":

    os.makedirs("figures/dataset_analysis", exist_ok=True)

    df = load_dataset("dataset/full_dataset.csv")

    check_dataset_quality(df)

    print_dataset_statistics(df)

    # plot_input_distributions(df, "figures/dataset_analysis")

    # plot_output_distributions(df, "figures/dataset_analysis")

    plot_correlation_matrix(df, "figures/dataset_analysis")

    # print("\nDataset analysis complete.")