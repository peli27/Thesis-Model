import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.multioutput import MultiOutputRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


# ----- File paths -----

TRAIN_DATASET_PATH = "dataset/train_dataset.csv"
VALIDATION_DATASET_PATH = "dataset/validation_dataset.csv"
TEST_DATASET_PATH = "dataset/test_dataset.csv"

RESULTS_FOLDER = "results"
FIGURES_FOLDER = "figures/baseline_models"

PREDICTED_VS_ACTUAL_FOLDER = f"{FIGURES_FOLDER}/predicted_vs_actual"
RESIDUALS_FOLDER = f"{FIGURES_FOLDER}/residuals"

# ----- Inputs and Outputs -----

INPUT_COLUMNS = [
    "L",
    "b",
    "h",
    "E",
    "rho",
    "damping_ratio",
    "force_amplitude",
    "excitation_frequency"
]

OUTPUT_COLUMNS = [
    "f1",
    "f2",
    "f3",
    "f4",
    "peak_displacement",
    "rms_displacement"
]

# ----- Define functions -----

def create_folders():
    """
    Creates folders for saving results and figures.
    """
    os.makedirs(RESULTS_FOLDER, exist_ok=True)
    os.makedirs(PREDICTED_VS_ACTUAL_FOLDER, exist_ok=True)
    os.makedirs(RESIDUALS_FOLDER, exist_ok=True)


def load_datasets():
    """
    Loads train, validation and test datasets.
    """
    train_df = pd.read_csv(TRAIN_DATASET_PATH)
    validation_df = pd.read_csv(VALIDATION_DATASET_PATH)
    test_df = pd.read_csv(TEST_DATASET_PATH)

    return train_df, validation_df, test_df


def prepare_data(train_df, validation_df, test_df):
    """
    Separates input and output columns.
    The training and validation datasets are combined for final baseline model training.
    """
    combined_train_df = pd.concat([train_df, validation_df], ignore_index=True)

    X_train = combined_train_df[INPUT_COLUMNS]
    y_train = combined_train_df[OUTPUT_COLUMNS]

    X_test = test_df[INPUT_COLUMNS]
    y_test = test_df[OUTPUT_COLUMNS]

    return X_train, y_train, X_test, y_test


def calculate_metrics(y_true, y_pred, model_name, training_time, prediction_time):
    """
    Calculates RMSE, MAE and R2 for each output variable.
    Also calculates average model performance across all outputs.
    """
    results = []

    for i, output_name in enumerate(OUTPUT_COLUMNS):
        y_true_output = y_true.iloc[:, i]
        y_pred_output = y_pred[:, i]

        rmse = np.sqrt(mean_squared_error(y_true_output, y_pred_output))
        mae = mean_absolute_error(y_true_output, y_pred_output)
        r2 = r2_score(y_true_output, y_pred_output)

        results.append({
            "model": model_name,
            "output": output_name,
            "rmse": rmse,
            "mae": mae,
            "r2": r2,
            "training_time_seconds": training_time,
            "prediction_time_seconds": prediction_time
        })

    average_rmse = np.mean([row["rmse"] for row in results])
    average_mae = np.mean([row["mae"] for row in results])
    average_r2 = np.mean([row["r2"] for row in results])

    results.append({
        "model": model_name,
        "output": "average",
        "rmse": average_rmse,
        "mae": average_mae,
        "r2": average_r2,
        "training_time_seconds": training_time,
        "prediction_time_seconds": prediction_time
    })

    return results


def plot_predicted_vs_actual(y_true, y_pred, model_name):
    """
    Saves predicted vs actual plots for each output variable.
    """
    for i, output_name in enumerate(OUTPUT_COLUMNS):
        actual = y_true.iloc[:, i]
        predicted = y_pred[:, i]

        plt.figure(figsize=(7, 6))
        plt.scatter(actual, predicted, alpha=0.6)

        min_value = min(actual.min(), predicted.min())
        max_value = max(actual.max(), predicted.max())

        plt.plot([min_value, max_value], [min_value, max_value], linestyle="--")

        plt.xlabel("Actual value")
        plt.ylabel("Predicted value")
        plt.title(f"{model_name}: Predicted vs Actual - {output_name}")

        plt.grid(True)
        plt.tight_layout()

        filename = f"{PREDICTED_VS_ACTUAL_FOLDER}/{model_name}_{output_name}_predicted_vs_actual.png"
        plt.savefig(filename, dpi=300)
        plt.close()


def plot_residuals(y_true, y_pred, model_name):
    """
    Saves residual plots for each output variable.
    Residual = actual - predicted.
    """
    for i, output_name in enumerate(OUTPUT_COLUMNS):
        actual = y_true.iloc[:, i]
        predicted = y_pred[:, i]
        residuals = actual - predicted

        plt.figure(figsize=(7, 6))
        plt.scatter(predicted, residuals, alpha=0.6)

        plt.axhline(0, linestyle="--")

        plt.xlabel("Predicted value")
        plt.ylabel("Residual error")
        plt.title(f"{model_name}: Residual Plot - {output_name}")

        plt.grid(True)
        plt.tight_layout()

        filename = f"{RESIDUALS_FOLDER}/{model_name}_{output_name}_residuals.png"
        plt.savefig(filename, dpi=300)
        plt.close()


def train_and_evaluate_model(model, model_name, X_train, y_train, X_test, y_test):
    """
    Trains one model, evaluates it and saves plots.
    """
    print(f"\nTraining {model_name}...")

    start_train_time = time.perf_counter()
    model.fit(X_train, y_train)
    end_train_time = time.perf_counter()

    training_time = end_train_time - start_train_time

    start_prediction_time = time.perf_counter()
    y_pred = model.predict(X_test)
    end_prediction_time = time.perf_counter()

    prediction_time = end_prediction_time - start_prediction_time

    metrics = calculate_metrics(
        y_true=y_test,
        y_pred=y_pred,
        model_name=model_name,
        training_time=training_time,
        prediction_time=prediction_time
    )

    plot_predicted_vs_actual(y_test, y_pred, model_name)
    plot_residuals(y_test, y_pred, model_name)

    print(f"{model_name} complete.")
    print(f"Training time: {training_time:.4f} seconds")
    print(f"Prediction time: {prediction_time:.4f} seconds")

    return metrics

# ----- Main -----
if __name__ == "__main__":

    create_folders()

    train_df, validation_df, test_df = load_datasets()

    print("\n--- Dataset Shapes ---")
    print(f"Training dataset: {train_df.shape}")
    print(f"Validation dataset: {validation_df.shape}")
    print(f"Test dataset: {test_df.shape}")

    X_train, y_train, X_test, y_test = prepare_data(
        train_df,
        validation_df,
        test_df
    )


    models = {
        "Linear_Regression": Pipeline([
            ("scaler", StandardScaler()),
            ("model", LinearRegression())
        ]),

        "Random_Forest": RandomForestRegressor(
            n_estimators=100,
            random_state=42,
            n_jobs=-1
        ),

        "Gradient_Boosting": MultiOutputRegressor(
            GradientBoostingRegressor(
                n_estimators=100,
                learning_rate=0.1,
                max_depth=3,
                random_state=42
            )
        )
    }

    all_metrics = []

    for model_name, model in models.items():
        model_metrics = train_and_evaluate_model(
            model=model,
            model_name=model_name,
            X_train=X_train,
            y_train=y_train,
            X_test=X_test,
            y_test=y_test
        )

        all_metrics.extend(model_metrics)

    metrics_df = pd.DataFrame(all_metrics)

    metrics_output_path = f"{RESULTS_FOLDER}/baseline_model_metrics.csv"
    metrics_df.to_csv(metrics_output_path, index=False)

    print("\n--- Baseline Model Metrics ---")
    print(metrics_df)

    print(f"\nSaved metrics to: {metrics_output_path}")
    print("\nBaseline model training complete.")