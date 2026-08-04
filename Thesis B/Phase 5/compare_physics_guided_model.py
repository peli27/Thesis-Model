"""
Compare the baseline displacement neural network against the
physics-guided neural network.

This script:
1. Loads baseline and physics-guided model metrics.
2. Combines them into one comparison table.
3. Saves comparison results.
4. Generates comparison plots for RMSE, MAE and R².
5. Optionally compares model error against resonance proximity.
"""

from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# ---- File Paths ----

BASE_DIR = Path(__file__).resolve().parent

PROJECT_DIR = BASE_DIR.parent

RESULTS_DIR = PROJECT_DIR / "results"

PLOTS_DIR = PROJECT_DIR / "figures" / "phase_5_plots"

PLOTS_DIR.mkdir(parents=True, exist_ok=True)

BASELINE_METRICS_PATH = (RESULTS_DIR / "displacement_NN_model_metrics.csv")

HYBRID_METRICS_PATH = (RESULTS_DIR / "Physics_Guided_NN" / "physics_guided_NN_model_metrics.csv")

# Do I need prediction files?

baseline_metrics = pd.read_csv(BASELINE_METRICS_PATH)
baseline_metrics["model"] = ("Baseline_Data_Driven_NN")

hybrid_metrics = pd.read_csv(HYBRID_METRICS_PATH)
hybrid_metrics["model"] = ("Physics_Guided_NN")

# Combine metrics into one DataFrame
comparison_metrics = pd.concat([baseline_metrics, hybrid_metrics], axis=0, ignore_index=True)

comparison_output_path = RESULTS_DIR / "baseline_vs_physics_guided_model_comparison.csv"

comparison_metrics.to_csv(comparison_output_path, index=False)

print("\n--- Baseline vs Physics-Guided Model Comparison ---")
print(comparison_metrics)

# ----- Calculate percentage improvement -----

def calculate_percentage_improvement(baseline_value, hybrid_value):
    """
    Calculates percentage improvement from baseline to hybrid.
    """
    return ((baseline_value - hybrid_value) / baseline_value * 100)

improvement_rows = []

for output in comparison_metrics["output"].unique():
    
    baseline_row = comparison_metrics[(comparison_metrics["output"] == output) & (comparison_metrics["model"] == "Baseline_Data_Driven_NN")].iloc[0]
    
    hybrid_row = comparison_metrics[(comparison_metrics["output"] == output) & (comparison_metrics["model"] == "Physics_Guided_NN")].iloc[0]

    rmse_improvement = calculate_percentage_improvement(baseline_value=baseline_row["rmse"], hybrid_value=hybrid_row["rmse"])

    mae_improvement = calculate_percentage_improvement(baseline_value=baseline_row["mae"], hybrid_value=hybrid_row["mae"])

    r2_change = (hybrid_row["r2"] - baseline_row["r2"])

    improvement_rows.append({
        "output": output,
        "rmse_percentage_improvement": rmse_improvement,
        "mae_percentage_improvement": mae_improvement,
        "r2_change": r2_change,
    })

improvement_df = pd.DataFrame(improvement_rows)

improvement_output_path = (RESULTS_DIR / "physics_guided_percentage_improvement.csv")

improvement_df.to_csv(improvement_output_path, index=False)

print("\n--- Percentage Improvement ---")
print(improvement_df)


# ----- Plot metric comparison -----

def plot_metric_comparison(metric_name, ylabel, filename):
    """
    Creates a grouped bar chart comparing baseline and hybrid models.
    """

    pivot_df = comparison_metrics.pivot(index="output", columns="model", values=metric_name)

    ax = pivot_df.plot(kind="bar", figsize=(8, 5))

    ax.set_xlabel("Output")
    ax.set_ylabel(ylabel)
    ax.set_title(f"Baseline vs Physics-Guided Model: {ylabel}")

    ax.legend(title="Model")
    plt.xticks(rotation=0)
    plt.grid(axis="y", alpha=0.3)
    plt.tight_layout()

    plt.savefig(PLOTS_DIR / filename, dpi=300)

    plt.close()


plot_metric_comparison(metric_name="rmse", ylabel="RMSE", filename="baseline_vs_physics_guided_rmse.png")

plot_metric_comparison(metric_name="mae", ylabel="MAE", filename="baseline_vs_physics_guided_mae.png")

plot_metric_comparison(metric_name="r2", ylabel="R²", filename="baseline_vs_physics_guided_r2.png")


# ----- Error vs resonance proximity -----

BASELINE_PREDICTIONS_PATH = (RESULTS_DIR / "displacement_NN_predictions.csv")
HYBRID_PREDICTIONS_PATH = (RESULTS_DIR / "Physics_Guided_NN" / "physics_guided_NN_predictions.csv")
HYBRID_TEST_DATA_PATH = (PROJECT_DIR / "dataset" / "hybrid" / "test_hybrid_dataset.csv")

required_optional_files = [
    BASELINE_PREDICTIONS_PATH,
    HYBRID_PREDICTIONS_PATH,
    HYBRID_TEST_DATA_PATH,
]

baseline_predictions = pd.read_csv(BASELINE_PREDICTIONS_PATH)

hybrid_predictions = pd.read_csv(HYBRID_PREDICTIONS_PATH)

hybrid_test_df = pd.read_csv(HYBRID_TEST_DATA_PATH)

resonance_proximity = hybrid_test_df["resonance_proximity"].values

for output_name in ["peak_displacement", "rms_displacement"]:

    baseline_error = np.abs(
        baseline_predictions[
            f"actual_{output_name}"
        ]
        - baseline_predictions[
            f"predicted_{output_name}"
        ]
        )

    hybrid_error = np.abs(
        hybrid_predictions[
            f"actual_{output_name}"
        ]
        - hybrid_predictions[
            f"predicted_{output_name}"
        ]
    )

    plt.figure(figsize=(8, 5))
    plt.scatter(resonance_proximity, baseline_error, alpha=0.5, label="Baseline Data-Driven NN")
    plt.scatter(resonance_proximity, hybrid_error, alpha=0.5, label="Physics-Guided NN")
    plt.xlabel("Resonance Proximity |1 - frequency ratio|")
    plt.ylabel(f"Absolute Error in {output_name}")
    plt.title(f"Prediction Error vs Resonance Proximity - {output_name}")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / (f"error_vs_resonance_proximity_{output_name}.png"), dpi=300)
    plt.close()