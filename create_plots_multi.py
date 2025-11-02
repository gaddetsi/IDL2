import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import os, sys
from pathlib import Path

from task2_1_a import load_data
from task2_1_c import h_numerical_cs_mae, m_numerical_cs_mae, print_metrics

def plot_error_histogram(errors, model_name, save_path=None):
    """Plot histogram of prediction errors."""
    plt.figure(figsize=(10, 6))
    plt.hist(errors, bins=50, edgecolor='black', alpha=0.7)
    plt.xlabel('Absolute Error (minutes)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    plt.title(f'{model_name} - Error Distribution', fontsize=14, fontweight='bold')
    plt.axvline(np.mean(errors), color='r', linestyle='--', linewidth=2, label=f'Mean: {np.mean(errors):.1f} min')
    plt.axvline(np.median(errors), color='g', linestyle='--', linewidth=2, label=f'Median: {np.median(errors):.1f} min')
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

def to_decimal(y):
    """Convert split hours and minutes to decimal hours."""
    hours = y[0]
    decimal_hours = hours + (y[1] / 59)
    return decimal_hours

def plot_predictions_vs_true(pred, true, model_name, save_path=None):
    """Scatter plot of predictions vs true values."""
    # Scale back to original values and to decimal hours
    plt.figure(figsize=(10, 10))
    plt.scatter(true, pred, alpha=0.3, s=10)
    plt.plot([0, 12], [0, 12], 'r--', linewidth=2, label='Perfect prediction')
    plt.xlabel('True Time (hours)', fontsize=12)
    plt.ylabel('Predicted Time (hours)', fontsize=12)
    plt.title(f'{model_name} - Predictions vs True Values', fontsize=14, fontweight='bold')
    plt.xlim([0, 12])
    plt.ylim([0, 12])
    plt.legend()
    plt.grid(True, alpha=0.3)
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()


if __name__ == "__main__":
    dir = f"./images/multi"
    os.makedirs(dir, exist_ok=True)

    # load data
    X_train, y_train, X_val, y_val, X_test, y_test = load_data(seed=42)

    # split y_test into hours and mintes arrays
    y_test_split = list()
    y_test_split.append(y_test[:, 0].reshape(-1, 1))
    y_test_split.append(y_test[:, 1].reshape(-1, 1))

    # load model
    for model_path in sys.argv[1:]:
        model = keras.models.load_model(model_path)

        # make predictions
        y_pred = model.predict(X_test)
        
        curr_model = Path(model_path).stem

        metrics = print_metrics(y_pred, y_test_split, curr_model)

        errors = metrics['errors']

        # convert split hours and minutes to decimal hours for plotting
        true_decimal = to_decimal(y_test_split)
        pred_decimal = to_decimal(y_pred)

        # plot error histogram
        plot_error_histogram(errors, model_name=curr_model, save_path=f"{dir}/{curr_model}_error_histogram.png")

        # plot predictions vs true values
        plot_predictions_vs_true(pred_decimal, true_decimal, model_name=curr_model, save_path=f"{dir}/{curr_model}_pred_vs_true.png")