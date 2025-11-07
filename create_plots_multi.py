# EXAMPLE USAGE: python ./create_plots_multi.py ./saved_models/multi_*

import matplotlib.pyplot as plt
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import os, sys
from pathlib import Path

from task2_1_a import load_data
from task2_1_c import h_numerical_cs_mse, m_numerical_cs_mse, common_sense_mse_cr

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

def sin_cos_to_hours(sin_cos_predictions):
    """Convert [cos, sin] predictions back to hours (0-12)."""
    cos_vals = sin_cos_predictions[:, 0]
    sin_vals = sin_cos_predictions[:, 1]
    angles = np.arctan2(sin_vals, cos_vals)  # arctan2(sin, cos)
    angles = (angles % (2 * np.pi))  # ensure positive angles
    hours = angles * 12 / (2 * np.pi)  # convert to hours
    return hours

def sin_cos_to_minutes(sin_cos_predictions):
    """Convert [cos, sin] predictions back to minutes (0-59)."""
    cos_vals = sin_cos_predictions[:, 0]
    sin_vals = sin_cos_predictions[:, 1]
    angles = np.arctan2(sin_vals, cos_vals)  # arctan2(sin, cos)
    angles = (angles % (2 * np.pi))  # ensure positive angles
    minutes = angles * 59 / (2 * np.pi)  # convert to minutes
    return minutes

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
    
def get_hours_minutes(hours, minutes):
    """Convert one-hot or sin-cos encoded hours and minutes to numerical hours and minutes."""
    hours_ = hours
    if hours_.shape[1] == 12:
        hours_ = np.argmax(hours_, axis=1).reshape(-1, 1)
    elif hours_.shape[1] == 2:
        hours_ = np.floor(sin_cos_to_hours(hours_).reshape(-1, 1))

    minutes_ = minutes
    if minutes_.shape[1] == 2:
        minutes_ = sin_cos_to_minutes(minutes_).reshape(-1, 1)

    return hours_, minutes_

def to_decimal(y):
    """Convert split hours and minutes to decimal hours."""
    hours = y[0]
    minutes = y[1]
    hours, minutes = get_hours_minutes(hours, minutes)

    decimal_hours = hours + (minutes / 60)
    return decimal_hours

def split_to_diff_min(pred_time, true_time):
    """Calculate absolute difference in minutes between predicted and true times."""
    # if hours are one-hot encoded, convert to numerical
    hours: np.ndarray = pred_time[0]
    minutes: np.ndarray = pred_time[1]
    true_hours: np.ndarray = true_time[0]
    true_minutes: np.ndarray = true_time[1]
    hours, minutes = get_hours_minutes(hours, minutes)
    true_hours, true_minutes = get_hours_minutes(true_hours, true_minutes)

    # convert time to total minutes
    pred_total_min = (hours * 60) + minutes
    true_total_min = (true_hours * 60) + true_minutes

    # common sense loss in minutes
    diff_min = np.abs(pred_total_min - true_total_min)
    csl = np.minimum(diff_min, 720 - diff_min)

    return csl

def print_metrics(pred_time, true_time, model_name):
    """Print comprehensive evaluation metrics."""
    diff_min = split_to_diff_min(pred_time, true_time)

    mean_err = np.mean(diff_min)
    median_err = np.median(diff_min)
    std_err = np.std(diff_min)
    max_err = np.max(diff_min)
    
    within_0 = np.mean(diff_min <= 0) * 100
    within_1 = np.mean(diff_min <= 1) * 100
    within_5 = np.mean(diff_min <= 5) * 100
    within_10 = np.mean(diff_min <= 10) * 100
    within_15 = np.mean(diff_min <= 15) * 100
    within_30 = np.mean(diff_min <= 30) * 100
    
    print(f"\n{'=' * 80}")
    print(f"{model_name} - TEST SET RESULTS")
    print(f"{'=' * 80}")
    print(f"Mean Absolute Error:    {mean_err:.2f} minutes")
    print(f"Median Absolute Error:  {median_err:.2f} minutes")
    print(f"Std Deviation:          {std_err:.2f} minutes")
    print(f"Max Error:              {max_err:.2f} minutes")
    print(f"\nAccuracy within thresholds:")
    print(f"  Within 0 minutes:     {within_0:.1f}%")
    print(f"  Within 1 minute:      {within_1:.1f}%")
    print(f"  Within 5 minutes:     {within_5:.1f}%")
    print(f"  Within 10 minutes:    {within_10:.1f}%")
    print(f"  Within 15 minutes:    {within_15:.1f}%")
    print(f"  Within 30 minutes:    {within_30:.1f}%")
    
    return {
        'mean': mean_err,
        'median': median_err,
        'std': std_err,
        'max': max_err,
        'within_0': within_0,
        'within_1': within_1,
        'within_5': within_5,
        'within_10': within_10,
        'within_15': within_15,
        'within_30': within_30,
        'predictions': pred_time,
        'errors': diff_min
    }



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