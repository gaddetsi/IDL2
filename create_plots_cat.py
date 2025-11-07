# EXAMPLE USAGE: python ./create_plots_cat.py ./saved_models/loss_*

import matplotlib.pyplot as plt
import numpy as np
from tensorflow import keras
import os, sys
from pathlib import Path

from task2_1_a import preprocess_cat, print_metrics, to_categorical

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


def plot_predictions_vs_true(pred_classes, true_classes, model_name, save_path=None):
    """Scatter plot of predictions vs true values."""
    plt.figure(figsize=(10, 10))
    plt.scatter(true_classes/(24/12), pred_classes/(24/12), alpha=0.3, s=10)
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
    # create output directory if it doesn't exist
    dir = f"./images/categorical"
    os.makedirs(dir, exist_ok=True)
    # load model
    for model_path in sys.argv[1:]:
        # select parameters
        curr_model = Path(model_path).stem
        if curr_model.endswith("hard"):
            num_classes = 720
            easy = False
        else:
            num_classes = 24
            easy = False

        # load preprocessed data
        X_train, y_train, X_val, y_val, X_test, y_test, input_shape = preprocess_cat(easy,num_classes)
        # convert y_test to categorical labels
        y_test = to_categorical(y_test, num_classes)
        # load model
        model = keras.models.load_model(model_path)
        # make predictions
        y_pred = model.predict(X_test)
        # print metrics
        metrics = print_metrics(y_test, y_pred, curr_model, num_classes)
        # get common sense loss
        errors = metrics['errors']

        # plot error histogram
        plot_error_histogram(errors, model_name=curr_model, save_path=f"{dir}/{curr_model}_error_histogram.png")

        # plot predictions vs true values
        plot_predictions_vs_true(np.argmax(y_pred, axis=1), np.argmax(y_test, axis=1), model_name=curr_model, save_path=f"{dir}/{curr_model}_pred_vs_true.png")