import numpy as np
import matplotlib.pyplot as plt
import os
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.model_selection import train_test_split

# Create images directory for saving plots
os.makedirs("images", exist_ok=True)

# Set random seeds 
np.random.seed(42)
keras.utils.set_random_seed(42)


print("=" * 80)
print("LOADING DATA")
print("=" * 80)

X = np.load("data/A1_data_150/images.npy").astype("float32") / 255.0
y = np.load("data/A1_data_150/labels.npy")  

print(f"Images shape: {X.shape}")
print(f"Labels shape: {y.shape}")
print(f"Hour range: [{y[:,0].min()}, {y[:,0].max()}]")
print(f"Minute range: [{y[:,1].min()}, {y[:,1].max()}]")

# Split into train/val/test (80/10/10)
X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.10, random_state=42, shuffle=True
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full, test_size=0.1111, random_state=42, shuffle=True
)

# Add channel dimension for CNN
X_train = X_train[..., np.newaxis]
X_val = X_val[..., np.newaxis]
X_test = X_test[..., np.newaxis]

print(f"\nTrain: {X_train.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
print(f"Train labels: {y_train.shape}, Val labels: {y_val.shape}, Test labels: {y_test.shape}")

#HELPER FUNCTIONS

def labels_to_hours_float(y):
    """Convert (hour, minute) → single float hour value (0-12)."""
    return y[:, 0] + y[:, 1] / 60.0


def circular_hours_diff_to_minutes(pred_hours, true_hours):
    """
    Compute smallest circular difference in minutes.
    Handles the wrap-around at 12 hours.
    """
    diff = np.abs(pred_hours - true_hours)
    diff = np.minimum(diff, 12 - diff)  # circular distance
    return diff * 60


def labels_to_sin_cos(y):
    """
    Convert (hour, minute) → [cos, sin] of clock angle.
    This encoding handles the circular nature of time.
    """
    hours_float = y[:, 0] + y[:, 1] / 60.0
    angle = 2 * np.pi * hours_float / 12.0  # full rotation = 12 hours
    y_cos = np.cos(angle)
    y_sin = np.sin(angle)
    return np.stack([y_cos, y_sin], axis=1)  # shape: (N, 2)


def sin_cos_to_hours(sin_cos_predictions):
    """Convert [cos, sin] predictions back to hours (0-12)."""
    cos_vals = sin_cos_predictions[:, 0]
    sin_vals = sin_cos_predictions[:, 1]
    angles = np.arctan2(sin_vals, cos_vals)  # arctan2(sin, cos)
    angles = (angles % (2 * np.pi))  # ensure positive angles
    hours = angles * 12 / (2 * np.pi)  # convert to hours
    return hours


def print_metrics(pred_hours, true_hours, model_name):
    """Print comprehensive evaluation metrics."""
    diff_min = circular_hours_diff_to_minutes(pred_hours, true_hours)
    
    mean_err = np.mean(diff_min)
    median_err = np.median(diff_min)
    std_err = np.std(diff_min)
    max_err = np.max(diff_min)
    
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
    print(f"  Within 5 minutes:     {within_5:.1f}%")
    print(f"  Within 10 minutes:    {within_10:.1f}%")
    print(f"  Within 15 minutes:    {within_15:.1f}%")
    print(f"  Within 30 minutes:    {within_30:.1f}%")
    
    return {
        'mean': mean_err,
        'median': median_err,
        'std': std_err,
        'max': max_err,
        'within_5': within_5,
        'within_10': within_10,
        'within_15': within_15,
        'within_30': within_30,
        'predictions': pred_hours,
        'errors': diff_min
    }


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


def plot_predictions_vs_true(pred_hours, true_hours, model_name, save_path=None):
    """Scatter plot of predictions vs true values."""
    plt.figure(figsize=(10, 10))
    plt.scatter(true_hours, pred_hours, alpha=0.3, s=10)
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



# BUILD CNN MODELS

def build_cnn_regression(input_shape):
    """CNN for plain regression (predicting hours as float)."""
    inputs = keras.Input(shape=input_shape)
    
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(256, activation='relu')(x)
    x = layers.Dropout(0.5)(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(1, activation='linear')(x)
    
    return keras.Model(inputs, outputs, name="cnn_regression")


def build_cnn_sin_cos(input_shape):
    """CNN for periodic regression (predicting sin/cos of time angle)."""
    inputs = keras.Input(shape=input_shape)
    
    # First conv block
    x = layers.Conv2D(32, (3, 3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(32, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Second conv block
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Conv2D(64, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.2)(x)
    
    # Third conv block
    x = layers.Conv2D(128, (3, 3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.MaxPooling2D((2, 2))(x)
    x = layers.Dropout(0.3)(x)
    
    # Dense layers
    x = layers.Flatten()(x)
    x = layers.Dense(256)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(128)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('relu')(x)
    x = layers.Dropout(0.3)(x)
    
    # Output layer: 2 nodes for cos and sin, bounded by tanh
    outputs = layers.Dense(2, activation='tanh')(x)
    
    return keras.Model(inputs, outputs, name="cnn_periodic")



#TRAIN PLAIN REGRESSION MODEL

print("\n" + "=" * 80)
print("TASK 2.B: PLAIN REGRESSION MODEL")
print("=" * 80)

# Prepare labels for plain regression
y_train_reg = labels_to_hours_float(y_train)
y_val_reg = labels_to_hours_float(y_val)
y_test_reg = labels_to_hours_float(y_test)

print(f"\nRegression labels - Train: {y_train_reg.shape}, Val: {y_val_reg.shape}, Test: {y_test_reg.shape}")
print(f"Label range: [{y_train_reg.min():.2f}, {y_train_reg.max():.2f}] hours")

# Build and compile model
model_reg = build_cnn_regression(X_train.shape[1:])
model_reg.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss='mse',
    metrics=['mae']
)
model_reg.summary()

# Callbacks
callbacks_reg = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', patience=5, factor=0.5, verbose=1, min_lr=1e-7
    )
]

# Train
print("\nTraining plain regression model...")
history_reg = model_reg.fit(
    X_train, y_train_reg,
    validation_data=(X_val, y_val_reg),
    epochs=50,
    batch_size=128,
    verbose=1,
    callbacks=callbacks_reg
)

# Evaluate
print("\nEvaluating plain regression model on test set...")
preds_reg = model_reg.predict(X_test, verbose=0).flatten()
preds_reg_mod = np.mod(preds_reg, 12.0) # wrap around at 12 hours

metrics_reg = print_metrics(preds_reg_mod, y_test_reg, "PLAIN REGRESSION")

# Visualizations
plot_error_histogram(metrics_reg['errors'], "Plain Regression", "images/plain_reg_errors.png")
plot_predictions_vs_true(preds_reg_mod, y_test_reg, "Plain Regression", "images/plain_reg_scatter.png")



#Build and Train PERIODIC (SIN/COS) REGRESSION MODEL


print("\n" + "=" * 80)
print("TASK 2.D: PERIODIC (SIN/COS) REGRESSION MODEL")
print("=" * 80)

# Prepare labels for periodic regression
y_train_sc = labels_to_sin_cos(y_train)
y_val_sc = labels_to_sin_cos(y_val)
y_test_sc = labels_to_sin_cos(y_test)

print(f"\nPeriodic labels - Train: {y_train_sc.shape}, Val: {y_val_sc.shape}, Test: {y_test_sc.shape}")
print(f"Cos range: [{y_train_sc[:,0].min():.2f}, {y_train_sc[:,0].max():.2f}]")
print(f"Sin range: [{y_train_sc[:,1].min():.2f}, {y_train_sc[:,1].max():.2f}]")

# VERIFY ENCODING IS CORRECT
print("\n" + "=" * 80)
print("VERIFYING SIN/COS ENCODING (first 5 samples):")
print("=" * 80)
for i in range(5):
    h, m = y_train[i]
    cos_val, sin_val = y_train_sc[i]
    angle = np.arctan2(sin_val, cos_val)
    recovered_hours = (angle % (2*np.pi)) * 12 / (2*np.pi)
    original_hours = h + m/60.0
    print(f"Original: {h:02d}:{m:02d} ({original_hours:.2f}h) -> cos={cos_val:.3f}, sin={sin_val:.3f} -> Recovered: {recovered_hours:.2f}h")

# Build and compile model
model_sc = build_cnn_sin_cos(X_train.shape[1:])
model_sc.compile(
    optimizer=keras.optimizers.Adam(learning_rate=3e-4),  
    loss='mse',
    metrics=['mae']
)
model_sc.summary()

# Callbacks
callbacks_sc = [
    keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', patience=8, factor=0.5, verbose=1, min_lr=1e-7
    ),
    keras.callbacks.ModelCheckpoint(
        'best_periodic_model.keras', 
        monitor='val_loss', 
        save_best_only=True, 
        verbose=1
    )
]

# Train periodic regression model
print("\nTraining periodic regression model...")
history_sc = model_sc.fit(
    X_train, y_train_sc,
    validation_data=(X_val, y_val_sc),
    epochs=60,
    batch_size=128,
    verbose=1,
    callbacks=callbacks_sc
)

# Load best model
model_sc = keras.models.load_model('best_periodic_model.keras')

# Evaluate
print("\nEvaluating periodic regression model on test set...")
pred_sc = model_sc.predict(X_test, verbose=0)

# Check prediction statistics
print(f"\nPrediction statistics:")
print(f"  Cos predictions - mean: {pred_sc[:,0].mean():.3f}, std: {pred_sc[:,0].std():.3f}, range: [{pred_sc[:,0].min():.3f}, {pred_sc[:,0].max():.3f}]")
print(f"  Sin predictions - mean: {pred_sc[:,1].mean():.3f}, std: {pred_sc[:,1].std():.3f}, range: [{pred_sc[:,1].min():.3f}, {pred_sc[:,1].max():.3f}]")

pred_hours_sc = sin_cos_to_hours(pred_sc)
true_hours_sc = sin_cos_to_hours(y_test_sc)

metrics_sc = print_metrics(pred_hours_sc, true_hours_sc, "PERIODIC REGRESSION")

# Visualizations
plot_error_histogram(metrics_sc['errors'], "Periodic Regression", "images/periodic_reg_errors.png")
plot_predictions_vs_true(pred_hours_sc, true_hours_sc, "Periodic Regression", "images/periodic_reg_scatter.png")



# 6. COMPARISON

print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

print("\n{:<30} {:<20} {:<20}".format("Metric", "Plain Regression", "Periodic Regression"))
print("-" * 80)
print("{:<30} {:<20.2f} {:<20.2f}".format("Mean Error (min)", metrics_reg['mean'], metrics_sc['mean']))
print("{:<30} {:<20.2f} {:<20.2f}".format("Median Error (min)", metrics_reg['median'], metrics_sc['median']))
print("{:<30} {:<20.2f} {:<20.2f}".format("Std Dev (min)", metrics_reg['std'], metrics_sc['std']))
print("{:<30} {:<20.2f} {:<20.2f}".format("Max Error (min)", metrics_reg['max'], metrics_sc['max']))
print("{:<30} {:<20.1f} {:<20.1f}".format("Within 10 min (%)", metrics_reg['within_10'], metrics_sc['within_10']))
print("{:<30} {:<20.1f} {:<20.1f}".format("Within 15 min (%)", metrics_reg['within_15'], metrics_sc['within_15']))

improvement = ((metrics_reg['mean'] - metrics_sc['mean']) / metrics_reg['mean']) * 100
print(f"\nImprovement: {improvement:.1f}%")

# Plot training history comparison
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

axes[0].plot(history_reg.history['loss'], label='Train')
axes[0].plot(history_reg.history['val_loss'], label='Validation')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss (MSE)')
axes[0].set_title('Plain Regression - Training History')
axes[0].legend()
axes[0].grid(True, alpha=0.3)
axes[0].set_yscale('log')

axes[1].plot(history_sc.history['loss'], label='Train')
axes[1].plot(history_sc.history['val_loss'], label='Validation')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Loss (MSE)')
axes[1].set_title('Periodic Regression - Training History')
axes[1].legend()
axes[1].grid(True, alpha=0.3)
axes[1].set_yscale('log')

plt.tight_layout()
plt.savefig('images/training_history_comparison.png', dpi=300, bbox_inches='tight')
plt.show()


