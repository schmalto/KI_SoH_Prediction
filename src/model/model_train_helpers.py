"""
Helper utilities for training and evaluating machine learning models for SoH prediction.

This module provides functions and classes for:
- Loading and preprocessing TFRecord datasets
- Training models with appropriate callbacks
- Evaluating model performance
- Visualizing prediction results
- Post-processing predictions to enforce monotonicity
"""
from sklearn.metrics import mean_absolute_percentage_error
from sklearn.isotonic import IsotonicRegression
import scipy
from termcolor import colored
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
import tensorflow as tf
import keras
from pathlib import Path
import pickle
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logging

# Standard library imports

# Third-party imports

# Define base directory relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
TRAINING_SETS_DIR = os.path.join(BASE_DIR, "training_sets")
MODEL_DIR = os.path.join(BASE_DIR, "src", "model")
CHECKPOINT_DIR = os.path.join(MODEL_DIR, "checkpoints_paper")
HISTORY_DIR = os.path.join(MODEL_DIR, "training_hist")
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")
LOGS_DIR = os.path.join(MODEL_DIR, "logs")

# Ensure all required directories exist
for directory in [CHECKPOINT_DIR, HISTORY_DIR, PLOTS_DIR, LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

matplotlib.use('svg')  # Use SVG backend for matplotlib


class PlotLossAccuracy(keras.callbacks.Callback):
    """Keras callback to log and save training metrics history.

    This callback tracks loss and accuracy metrics during training and saves
    the history to a pickle file after each epoch.

    Attributes:
        modelname: Name of the model for saving history files
    """

    def __init__(self, modelname):
        """Initialize callback with the model name.

        Args:
            modelname: Name used for saving history files
        """
        super(PlotLossAccuracy, self).__init__()
        self.modelname = modelname
        self.i = 0
        self.x = []
        self.acc = []
        self.losses = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    def on_train_begin(self, logs=None):
        """Initialize tracking lists at the start of training."""
        logs = logs or {}
        self.i = 0
        self.x = []
        self.acc = []
        self.losses = []
        self.val_losses = []
        self.val_acc = []
        self.logs = []

    def on_epoch_end(self, epoch, logs=None):
        """Update and save metrics at the end of each epoch.

        Args:
            epoch: Current epoch number
            logs: Dictionary containing the metrics
        """
        logs = logs or {}
        self.logs.append(logs)
        self.x.append(int(self.i))
        self.losses.append(logs.get('loss'))
        self.val_losses.append(logs.get('val_loss'))
        self.acc.append(logs.get('accuracy'))
        self.val_acc.append(logs.get('val_accuracy'))

        # Save history to file
        history_path = os.path.join(HISTORY_DIR, f"{self.modelname}_history.pkl")
        with open(history_path, "wb") as f:
            pickle.dump(self.logs, f)
        self.i += 1


def train_model(model, train_ds, val_ds, model_name, n_epochs=500):
    """Train a model with appropriate callbacks and save checkpoints.

    Args:
        model: Keras model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        model_name: Name for saving checkpoints and logs
        n_epochs: Maximum number of epochs to train

    Returns:
        tuple: (trained_model, training_history)
    """
    # Set up callbacks
    callbacks = [
        PlotLossAccuracy(model_name),
        modelCheckpoint(model_name),
        keras.callbacks.TensorBoard(log_dir=os.path.join(LOGS_DIR, model_name)),
        keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True
        )
    ]

    # Train the model
    history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=n_epochs,
        callbacks=callbacks
    )

    # Save final model
    final_model_path = os.path.join(CHECKPOINT_DIR, "train_end", f"{model_name}.keras")
    os.makedirs(os.path.dirname(final_model_path), exist_ok=True)
    model.save(final_model_path)

    return model, history


def load_dataset_from_record(path):
    """Load dataset from TFRecord file with appropriate parsing.

    Args:
        path: Path to TFRecord file

    Returns:
        tf.data.Dataset: Parsed dataset
    """
    if not os.path.exists(path):
        raise FileNotFoundError(f"Dataset file not found: {path}")

    # Select appropriate parsing function based on dataset type
    if "cooling" in path:
        raw_dataset = tf.data.TFRecordDataset(path)
        return raw_dataset.map(_parse_function_cooling).map(unpack_parsed_set)
    elif "vth" in path:
        raw_dataset = tf.data.TFRecordDataset(path)
        return raw_dataset.map(_parse_function_vth).map(unpack_parsed_set)
    else:
        raw_dataset = tf.data.TFRecordDataset(path)
        return raw_dataset.map(_parse_function).map(unpack_parsed_set)


def unpack_parsed_set(parsed_example):
    """Extract features and labels from parsed TFRecord example.

    Args:
        parsed_example: Parsed TensorFlow example

    Returns:
        tuple: (time_series_features, label)
    """
    return parsed_example['time_series'], parsed_example['label']


def _parse_function_cooling(example_proto):
    """Parse TFRecord example for cooling data.

    Args:
        example_proto: Serialized TensorFlow example

    Returns:
        dict: Parsed example with time_series and label
    """
    feature_description = {
        'time_series': tf.io.FixedLenFeature([1], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.float32),
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def _parse_function_vth(example_proto):
    """Parse TFRecord example for Vth data.

    Args:
        example_proto: Serialized TensorFlow example

    Returns:
        dict: Parsed example with time_series and label
    """
    feature_description = {
        'time_series': tf.io.FixedLenFeature([25], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.float32),
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def _parse_function(example_proto):
    """Parse TFRecord example for standard data.

    Args:
        example_proto: Serialized TensorFlow example

    Returns:
        dict: Parsed example with time_series and label
    """
    feature_description = {
        'time_series': tf.io.FixedLenFeature([50], tf.float32),
        'label': tf.io.FixedLenFeature([1], tf.float32),
    }
    return tf.io.parse_single_example(example_proto, feature_description)


def generate_training_data(training_set_names, evaluation=False):
    """Generate training data from specified dataset names.

    Args:
        training_set_names: List of dataset names to load
        evaluation: Whether to return evaluation datasets

    Returns:
        For evaluation=False: tuple of (train_ds, val_ds, test_ds)
        For evaluation=True: tf.data.Dataset
    """
    return generate_combined_dataset(training_set_names, evaluation)


def generate_combined_dataset(training_set_names, evaluation=False):
    """Generate and potentially combine multiple datasets.

    Args:
        training_set_names: List of dataset names to load
        evaluation: Whether to return evaluation datasets

    Returns:
        For evaluation=False: tuple of (train_ds, val_ds, test_ds)
        For evaluation=True: tf.data.Dataset
    """
    base_path = TRAINING_SETS_DIR
    if evaluation:
        base_path = os.path.join(base_path, "evaluation")

    # Load datasets
    ds = []
    for name in training_set_names:
        dataset_path = os.path.join(base_path, name, "dataset.tfrecord")
        try:
            ds.append(load_dataset_from_record(dataset_path))
        except FileNotFoundError:
            print(colored(f"Warning: Dataset not found: {dataset_path}", "yellow"))

    # Verify that datasets were loaded
    if len(ds) == 0:
        raise ValueError(f"No datasets were loaded from: {training_set_names}")

    # Return single dataset or combine multiple datasets
    if len(ds) == 1:
        ret_ds = ds[0]
    else:
        ret_ds = combine_datasets(tuple(ds))

    # Split into train/val/test or return evaluation dataset
    if evaluation:
        return ret_ds
    return split_dataset(ret_ds)


def combine_datasets(datasets, train_split=0.7):
    """Combine multiple datasets into a single dataset.

    Args:
        datasets: Tuple of tf.data.Dataset objects
        train_split: Fraction of data for training

    Returns:
        tf.data.Dataset: Combined dataset
    """
    combined_ds = tf.data.Dataset.zip(datasets).map(
        lambda *args: ((tuple(arg[0] for arg in args)), args[0][1])
    )
    return combined_ds


def split_dataset(dataset, train_split=0.7, batch_size=256):
    """Split dataset into training, validation, and test sets.

    Args:
        dataset: Dataset to split
        train_split: Fraction of data for training (0 to 1)
        batch_size: Batch size for returned datasets

    Returns:
        tuple: (train_ds, val_ds, test_ds)
    """
    val_split = (1 - train_split) / 2

    # Calculate dataset sizes
    try:
        dataset_size = dataset.reduce(0, lambda x, _: x + 1).numpy()
    except tf.errors.InvalidArgumentError:
        print(colored("Warning: Could not determine dataset size, using estimates.", "yellow"))
        # Use a large buffer size for thorough shuffling
        dataset = dataset.shuffle(buffer_size=10000)
        train_size = 100  # Default sizes if we can't determine dataset size
        val_size = 50
        test_size = 50
        return (
            dataset.take(train_size).batch(batch_size),
            dataset.skip(train_size).take(val_size).batch(batch_size),
            dataset.skip(train_size + val_size).take(test_size).batch(batch_size)
        )

    # Calculate split sizes
    train_size = int(train_split * dataset_size)
    val_size = int(val_split * dataset_size)

    # Shuffle and split the dataset
    dataset = dataset.shuffle(buffer_size=dataset_size * 2)
    train_ds = dataset.take(train_size).batch(batch_size)
    val_ds = dataset.skip(train_size).take(val_size).batch(batch_size)
    test_ds = dataset.skip(train_size + val_size).batch(batch_size)

    return train_ds, val_ds, test_ds


def evaluate_model(model, test_ds):
    """Evaluate model on test dataset.

    Args:
        model: Model to evaluate
        test_ds: Test dataset

    Returns:
        float: Test loss
    """
    test_loss = model.evaluate(test_ds)
    print(f"Test Loss: {test_loss}")
    return test_loss


def modelCheckpoint(modelname):
    """Create model checkpoint callback.

    Args:
        modelname: Name of model for checkpoint files

    Returns:
        keras.callbacks.ModelCheckpoint: Checkpoint callback
    """
    checkpoint_path = os.path.join(CHECKPOINT_DIR, f"{modelname}.keras")
    return keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        save_weights_only=False,
        monitor="val_loss",
        mode="min",
        save_best_only=True,
        verbose=1
    )


def test_single_values_in_model(model, test_ds):
    """Test model on single examples for debugging.

    Args:
        model: Model to test
        test_ds: Dataset containing test examples
    """
    test_ds = test_ds.shuffle(1000)
    element = test_ds.skip(1).take(1)
    for x, y in element:
        print(f"X Value: {x}")
        print(f"Predicted: {model.predict(x)}")
        print(f"Actual: {y}")


def plotModelLoss(modelname, save=False):
    """Plot training and validation loss curves.

    Args:
        modelname: Model name for loading history
        save: Whether to save the plot to file

    Returns:
        matplotlib.figure.Figure: The generated figure
    """
    # Load training history
    loss_path = os.path.join(HISTORY_DIR, f"{modelname}_history.pkl")
    try:
        with open(loss_path, 'rb') as f:
            data = pickle.load(f)
    except FileNotFoundError:
        print(colored(f"Training history not found: {loss_path}", "red"))
        return None

    # Extract loss values
    train_loss = []
    val_loss = []
    for el in data:
        train_loss.append(el.get('loss', 0))
        val_loss.append(el.get('val_loss', 0))

    # Create plot
    fig = plt.figure(frameon=True, figsize=(6.5, 5))
    ax = fig.add_subplot(111)
    ax.plot(val_loss, label="Validation loss", linewidth=1.5)
    ax.plot(train_loss, label="Training loss", linewidth=1.5)
    ax.legend(fontsize=12)
    ax.grid(linestyle='--', alpha=0.6)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_xlabel('Epochs', fontsize=12)
    ax.set_ylabel('Loss [Mean Squared Error]', fontsize=12)

    # Save figure if requested
    if save:
        save_path = os.path.join(PLOTS_DIR, f"{modelname}_loss.pdf")
        fig.savefig(save_path, format='pdf', bbox_inches='tight')
        print(colored(f"Loss plot saved to {save_path}", "green"))

    plt.tight_layout()
    plt.show()

    return fig


def rsquared(x, y):
    """Calculate the coefficient of determination (R²).

    Args:
        x: Array of predictions
        y: Array of true values

    Returns:
        float: R² value
    """
    slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(x, y)
    return r_value**2


def enforce_monotonic_decreasing_ir(predictions):
    """Enforce monotonically decreasing predictions using isotonic regression.

    Args:
        predictions: Array of predictions

    Returns:
        np.ndarray: Monotonically decreasing predictions
    """
    ir = IsotonicRegression(increasing=False)
    x = np.arange(len(predictions))
    predictions_monotonic = ir.fit_transform(x, predictions)
    return predictions_monotonic


def enforce_monotonic_decreasing(predictions, window_size=5):
    """Enforce monotonically decreasing predictions using moving average.

    Args:
        predictions: Array of predictions
        window_size: Size of window for moving average

    Returns:
        np.ndarray: Monotonically decreasing predictions
    """
    predictions_monotonic = predictions.copy()
    for i in range(window_size, len(predictions_monotonic)):
        avg = np.mean(predictions_monotonic[(i-window_size): (i-1)])
        if predictions_monotonic[i] > avg:
            predictions_monotonic[i] = avg
    return predictions_monotonic


def plot_model_predictions(modelname, training_set_names, dut, batch_size=256, apply_filter=False):
    """Plot model predictions against true values.

    Args:
        modelname: Name of the model to evaluate
        training_set_names: List of dataset names to use
        dut: Device under test identifier (for file naming)
        batch_size: Batch size for evaluation
        apply_filter: Whether to apply monotonicity filtering
    """
    # Load dataset and model
    try:
        test_ds = generate_training_data(training_set_names, evaluation=True)
        model_path = os.path.join(CHECKPOINT_DIR, f"{modelname}.keras")
        model = keras.models.load_model(model_path)
    except (FileNotFoundError, ValueError) as e:
        print(colored(f"Error loading model or dataset: {e}", "red"))
        return

    # Create output directory
    save_path = os.path.join(PLOTS_DIR, modelname)
    Path(save_path).mkdir(parents=True, exist_ok=True)

    # Get predictions and actual values
    predictions = []
    actual = []
    el = test_ds.batch(batch_size)

    print("Evaluating model...")
    model.evaluate(el)

    for x, y in el:
        predictions.append(model.predict(x, verbose=0))
        actual.append(y)

    # Process predictions
    predictions = np.array(predictions[:-1]).flatten()
    predictions = predictions / len(predictions)

    # Process actual values
    actual = np.array(actual[:-1]).flatten()
    actual = actual / len(actual)
    actual = np.subtract(actual, np.ones(len(actual)))

    # Calculate metrics
    print(f"MAPE: {mean_absolute_percentage_error(actual, predictions)}")
    print(f"R² before filtering: {rsquared(predictions, actual)}")

    # Create RUL array (assuming decreasing from 1 to 0)
    rul = np.arange(len(predictions)) / len(predictions)
    rul = 1 - rul

    # Apply filtering if requested
    if apply_filter:
        predictions_filtered = np.clip(predictions, 0, 1)
        predictions_filtered = enforce_monotonic_decreasing_ir(predictions_filtered)
        predictions_filtered = enforce_monotonic_decreasing(predictions_filtered)
        print(f"R² after filtering: {rsquared(predictions_filtered, actual)}")

    # Create plot
    plt.figure(figsize=(9, 7))

    # Plot predictions
    plt.scatter(rul, predictions, label="$y_\\mathrm{pred}$", marker="x", color="black", s=2)

    # Plot filtered predictions if requested
    if apply_filter:
        plt.plot(rul, predictions_filtered, label="Predictions (filtered)", color="blue", linewidth=1.5)

    # Plot ground truth line
    plt.plot(rul, rul, label="$y_\\mathrm{true}$", color="green", linewidth=1.5)

    # Configure plot
    plt.xlabel('Normed Degradation (true)')
    plt.ylabel('Normed Degradation (est)')
    plt.legend()
    plt.grid(linestyle='--', alpha=0.6)

    # Save results
    np.savetxt(os.path.join(save_path, f"predictions_{dut}.txt"), predictions)
    np.savetxt(os.path.join(save_path, f"actual_{dut}.txt"), rul)
    plt.savefig(os.path.join(save_path, f"predictions_{dut}.pdf"), bbox_inches='tight')

    plt.show()


def calculate_mape(pred, actual):
    """Calculate mean absolute percentage error.

    Args:
        pred: Predicted values
        actual: Actual values

    Returns:
        float: MAPE value as percentage
    """
    return np.mean(np.abs((actual - pred) / np.maximum(np.abs(actual), 1e-10))) * 100


def inspect_tfrecord(file_path):
    """Inspect the structure of a TFRecord file for debugging.

    Args:
        file_path: Path to TFRecord file
    """
    if not os.path.exists(file_path):
        print(colored(f"File not found: {file_path}", "red"))
        return

    raw_dataset = tf.data.TFRecordDataset(file_path)
    for i, raw_record in enumerate(raw_dataset.take(3)):  # Look at first 3 examples
        example = tf.train.Example()
        example.ParseFromString(raw_record.numpy())
        ts = example.features.feature['time_series'].float_list.value
        label = example.features.feature['label'].float_list.value
        print(f"Example {i}:")
        print(f"  - time_series shape: {np.array(ts).shape}")
        print(f"  - label: {label}")

    # Count total records
    count = sum(1 for _ in tf.data.TFRecordDataset(file_path))
    print(f"Total records: {count}")


def test_y_equal(ds_1, ds_2):
    """Test if two datasets have the same labels.

    Args:
        ds_1: First dataset
        ds_2: Second dataset

    Returns:
        bool: True if all labels match (except last intentionally modified one)
    """
    # Collect labels from first dataset
    y_val = []
    for x, y in ds_1:
        y_val.append(y.numpy())
    y_val[len(y_val)-1] = 16  # Intentional modification for testing

    # Collect labels from second dataset
    y_val_2 = []
    for x, y in ds_2:
        y_val_2.append(y.numpy())

    # Compare labels and report differences
    differences = 0
    for i in range(len(y_val)):
        if not np.array_equal(y_val[i], y_val_2[i]):
            print(f"Difference at index {i}: {y_val[i]} vs {y_val_2[i]}")
            differences += 1

    print(f"Found {differences} differences in {len(y_val)} labels.")
    return differences == 0 or (differences == 1 and len(y_val) > 0)
