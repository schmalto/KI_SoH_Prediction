"""
Model visualization utilities for neural network architectures and training metrics.

This module provides functions for:
- Visualizing model architectures using different visualization libraries
- Plotting training and validation losses
- Comparing multiple models' performance
- Generating prediction plots
"""
import os
import sys
import pickle
import numpy as np
import tensorflow as tf
import matplotlib
import matplotlib.pyplot as plt
from termcolor import colored
import visualkeras

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define base directory relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, "src", "model")
PLOTS_DIR = os.path.join(MODEL_DIR, "plots")
CHECKPOINTS_DIR = os.path.join(MODEL_DIR, "checkpoints_paper")
TRAINING_HIST_DIR = os.path.join(MODEL_DIR, "training_hist")

# Ensure all required directories exist
for directory in [PLOTS_DIR, CHECKPOINTS_DIR, TRAINING_HIST_DIR]:
    os.makedirs(directory, exist_ok=True)

# Configure matplotlib
matplotlib.use('TkAgg')  # Use TkAgg backend for interactive plotting


def load_model(model_name):
    """Load a model from checkpoints directory.

    Args:
        model_name: Name of the model to load

    Returns:
        The loaded Keras model or None if loading fails
    """
    model_path = os.path.join(CHECKPOINTS_DIR, f"{model_name}.keras")

    try:
        if not os.path.exists(model_path):
            print(colored(f"Model file not found: {model_path}", "red"))
            return None

        return tf.keras.models.load_model(model_path)
    except Exception as e:
        print(colored(f"Error loading model {model_name}: {e}", "red"))
        return None


def load_training_history(model_name):
    """Load training history from pickle file.

    Args:
        model_name: Name of the model

    Returns:
        List of training metrics per epoch or None if loading fails
    """
    history_path = os.path.join(TRAINING_HIST_DIR, f"{model_name}_history.pkl")

    try:
        if not os.path.exists(history_path):
            print(colored(f"History file not found: {history_path}", "red"))
            return None

        with open(history_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        print(colored(f"Error loading training history for {model_name}: {e}", "red"))
        return None


def text_callable(layer_index, layer):
    """Generate text description for layer visualization.

    Args:
        layer_index: Index of the layer
        layer: Layer object

    Returns:
        Tuple of (text, position_above)
    """
    # Every other piece of text is drawn above the layer
    above = bool(layer_index % 2)

    # Get the output shape of the layer
    output_shape = [x for x in list(layer.output_shape) if x is not None]

    # If the output shape is a list of tuples, take only the first one
    if isinstance(output_shape[0], tuple):
        output_shape = list(output_shape[0])
        output_shape = [x for x in output_shape if x is not None]

    # Create a string representation of the output shape
    output_shape_txt = ""
    for ii in range(len(output_shape)):
        output_shape_txt += str(output_shape[ii])
        if ii < len(output_shape) - 2:  # Add an x between dimensions
            output_shape_txt += "x"
        if ii == len(output_shape) - 2:  # Add a newline between the last two dimensions
            output_shape_txt += "\n"

    # Add the name of the layer to the text
    output_shape_txt += f"\n{layer.name}"

    return output_shape_txt, above


def generate_model_image(model_name):
    """Generate visualization images of the model architecture.

    Args:
        model_name: Name of the model to visualize
    """
    model = load_model(model_name)
    if model is None:
        return

    output_file = os.path.join(PLOTS_DIR, f"{model_name}.pdf")
    graph_output_file = os.path.join(PLOTS_DIR, f"{model_name}_graph.pdf")

    try:
        # Create layered view visualization
        visualkeras.layered_view(
            model,
            to_file=output_file,
            draw_volume=True,
            max_xy=700,
            legend=True,
            show_dimension=True
        )
        print(colored(f"Layered view saved to {output_file}", "green"))

        # Optionally create graph view visualization (commented out by default)
        # visualkeras.graph_view(
        #     model,
        #     to_file=graph_output_file,
        #     show_neurons=False
        # )
        # print(colored(f"Graph view saved to {graph_output_file}", "green"))
    except Exception as e:
        print(colored(f"Error generating model visualization: {e}", "red"))


def show_model_structure(model_name):
    """Print the structure of the model.

    Args:
        model_name: Name of the model
    """
    model = load_model(model_name)
    if model is None:
        return

    print("\nModel Structure:")
    print("-" * 80)
    model.summary()
    print("-" * 80)


def plotModelLoss(model_name):
    """Plot the training and validation loss for a model.

    Args:
        model_name: Name of the model
    """
    history = load_training_history(model_name)
    if history is None:
        return

    # Extract loss values
    train_losses = [h.get('loss', 0) for h in history]
    val_losses = [h.get('val_loss', 0) for h in history]

    # Create plot
    fig, ax = plt.subplots(figsize=(10, 6))

    ax.plot(train_losses, label="Training loss", linewidth=1.5)
    ax.plot(val_losses, label="Validation loss", linewidth=1.5)

    # Configure plot
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title(f"Training History: {model_name}", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save figure
    output_file = os.path.join(PLOTS_DIR, f"{model_name}_loss.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    print(colored(f"Loss plot saved to {output_file}", "green"))

    plt.show()

    # Print minimum losses
    min_train_loss = min(train_losses)
    min_val_loss = min(val_losses)
    print(f"Minimum training loss: {min_train_loss:.6f}")
    print(f"Minimum validation loss: {min_val_loss:.6f}")


def plotTrainLoss(models):
    """Compare training loss for multiple models.

    Args:
        models: List of model names to compare
    """
    if not isinstance(models, list):
        models = [models]

    fig, ax = plt.subplots(figsize=(12, 7))

    for model_name in models:
        history = load_training_history(model_name)
        if history is None:
            continue

        val_losses = [h.get('val_loss', 0) for h in history]
        val_losses = np.clip(val_losses, 0, 150)  # Clip extreme values for better visualization
        ax.plot(val_losses, label=f"{model_name}", linewidth=1.5)

    # Configure plot
    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("Loss", fontsize=12)
    ax.set_title("Model Comparison", fontsize=14)
    ax.legend(fontsize=10)
    ax.grid(True, linestyle='--', alpha=0.7)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # Save figure
    model_names = "_vs_".join([m[-3:] for m in models])  # Use last 3 chars of each model name
    output_file = os.path.join(PLOTS_DIR, f"comparison_{model_names}.pdf")
    plt.savefig(output_file, bbox_inches='tight')
    print(colored(f"Comparison plot saved to {output_file}", "green"))

    plt.show()


def mse_to_mae(mse):
    """Convert Mean Squared Error to Mean Absolute Error.

    Args:
        mse: Mean Squared Error value

    Returns:
        Approximate Mean Absolute Error
    """
    return np.sqrt(mse * 2 / np.pi)


def mae_to_mape(mae):
    """Convert Mean Absolute Error to Mean Absolute Percentage Error.

    Args:
        mae: Mean Absolute Error value

    Returns:
        Approximate Mean Absolute Percentage Error
    """
    average_cycles = 32000  # Average number of cycles
    return mae / average_cycles * 100  # Convert to percentage


def print_model_size():
    """Print the sizes of all models in the checkpoints directory."""
    models = [f for f in os.listdir(CHECKPOINTS_DIR) if f.endswith('.keras')]

    print("\nModel Size Comparison")
    print("-" * 80)
    print(f"{'Model Name':<40} {'Size (MB)':<10} {'Parameters':<12}")
    print("-" * 80)

    for model_file in sorted(models):
        model_name = os.path.splitext(model_file)[0]
        model_path = os.path.join(CHECKPOINTS_DIR, model_file)

        try:
            # Get file size
            size_mb = os.path.getsize(model_path) / (1024 * 1024)

            # Load model to get parameter count
            model = load_model(model_name)
            if model:
                params = model.count_params()
                print(f"{model_name:<40} {size_mb:.2f} MB {params:,}")
            else:
                print(f"{model_name:<40} {size_mb:.2f} MB {'N/A':<12}")

        except Exception as e:
            print(f"{model_name:<40} Error: {str(e)}")

    print("-" * 80)


def list_available_models():
    """List all available models in the checkpoints directory."""
    models = [os.path.splitext(f)[0] for f in os.listdir(CHECKPOINTS_DIR) if f.endswith('.keras')]

    print("\nAvailable Models:")
    print("-" * 80)
    for i, model in enumerate(sorted(models), 1):
        print(f"{i}. {model}")
    print("-" * 80)

    return sorted(models)


def plot_predictions(model_name, dataset_names=None):
    """Generate prediction plots using the specified model.

    Args:
        model_name: Name of the model to use
        dataset_names: List of dataset names to use for prediction
    """
    if dataset_names is None:
        dataset_names = ["rdson_sampled_all", "vth_sampled_all", "cooling_to_break"]

    # Import here to avoid circular imports
    from src.model.model_train_helpers import plot_model_predictions

    try:
        plot_model_predictions(model_name, dataset_names, "DUT_88")
        print(colored(f"Prediction plot for {model_name} generated successfully", "green"))
    except Exception as e:
        print(colored(f"Error generating prediction plot: {e}", "red"))


def main():
    """Main function to handle command line arguments."""
    available_models = list_available_models()

    if len(sys.argv) > 1:
        command = sys.argv[1]

        if command == "list":
            # Already listed models at start
            pass
        elif command == "loss" and len(sys.argv) > 2:
            model_name = sys.argv[2]
            if model_name in available_models:
                plotModelLoss(model_name)
            else:
                print(colored(f"Model {model_name} not found", "red"))
        elif command == "compare" and len(sys.argv) > 2:
            model_names = sys.argv[2:]
            models_to_compare = [m for m in model_names if m in available_models]
            if models_to_compare:
                plotTrainLoss(models_to_compare)
            else:
                print(colored("None of the specified models were found", "red"))
        elif command == "structure" and len(sys.argv) > 2:
            model_name = sys.argv[2]
            if model_name in available_models:
                show_model_structure(model_name)
            else:
                print(colored(f"Model {model_name} not found", "red"))
        elif command == "visualize" and len(sys.argv) > 2:
            model_name = sys.argv[2]
            if model_name in available_models:
                generate_model_image(model_name)
            else:
                print(colored(f"Model {model_name} not found", "red"))
        elif command == "predict" and len(sys.argv) > 2:
            model_name = sys.argv[2]
            if model_name in available_models:
                plot_predictions(model_name)
            else:
                print(colored(f"Model {model_name} not found", "red"))
        elif command == "sizes":
            print_model_size()
        else:
            print(colored("Unknown command or missing arguments", "red"))
            print("Usage: python visualize_model.py [list|loss|compare|structure|visualize|predict|sizes] [model_name(s)]")
    else:
        # Default action when no arguments are provided
        model_name = "multi_input_vth_rdson_cooling_003"
        if model_name in available_models:
            print(colored(f"Generating prediction plot for default model: {model_name}", "green"))
            plot_predictions(model_name)
        else:
            print(colored(f"Default model {model_name} not found.", "yellow"))
            print("Use 'python visualize_model.py list' to see available models")


if __name__ == "__main__":
    main()
