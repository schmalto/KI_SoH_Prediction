# AI State of Health Prediction Framework Documentation

:warning: **Note:** This documentation is largely auto-generated and may require manual review and editing for accuracy and completeness. See the [Disclaimer](#disclaimer) section for more details. :warning:

## Table of Contents

1. [Introduction](#1-introduction)
2. [Installation](#2-installation)
3. [Project Structure](#3-project-structure)
4. [Components and Modules](#4-components-and-modules)
    - [Dataset Generation](#41-dataset-generation)
    - [Model Architectures](#42-model-architectures)
    - [Training Utilities](#43-training-utilities)
    - [Visualization Tools](#44-visualization-tools)
5. [Workflow Guide](#5-workflow-guide)
    - [Data Preparation](#51-data-preparation)
    - [Model Training](#52-model-training)
    - [Model Evaluation](#53-model-evaluation)
6. [API Reference](#6-api-reference)
7. [Best Practices](#7-best-practices)
8. [Troubleshooting](#8-troubleshooting)

## 1. Introduction

This framework is part of a research project at the ILH at the University of Stuttgart to estimate the aging of power transistors (MOSFETs) using machine learning algorithms. It provides tools for processing electrical parameter measurements (RDSon, cooling curves, threshold voltages) to predict device degradation over time.

The system supports:

- Processing raw measurement data into ML-ready datasets
- Multiple neural network architectures (CNN, LSTM, multi-input models)
- Hyperparameter tuning
- Model training, evaluation, and visualization

## 2. Installation

### Prerequisites

- Python 3.8+
- TensorFlow 2.5+
- CUDA (for GPU acceleration, recommended)

### Setup

1. Clone the repository:

    ```bash
    git clone <repository-url>   
    cd KI_SoH_Pred   
    ```

2. Data Setup:
    - Place your raw measurement data in the `data/` directory
    - Note: Data is not included in the repository; you must source it externally

## 3. Project Structure

```bash
KI_SoH_Pred/
├── data/                  # Raw measurement data (not included in repo)
├── datasets/          # Generated and processed datasets
├── models/                # Saved trained models
├── plots/                 # Generated plots and visualizations└── src/    
├── dataset_gen/       # Dataset generation utilities 
│   ├── channel_conv.py           # Preprocess and convert raw measurements
│   ├── convert_mat_to_dataset.py # Convert MATLAB files to TF datasets    
│   ├── data_stats.py             # Statistical analysis tools    
│   ├── dataset_generation.py     # Generate numpy datasets   
│   ├── augmentation.py           # Data augmentation techniques    
│   └── plot_seq.py               # Visualize sequence data    
└── model/             # Model definitions, training, and evaluation        
    ├── cnn_models/              # CNN-based model architectures        
    │   ├── multi_input_model.py
    │   ├── multi_mode_model.py
    │   ├── lstm_model.py
    │   └── ...                   # Other model variants        
    ├── tuner_model_gen/         # Hyperparameter tuning utilities        
    ├── checkpoints_paper/       # Model checkpoint files        
    ├── training_hist/           # Training history logs        
    ├── model_train_helpers.py   # Model training/evaluation utilities        
└   ── visualize_model.py       # Visualization tools for models and training 
```

## 4. Components and Modules

### 4.1 Dataset Generation

The dataset generation pipeline converts raw electrical measurements into structured datasets for machine learning. It includes:

#### `channel_conv.py`

- **Purpose:** Preprocess and convert raw measurement data.
- **Key Functions:**  
- `convert_segmented_data(folder)`: Converts segmented data files.  
- `get_curves(cycles)`: Extracts specific curve types.  - `calculate_rdson_non_seg(voltage, current)`: Computes RDS(on) values.

#### `dataset_generation.py`

- **Purpose:** Create structured numpy datasets.- **Key Functions:**  - `generate_dataset_numpy(folder, feature_type, label_type)`: Generates datasets.  - `sampling(cycles, method)`: Implements various sampling strategies.  
- `generate_set(folder, feature_type, label_type)`: Generates feature/label pairs.

#### `convert_mat_to_dataset.py`

- **Purpose:** Convert MATLAB data into TensorFlow datasets.
- **Key Functions:**  
- `create_datasets(datasets, feature_type, label_type)`: Processes multiple datasets.  
- `process_dataset(dataset, feature_type, label_type)`: Processes an individual dataset.

#### `data_stats.py`

- **Purpose:** Perform statistical analysis on datasets.
- **Key Functions:**  
- `get_all_data()`: Aggregates data from multiple datasets.  
- `display_corr_matrix_heatmap()`: Visualizes correlations between parameters.  - `analyze_feature_importance()`: Assesses feature significance.

#### `plot_seq.py`

- **Purpose:** Create technical drawing style plots of sequence data.
- **Key Functions:**  
- `plot_data(x, y, sequence_points)`: Creates technical drawing style plots with dimension lines.  
- `create_sequence_visualization(data, sequence_length)`: Visualizes data segmentation.

### 4.2 Model Architectures

#### `multi_input_model.py`

- **Purpose:** Implements a multi-input CNN that combines RDSon and cooling data.
- **Key Functions:**  
- `create_mi_model()`: Creates a multi-input CNN model architecture.  
- `train_model()`: Trains the model with appropriate callbacks.
- **Usage Example:**

```python
    from src.model.cnn_models.multi_input_model import create_mi_model, train_model
    model = create_mi_model()
    trained_model, history = train_model(model, train_ds, val_ds, "multi_input_model_001")

```

#### `multi_mode_model.py`

- **Purpose:** Advanced multi-input CNN with enhanced architecture.
- **Features:**  
- Multiple CNN blocks for robust feature extraction.  
- Dropout layers to mitigate overfitting.  
- Combined prediction head for multi-modal inputs.

#### Other Architectures

- **LSTM Models:** For handling sequential data.
- **Max Models:** Focused on peak parameter extraction.
- **Tunable Models:** Configurable options for hyperparameter optimization.

### 4.3 Training Utilities

#### `model_train_helpers.py`

- **Purpose:** Utilities for training, evaluation, and model saving.
- **Key Functions:**  
- `train_model(model, train_ds, val_ds, model_name)`: Trains the model with callbacks.  
- `generate_training_data(training_set_names)`: Loads training datasets.  
- `load_dataset_from_record(path)`: Loads datasets from TFRecord files.  
- `enforce_monotonic_decreasing()`: Post-processes predictions to enforce expected physical behaviors.  
- `plotModelLoss(modelname)`: Plots training and validation loss curves.  
- `rsquared(x, y)`: Calculates the coefficient of determination.  
- `plot_model_predictions()`: Generates prediction plots.

### 4.4 Visualization Tools

#### `visualize_model.py`

- **Purpose:** Tools to visualize model architectures, training progress, and predictions.
- **Key Functions:**  
- `plotModelLoss(model_name)`: Plots training and validation loss curves.
- `plot_predictions(model_name, dataset_names)`: Visualizes model predictions against ground truth.
- `print_model_size()`: Displays the sizes of different models.
- `generate_model_image(model_name)`: Creates visualization of model architecture.
- `show_model_structure(model_name)`: Prints the model structure to the console.
- **Command Line Interface:**  

```bash
python src/model/visualize_model.py list  # List available models  
python src/model/visualize_model.py loss model_name  # Show loss plot  
python src/model/visualize_model.py predict model_name  # Generate predictions  
```

## 5. Workflow Guide

### 5.1 Data Preparation

Convert raw MATLAB or RDSON data into processed datasets:

```python
from src.dataset_gen.channel_conv import convert_segmented_data
from src.dataset_gen.convert_mat_to_dataset import create_datasets

# Convert raw segmented data
convert_segmented_data('TO_54')

# Generate processed datasets
create_datasets(datasets=['TO_54', 'TO_32', 'TO_88'],
feature_type='cooling',    # Options: 'rdson', 'cooling', 'vth_up', etc.
   label_type='to_break',      # Options: 'to_break', 'cycle' #
    augmentations=2)
```

### 5.2 Model Training

#### Standard Training

```python
from src.model.cnn_models.multi_mode_model import create_mi_model, train_model
from src.model.model_train_helpers import generate_combined_dataset
# Generate training, validation, and test datasets
train_ds, val_ds, test_ds = generate_combined_dataset(to_break, rdson, cooling)

# Create and train the model
model = create_mi_model()
model_name = "cooling_soh_predictor"
trained_model, history = train_model(model, train_ds, val_ds, model_name)
```

#### Hyperparameter Tuning

```python
from src.model.tuner_model_gen.tuner_helpers import tuner_start
from src.model.tuner_model_gen.tuner_cnn import build_model

# Run hyperparameter tuning to find the best model configuration
best_model = tuner_start("cnn_tuning", build_model)
```

### 5.3 Model Evaluation

Evaluate your trained model and visualize the outcomes:

```python
from src.model.visualize_model import plotModelLoss, plot_predictions

# Plot loss curves to review training progress
plotModelLoss("cooling_soh_predictor")

# Generate prediction plots comparing model output to ground truth
plot_predictions("cooling_soh_predictor", ["rdson_sampled_all", "cooling_to_break"])
```

## 6. API Reference

### Dataset Generation

```python
# Create datasets from raw data
create_datasets(
datasets,           # List of dataset names    
feature_type,       # Type of feature to extract (rdson, cooling, vth_up, etc.) 
label_type,         # Type of label to generate (to_break, cycle)    
augmentations=0     # Number of augmentations to perform
)
# Generate numpy datasets
generate_dataset_numpy(  
folder,             # Folder containing raw data    
feature_type,       # Type of feature to extract    
label_type          # Type of label to generate
)
```

### Model Training

```python
# Train a model
train_model(  
model,              # Model to train    
train_ds,           # Training dataset    
val_ds,             # Validation dataset    
model_name,         # Name for saving the model    
n_epochs=500        # Maximum number of epochs
)

# Generate training data
generate_training_data( 
training_set_names, # List of dataset names    
evaluation=False    # Whether to return evaluation datasets
)
```

### Visualization

```python
# Plot model loss
plotModelLoss(    
modelname,          # Name of the model    
save=False          # Whether to save the plot
)

# Plot model predictions
plot_model_predictions( 
modelname,          # Name of the model    
training_set_names, # List of dataset names    
dut,                # Device under test identifier    
batch_size=256,     # Batch size    
apply_filter=False  # Whether to apply monotonicity filtering
)
```

## 7. Best Practices

- **Data Preprocessing:**
- Apply batch normalization to input data.  
- Use appropriate sampling methods for large time series data.  
- Consider data augmentation for small datasets.

- **Model Architecture:**  
- For best results, use multi-input models combining different parameter types.  
- Add sufficient dropout (0.3-0.4) to prevent overfitting.  
- Use batch normalization after convolutional layers.

- **Training:**  
- Start with a learning rate of 0.0001.  
- Use early stopping with patience of 5-10 epochs.  
- Implement learning rate reduction on plateau.  
- Monitor training and validation loss to detect overfitting.

- **Evaluation:**  
- Apply monotonicity constraints to predictions for physically realistic outputs.  - Use multiple metrics (MSE, MAPE, R²) for comprehensive evaluation.  
- Compare model performance against simple baselines.

- **Path Management:**  
- Always use relative paths for better portability.  
- Ensure directories exist before saving files.

## 8. Troubleshooting

### Common Issues and Solutions

- **Data Path Errors:**  
- Verify that all BASE_DIR variables are correctly defined.  
- Ensure data is placed in the expected directories.  
- Use `os.path.exists()` to check if files/directories exist before operations.

- **Model Loading Errors:**  
- Check that model checkpoint files exist in the specified paths.  
- Verify model name consistency between saving and loading operations.  
- Ensure TensorFlow version compatibility with saved models.

- **Training Issues:**  
- If loss doesn't decrease, try adjusting the learning rate.  
- For overfitting, increase dropout or reduce model complexity.  
- For underfitting, increase model capacity or reduce regularization.

- **Memory Issues:**  
- Reduce batch size for large models or datasets.  
- Use TensorFlow's mixed precision training.  
- Process datasets in chunks if they're too large.

- **Visualization Problems:**  
- Check that matplotlib backend is set appropriately.  
- Ensure all paths for saving visualizations exist.  
- Verify that the training history files are available.

### Debugging Tips

- Use `inspect_tfrecord()` function to check dataset structure and content.
- Add print statements to track data flow in complex operations.
- Use TensorBoard for visualization of model architecture and training metrics.
- Check for NaN values in inputs and outputs with `tf.debugging.check_numerics()`.

---

This documentation provides a comprehensive guide to using the AI State of Health Prediction Framework. For additional support or specific implementation details, refer to the individual module docstrings and function headers in the source code.

### Disclaimer 

:warning: **Note:**
This documentation was largely auto-generated by GitHub Copilot using Claude 3.7 Sonnet Thinking based on the project's source code and structure. Some sections may require manual review and editing for accuracy and completeness. The information provided here is intended as a general guide and may not cover all possible scenarios or edge cases. For specific questions or issues, consult the project maintainer or refer to the source code directly. :warning:
