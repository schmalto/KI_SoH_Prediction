"""
Multi-modal CNN model for SoH prediction using RDS(on) curves and cooling data.
"""
import os
import tensorflow as tf
from keras.layers import (
    Dense, Dropout, Flatten, Conv1D, MaxPool1D, Input,
    BatchNormalization, Concatenate
)
from src.model.model_train_helpers import (
    PlotLossAccuracy, modelCheckpoint, generate_combined_dataset,
    test_single_values_in_model
)

# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define base directory relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "src", "model", "checkpoints")
TEST_DATA_DIR = os.path.join(BASE_DIR, "src", "model")

# Ensure directories exist
for directory in [MODEL_DIR, CHECKPOINT_DIR, TEST_DATA_DIR]:
    os.makedirs(directory, exist_ok=True)


def create_mi_model():
    """
    Create a multi-input CNN model that combines RDS(on) time series with cooling data.
    
    Returns:
        tf.keras.Model: Compiled multi-input model
    """
    # Branch 1: RDS(on) input with CNN layers
    input_rdson = Input(shape=(9940, 1), name='rdson')
    
    # Initial normalization
    x = BatchNormalization()(input_rdson)
    x = BatchNormalization()(x)
    
    # CNN block 1
    x = Conv1D(64, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    
    # CNN block 2
    x = Conv1D(64, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    
    # CNN block 3
    x = Conv1D(128, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    
    # CNN block 4
    x = Conv1D(256, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(8)(x)
    
    # Dense layer for feature extraction
    x = Dense(32, activation='relu')(x)

    # Branch 2: Delta T input
    input_cooling = Input(shape=(1,), name='cooling_input')
    z = Dense(128, activation='relu')(input_cooling)
    z = Dense(32, activation='relu')(z)

    # Combine branches
    combined = Concatenate()([x, z])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.4)(combined)
    predictions = Dense(1)(combined)

    # Create and compile model
    model = tf.keras.models.Model(
        inputs=[input_rdson, input_cooling], 
        outputs=predictions
    )
    
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model.compile(
        optimizer=optimizer, 
        loss='mean_squared_error'
    )
    
    model.summary()
    return model


def train_model(model, train_ds, val_ds, model_name):
    """
    Train the multi-input model with appropriate callbacks.
    
    Args:
        model: The model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        model_name: Name for saving the model
        
    Returns:
        tuple: (trained_model, training_history)
    """
    n_epochs = 10
    
    # Set up callbacks
    callbacks = [
        PlotLossAccuracy(model_name),
        modelCheckpoint(model_name),
        tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', 
            patience=5, 
            restore_best_weights=True
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', 
            patience=3, 
            factor=0.5
        )
    ]

    # Train model
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=n_epochs, 
        batch_size=32, 
        callbacks=callbacks
    )
    
    # Save final model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    model.save(model_path)
    
    return model, history


def start_training():
    """
    Start the training process by generating datasets and training the model.
    
    Returns:
        tuple: (trained_model, training_history)
    """
    model_name = "lstm_cnn"
    
    # Generate datasets
    train_ds, val_ds, test_ds = generate_combined_dataset(
        "to_break", "rdson", "cooling"
    )
    
    # Create and train model
    model = create_mi_model()
    trained_model, history = train_model(model, train_ds, val_ds, model_name)
    
    # Save test dataset for later evaluation
    test_ds_path = os.path.join(TEST_DATA_DIR, "test_ds.tfrecord")
    tf.data.experimental.save(test_ds, test_ds_path)
    
    return trained_model, history


def evaluate_model():
    """
    Evaluate a trained model on test data.
    """
    try:
        test_ds_path = os.path.join(TEST_DATA_DIR, "test_ds.tfrecord")
        test_ds = tf.data.experimental.load(test_ds_path)
        
        model_path = os.path.join(CHECKPOINT_DIR, "multi_input_cnn_3.keras")
        model = tf.keras.models.load_model(model_path)
        
        test_single_values_in_model(model, test_ds)
    except Exception as e:
        print(f"Error during model evaluation: {e}")


if __name__ == "__main__":
    start_training()
    # evaluate_model()
