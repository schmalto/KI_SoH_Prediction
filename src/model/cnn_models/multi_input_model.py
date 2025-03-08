import os
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPool1D, BatchNormalization, Dropout,
    Flatten, Dense, Concatenate
)
from src.model.model_train_helpers import PlotLossAccuracy, modelCheckpoint, generate_combined_dataset, test_single_values_in_model
from keras.utils import plot_model


# Suppress TensorFlow warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define base directory relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MODEL_DIR = os.path.join(BASE_DIR, "models")
CHECKPOINT_DIR = os.path.join(BASE_DIR, "src", "model", "checkpoints")
TEST_DATA_DIR = os.path.join(BASE_DIR, "src", "model")

# Ensure directories exist
for directory in [MODEL_DIR, CHECKPOINT_DIR]:
    os.makedirs(directory, exist_ok=True)

# Uncomment to use mixed precision
# tf.keras.mixed_precision.set_global_policy('mixed_float16')


def create_mi_model():
    """Create a multi-input CNN model for SoH prediction."""
    # RDSon input branch
    input_rdson = Input(shape=(9940, 1), name='rdson')
    x = BatchNormalization()(input_rdson)
    x = BatchNormalization()(x)
    x = Conv1D(64, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    # x = MaxPool1D(2)(x)
    x = Conv1D(64, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(128, 2, activation='relu',)(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    # x = Dropout(0.3)(x)
    x = Conv1D(256, 2, activation='relu',)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.3)(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(512, 2, activation='relu',)(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(1024, 2, activation='relu',)(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(4)(x)
    x = Flatten()(x)
    x = Dense(256, activation='relu',)(x)
    x = Dense(128, activation='relu',)(x)
    x = Dropout(0.3)(x)

    # Cooling input branch
    input_cooling = Input(shape=(1,), name='cooling')
    z = Dense(128, activation='relu',)(input_cooling)
    z = Dropout(0.3)(z)
    z = Dense(64, activation='relu',)(z)
    z = Dropout(0.3)(z)

    # Combine branches
    combined = Concatenate()([x, z])
    combined = Dense(64, activation='relu',)(combined)
    combined = Dropout(0.3)(combined)
    predictions = Dense(1)(combined)

    # Create model
    model = tf.keras.models.Model(
        inputs=[input_rdson, input_cooling], outputs=predictions
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mean_squared_error'
    )
    print(model.summary())
    return model


def train_model(model, train_ds, val_ds, model_name):
    """Train the model and save checkpoints.
    
    Args:
        model: Model to train
        train_ds: Training dataset
        val_ds: Validation dataset
        model_name: Name for saving the model
        
    Returns:
        tuple: (trained_model, training_history)
    """
    n_epochs = 200
    pltCallBack = PlotLossAccuracy(model_name)
    saveCallBack = modelCheckpoint(model_name)
    
    history = model.fit(
        train_ds, 
        validation_data=val_ds, 
        epochs=n_epochs,
        batch_size=128, 
        callbacks=[pltCallBack, saveCallBack]
    )
    
    # Save final model
    model_path = os.path.join(MODEL_DIR, f"{model_name}.keras")
    model.save(model_path)
    
    return model, history


def start_training():
    """Start the training process."""
    model_name = "multi_input_cnn_19"
    features = ['rdson', 'cooling', 'vth_down']
    
    # Generate datasets
    train_ds, val_ds, test_ds = generate_combined_dataset(
        "to_break", features[0], features[1])
        
    # Create and visualize model
    model = create_mi_model()
    plot_path = os.path.join(CHECKPOINT_DIR, f"{model_name}.png")
    plot_model(model, to_file=plot_path, show_shapes=True)
    
    # Train model
    trained_model = train_model(model, train_ds, val_ds, model_name)
    return trained_model


def evaluate_model():
    """Evaluate the model on test data."""
    test_ds_path = os.path.join(TEST_DATA_DIR, "test_ds.tfrecord")
    test_ds = tf.data.Dataset.load(test_ds_path)
    
    model_path = os.path.join(CHECKPOINT_DIR, "multi_input_cnn_3.keras")
    model = tf.keras.models.load_model(model_path)
    
    test_single_values_in_model(model, test_ds)


if __name__ == "__main__":
    start_training()
    # evaluate_model()
