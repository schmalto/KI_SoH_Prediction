import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPool1D, BatchNormalization, Dropout,
    Flatten, Dense, Concatenate
)
from src.model.model_train_helpers import (
    generate_training_data, train_model
)
from keras.utils import plot_model


# tf.keras.mixed_precision.set_global_policy('mixed_float16')


def create_mi_model():
    input_rdson = Input(shape=(100, 1), name='rdson')
    x = BatchNormalization()(input_rdson)
    x = BatchNormalization()(x)
    x = Conv1D(64, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    # x = MaxPool1D(2)(x)
    x = Conv1D(64, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    # x = MaxPool1D(2)(x)
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
    # x = MaxPool1D(2)(x)
    x = Conv1D(1024, 2, activation='relu',)(x)
    x = BatchNormalization()(x)
    # x = MaxPool1D(2)(x)
    x = Conv1D(2048, 2, activation='relu',)(x)
    # x = MaxPool1D(2)(x)
    x = BatchNormalization()(x)
    # x = Conv1D(4096, 2, activation='relu',)(x)
    # x = BatchNormalization()(x)
    # x = BatchNormalization()(x)
    # x = Conv1D(6000, 2, activation='relu',)(x)
    x = MaxPool1D(4)(x)
    x = Flatten()(x)
    # x = Dense(4096, activation='relu',)(x)
    # x = Dense(2048, activation='relu',)(x)
    x = Dense(1024, activation='relu',)(x)
    x = Dense(256, activation='relu',)(x)
    # x = Dense(128, activation='relu',)(x)
    x = Dropout(0.3)(x)

    input_cooling = Input(shape=(1,), name='cooling')
    z = Dense(128, activation='relu',)(input_cooling)
    # z = Dropout(0.3)(z)
    z = Dense(32, activation='relu',)(z)
    # z = Dropout(0.3)(z)

    combined = Concatenate()([x, z])
    combined = Dense(64, activation='relu',)(combined)
    # combined = Dropout(0.3)(combined)
    predictions = Dense(1)(combined)

    model = tf.keras.models.Model(
        inputs=[input_rdson, input_cooling], outputs=predictions
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),
        loss='mean_squared_error'
    )
    print(model.summary())
    return model


def start_training():
    model_name = "multi_input_random_samples_complex_all_n100_2"
    train_ds, val_ds, test_ds = generate_training_data(
        ["rdson_sampled_all", "cooling_to_break"]
    )
    # train_ds, val_ds, test_ds = generate_combined_dataset("to_break", "rdson", "cooling")
    model = create_mi_model()
    plot_model(model, to_file=f"src/model/checkpoints/{model_name}.png", show_shapes=True)
    trained_model = train_model(model, train_ds, val_ds, model_name)
    return trained_model


if __name__ == "__main__":
    start_training()
