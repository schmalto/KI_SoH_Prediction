import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPool1D, BatchNormalization, Dropout,
    Flatten, Dense, Concatenate, LSTM, GRU
)
from src.model.model_train_helpers import (
    generate_combined_dataset,
    train_model
)


# tf.keras.mixed_precision.set_global_policy('mixed_float16')


def create_mi_model():
    input_rdson = Input(shape=(50, 1), name='rdson')
    x = BatchNormalization()(input_rdson)
    # x = Conv1D(16, 2, activation='relu',)(x) # v4
    # x = Conv1D(32, 2, activation='relu',)(x)
    # x = Dropout(0.15)(x)  # v3
    x = Conv1D(64, 2, activation='relu',)(x)
    x = Dropout(0.15)(x)  # v3
    x = BatchNormalization()(x)
    x = Conv1D(128, 2, activation='relu',)(x)
    x = Dropout(0.15)(x)  # v3
    x = BatchNormalization()(x)
    x = Conv1D(256, 2, activation='relu',)(x)
    x = Dropout(0.15)(x)
    x = BatchNormalization()(x)
    x = Flatten()(x)
    # x = Dropout(0.05)(x)
    x = Dense(512, activation='relu',)(x)

    input_vth = Input(shape=(25, 1), name='vth')
    y = BatchNormalization()(input_vth)
    y = Conv1D(64, 2, activation='relu',)(y)
    y = Dropout(0.3)(y)  # v3
    y = Conv1D(128, 2, activation='relu',)(y)
    y = Dropout(0.3)(y)
    y = BatchNormalization()(y)
    y = MaxPool1D(2)(y)
    y = Conv1D(256, 2, activation='relu',)(y)
    y = Dropout(0.3)(y)  # v3
    y = BatchNormalization()(y)
    y = Flatten()(y)
    y = Dense(128, activation='relu',)(y)

    input_cooling = Input(shape=(1,), name='cooling')
    z = Dense(256, activation='relu',)(input_cooling)
    z = Dropout(0.2)(z)
    z = Dense(32, activation='relu',)(z)

    combined = Concatenate()([x, z, y])
    combined = BatchNormalization()(combined)
    # combined = Dense(2048, activation='relu',)(combined)  # v25
    combined = Dense(1024, activation='relu',)(combined)  # v25
    combined = Dropout(0.2)(combined)  # 0.1
    combined = Dense(128, activation='relu',)(combined)
    # combined = Dropout(0.1)(combined)  # v25
    combined = Dense(64, activation='relu',)(combined)
    predictions = Dense(1)(combined)

    model = tf.keras.models.Model(
        inputs=[input_rdson, input_vth, input_cooling], outputs=predictions
    )
    mean_absolute_percentage_error = tf.keras.losses.MeanAbsolutePercentageError()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss=mean_absolute_percentage_error,
    )
    print(model.summary())
    return model

    # v1: 31 15Epoch
    # v2: 25 15Epoch


def start_training():
    model_name = "multi_input_vth_rdson_cooling_003"
    train_ds, val_ds, test_ds = generate_combined_dataset(
        ("rdson_sampled_all", "vth_sampled_all", "cooling_to_break"))
    model = create_mi_model()
    # plot_model(
    #     model, to_file=f"src/model/checkpoints/{model_name}.png", show_shapes=True)
    trained_model = train_model(model, train_ds, val_ds, model_name)
    return trained_model


if __name__ == "__main__":
    start_training()
    # evaluate_model()
