import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, BatchNormalization, Dropout,
    Flatten, Dense, Concatenate
)
from src.model.model_train_helpers import (
    generate_combined_dataset,
    train_model
)


def create_mi_model():
    input_rdson = Input(shape=(100, 1), name='rdson')
    x = BatchNormalization()(input_rdson)
    x = Conv1D(64, 2, activation='relu')(x)
    x = Dropout(0.2)(x)  # v6
    x = BatchNormalization()(x)
    x = Conv1D(128, 2, activation='relu',)(x)
    # x = Dropout(0.2)(x)  # v6
    x = Flatten()(x)
    x = Dense(64, activation='relu',)(x)

    input_vth = Input(shape=(100, 1), name='vth')
    y = BatchNormalization()(input_vth)
    y = Conv1D(64, 2, activation='relu',)(y)  # v9
    # y = Dropout(0.2)(y)  # v6
    y = BatchNormalization()(y)
    y = Conv1D(128, 2, activation='relu',)(y)  # v9
    # y = Dropout(0.2)(y)  # v6
    y = BatchNormalization()(y)
    y = Flatten()(y)
    y = Dense(64, activation='relu',)(y)  # v9

    input_cooling = Input(shape=(1,), name='cooling')
    z = Dense(128, activation='relu',)(input_cooling)  # v9
    z = Dense(32, activation='relu',)(z)  # v9

    combined = Concatenate()([x, z, y])
    combined = Dense(1024, activation='relu',)(combined)  # v11
    combined = Dense(128, activation='relu',)(combined)  # v9
    combined = Dense(64, activation='relu',)(combined)  # v9
    predictions = Dense(1)(combined)

    model = tf.keras.models.Model(
        inputs=[input_rdson, input_vth, input_cooling], outputs=predictions
    )
    mean_absolute_percentage_error = tf.keras.losses.MeanAbsolutePercentageError()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # v13
        loss=mean_absolute_percentage_error,
    )
    print(model.summary())
    return model


def start_training():
    model_name = "multi_input_vth_rdson_cooling_simple_12"
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
