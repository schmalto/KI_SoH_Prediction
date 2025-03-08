import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dropout, Dense, Concatenate
)
from src.model.model_train_helpers import (
    generate_combined_dataset,
    train_model
)


def create_mi_model():
    input_ids = Input(shape=(1,), name='ids')
    a = Dense(64, activation='relu',)(input_ids)
    a = Dense(16, activation='relu',)(a)
    input_rds = Input(shape=(1,), name='rds')
    b = Dense(64, activation='relu',)(input_rds)
    b = Dense(16, activation='relu',)(b)
    input_uds_max = Input(shape=(1,), name='uds_max')
    c = Dense(64, activation='relu',)(input_uds_max)
    c = Dense(16, activation='relu',)(c)
    input_usd_min = Input(shape=(1,), name='usd_min')
    d = Dense(64, activation='relu',)(input_usd_min)
    d = Dense(16, activation='relu',)(d)
    input_usd_max = Input(shape=(1,), name='usd_max')
    e = Dense(64, activation='relu',)(input_usd_max)
    e = Dense(16, activation='relu',)(e)
    input_uth_down = Input(shape=(1,), name='uth_down')
    f = Dense(64, activation='relu',)(input_uth_down)
    f = Dense(16, activation='relu',)(f)
    input_uth_up = Input(shape=(1,), name='uth_up')
    g = Dense(64, activation='relu',)(input_uth_up)
    g = Dense(16, activation='relu',)(g)

    combined = Concatenate()([a, b, c, d, e, f, g])
    # combined = Dense(512, activation='relu',)(combined)  # v11
    combined = Dense(128, activation='relu',)(combined)  # v9
    combined = Dropout(0.05)(combined)  # v6
    combined = Dense(64, activation='relu',)(combined)  # v9
    predictions = Dense(1)(combined)

    model = tf.keras.models.Model(
        inputs=[input_ids, input_rds, input_uds_max, input_usd_min, input_usd_max, input_uth_down, input_uth_up], outputs=predictions
    )
    mean_absolute_percentage_error = tf.keras.losses.MeanAbsolutePercentageError()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),  # v13
        loss=mean_absolute_percentage_error,
    )
    print(model.summary())
    return model


def start_training():
    model_name = "max_model_3"
    datasets = ("ids_to_break_max", "rds_to_break_max", "uds_max_to_break_max", "usd_min_to_break_max", "usd_max_to_break_max", "uth_down_to_break_max", "uth_up_to_break_max")
    train_ds, val_ds, test_ds = generate_combined_dataset(
        datasets
    )
    model = create_mi_model()
    # plot_model(
    #     model, to_file=f"src/model/checkpoints/{model_name}.png", show_shapes=True)
    trained_model = train_model(model, train_ds, val_ds, model_name)
    return trained_model


if __name__ == "__main__":
    start_training()
