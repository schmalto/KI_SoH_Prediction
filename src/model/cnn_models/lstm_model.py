import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
from termcolor import colored
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Conv1D, MaxPool1D, BatchNormalization, Dropout,
    Flatten, Dense, Concatenate, GlobalAveragePooling1D, LSTM
)
from tensorflow.keras.regularizers import l2
from src.model.model_train_helpers import (
    PlotLossAccuracy,
    modelCheckpoint,
    generate_combined_dataset,
    test_single_values_in_model,
    train_model
)
from tensorflow.keras.callbacks import EarlyStopping
from keras.utils import plot_model
from tensorflow.keras import regularizers


# tf.keras.mixed_precision.set_global_policy('mixed_float16')


def create_mi_model():
    input_rdson = Input(shape=(100, 1), name='rdson')
    x = BatchNormalization()(input_rdson)
    x = LSTM(256, return_sequences=True)(x)
    x = LSTM(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = Dropout(0.05)(x)  # v6
    x = LSTM(128, return_sequences=True)(x)
    x = BatchNormalization()(x)
    x = LSTM(64, return_sequences=False)(x)
    x = Dense(128)(x)
    predictions = Dense(1)(x)

    model = tf.keras.models.Model(
        inputs=[input_rdson], outputs=predictions
    )
    mean_absolute_percentage_error = tf.keras.losses.MeanAbsolutePercentageError()
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001),  # v13
        loss=mean_absolute_percentage_error,
    )
    print(model.summary())
    return model


def start_training():
    model_name = "lstm_model_sampled_3"
    train_ds, val_ds, test_ds = generate_combined_dataset(("rdson_sampled_all",),)
    model = create_mi_model()
    trained_model = train_model(model, train_ds, val_ds, model_name)
    return trained_model


if __name__ == "__main__":
    start_training()
