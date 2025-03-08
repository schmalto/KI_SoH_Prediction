import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
from src.model.model_train_helpers import evaluate_model, generate_training_data, train_model
import tensorflow as tf
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, Input, BatchNormalization
from numba import cuda
device = cuda.get_current_device()
device.reset()



def create_cnn_model():
    input_rdson = Input(shape=(8000, 1), name='rdson_time_series')
    x = Conv1D(16, kernel_size=2, activation='relu', padding='same')(input_rdson)
    x = BatchNormalization()(x)
    # x = MaxPool1D(pool_size=2)(x)
    x = Conv1D(32, kernel_size=2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, kernel_size=2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(256, kernel_size=2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=2)(x)
    x = Conv1D(512, kernel_size=2, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    # x = Conv1D(1024, kernel_size=2, activation='relu', padding='same')(x)
    x = MaxPool1D(pool_size=2)(x)  # changed from 4 to 2
    # x = BatchNormalization()(x)

    x = Flatten()(x)

    combined = Dense(128, activation='relu')(x)
    # combined = Dropout(0.4)(combined)
    predictions = Dense(1)(combined)
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model = tf.keras.models.Model(
        inputs=input_rdson, outputs=predictions)
    model.summary()
    mean_average_percentage_error = tf.keras.losses.MeanAbsolutePercentageError()
    model.compile(optimizer=optimizer, loss=mean_average_percentage_error)
    return model


def main():
    model_name = "benchmark_sample_n8000_2"
    train_ds, val_ds, test_ds = generate_training_data(["rdson_sampled_all"])
    model = create_cnn_model()
    trained_model, history = train_model(model, train_ds, val_ds, model_name)
    evaluate_model(trained_model, test_ds)


if __name__ == "__main__":
    main()
