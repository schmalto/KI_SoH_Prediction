from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, Input, BatchNormalization, LSTM, GlobalAveragePooling1D, Concatenate
import keras
import tensorflow as tf
import os
from model_train_helpers import PlotLossAccuracy, evaluate_model, generate_training_data, modelCheckpoint, generate_combined_dataset
from tensorflow.keras import mixed_precision

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from matplotlib.ticker import MaxNLocator
# from matplotlib import pyplot as plt


def create_cnn_model():
   # Branch 1: Rds(on) time-series input
    input_rdson = Input(shape=(20007, 1), name='rdson_time_series')
    x = BatchNormalization()(input_rdson)
    x = Conv1D(16, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(2)(x)
    x = Conv1D(128, 2, activation='relu')(x)
    x = MaxPool1D(2)(x)
    x = BatchNormalization()(x)
    x = Conv1D(128, 2, activation='relu')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(4)(x)
    x = Flatten()(x)
    x = Dense(32, activation='relu')(x)

    # Branch 2: Cooling input
    input_cooling = Input(shape=(1,), name='cooling')
    y = Dense(64, activation='relu')(input_cooling)
    y = Dense(32, activation='relu')(y)

    # Combine branches
    z = Concatenate()([x, y])
    z = Dense(64, activation='relu')(z)
    predictions = Dense(1)(z)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model = tf.keras.models.Model(
        inputs=[input_rdson, input_cooling], outputs=predictions)
    model.summary()
    model.compile(optimizer=optimizer, loss='mean_squared_error')
    return model


def train_model(model, train_ds, val_ds, model_name):
    plotLoss = PlotLossAccuracy(model_name)
    checkpoint = modelCheckpoint(model_name)
    n_epochs = 500
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=n_epochs, batch_size=8,
                        callbacks=[early_stopping, reduce_lr, plotLoss, checkpoint])
    return model, history


def main():
    model_name = "wo_cooling_model_1"
    # Ensure that generate_training_data includes data normalization
    train_ds, val_ds, test_ds = generate_combined_dataset(
        label_type="to_break",
        feature_1="wocooling",
        feature_2="cooling",
    )

    model = create_cnn_model()
    trained_model, history = train_model(model, train_ds, val_ds, model_name)
    evaluate_model(trained_model, test_ds)


if __name__ == "__main__":
    main()
