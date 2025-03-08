from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, Input, BatchNormalization, LSTM, GlobalAveragePooling1D, Concatenate
import keras
import tensorflow as tf
import os
from model_train_helpers import PlotLossAccuracy, evaluate_model, generate_training_data, modelCheckpoint, generate_combined_dataset

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from matplotlib.ticker import MaxNLocator
# from matplotlib import pyplot as plt


def create_cnn_model():
    input_vth_down = Input(shape=(1994, 1), name='vth_down')
    x = Conv1D(16, kernel_size=3, activation='relu',
               padding='same')(input_vth_down)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = MaxPool1D(pool_size=4)(x)
    x = Conv1D(512, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    x = Conv1D(1024, kernel_size=3, activation='relu', padding='same')(x)
    # x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = MaxPool1D(pool_size=8)(x)
    x = Flatten()(x)

    input_vth_up = Input(shape=(991, 1), name='vth_up')
    z = Conv1D(16, kernel_size=3, activation='relu',
               padding='same')(input_vth_up)
    z = BatchNormalization()(z)
    z = Conv1D(32, kernel_size=3, activation='relu', padding='same')(z)
    z = BatchNormalization()(z)
    z = Conv1D(64, kernel_size=3, activation='relu', padding='same')(z)
    z = Conv1D(128, kernel_size=3, activation='relu', padding='same')(z)
    z = BatchNormalization()(z)
    z = Conv1D(256, kernel_size=3, activation='relu', padding='same')(z)
    z = BatchNormalization()(z)
    z = MaxPool1D(pool_size=4)(z)
    z = Conv1D(512, kernel_size=3, activation='relu', padding='same')(z)
    z = BatchNormalization()(z)
    z = Conv1D(1024, kernel_size=3, activation='relu', padding='same')(z)
    # x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    z = MaxPool1D(pool_size=8)(z)
    z = Flatten()(z)

    # x = GlobalAveragePooling1D()(x)

    # Combine all branches
    # combined = Dense(256, activation='relu')(x)
    # combined = Dropout(0.4)(combined)
    combined = Concatenate()([x, z])
    combined = Dense(64, activation='relu')(combined)
    combined = Dropout(0.4)(combined)
    predictions = Dense(1)(combined)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model = tf.keras.models.Model(
        inputs=[input_vth_down, input_vth_up], outputs=predictions)
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
    model_name = "vth_up_down"
    # Ensure that generate_training_data includes data normalization
    train_ds, val_ds, test_ds = generate_combined_dataset(
        "to_break",
        "vth_down",
        "vth_up",
    )
    model = create_cnn_model()
    trained_model, history = train_model(model, train_ds, val_ds, model_name)
    evaluate_model(trained_model, test_ds)


if __name__ == "__main__":
    main()
