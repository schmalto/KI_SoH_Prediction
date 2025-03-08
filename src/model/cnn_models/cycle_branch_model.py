import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, Input, BatchNormalization, LSTM, GlobalAveragePooling1D, Concatenate
import keras
import tensorflow as tf
from src.model.model_train_helpers import PlotLossAccuracy, evaluate_model, generate_training_data, modelCheckpoint

# from matplotlib.ticker import MaxNLocator
# from matplotlib import pyplot as plt


def create_cnn_model():
    input_rdson = Input(shape=(100, 1), name='rdson_time_series')
    x = Conv1D(16, kernel_size=3, activation='relu',
               padding='same')(input_rdson)
    x = BatchNormalization()(x)
    x = Conv1D(32, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    # x = MaxPool1D(pool_size=2)(x)
    x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    x = Conv1D(128, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    # x = MaxPool1D(pool_size=4)(x)
    x = Conv1D(256, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    # x = MaxPool1D(pool_size=4)(x)
    x = Conv1D(512, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    # x = Conv1D(1024, kernel_size=3, activation='relu', padding='same')(x)
    # x = MaxPool1D(pool_size=2)(x)
    # x = Conv1D(2048, kernel_size=3, activation='relu', padding='same')(x)
    x = BatchNormalization()(x)
    # x = MaxPool1D(pool_size=2)(x)
    # x = Conv1D(64, kernel_size=3, activation='relu', padding='same')(x)
    # x = MaxPool1D(pool_size=8)(x)
    x = GlobalAveragePooling1D()(x)
    # x = LSTM(256, return_sequences=True)(x)
    # x = LSTM(128, return_sequences=False)(x)
    # x = LSTM(128, return_sequences=False)(x)

    # x = GlobalAveragePooling1D()(x)

    # Combine all branches
    # combined = Dense(256, activation='relu')(x)
    # combined = Dropout(0.4)(combined)
    combined = Dense(128, activation='relu')(x)
    combined = Dropout(0.4)(combined)
    predictions = Dense(1)(combined)

    optimizer = tf.keras.optimizers.Adam(learning_rate=0.01)
    model = tf.keras.models.Model(
        inputs=input_rdson, outputs=predictions)
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
    model_name = "cycle_branch_model_test_sample_random"
    # Ensure that generate_training_data includes data normalization
    train_ds, val_ds, test_ds = generate_training_data(
        feature_type="sampled",
        label_type="random",
    )
    model = create_cnn_model()
    trained_model, history = train_model(model, train_ds, val_ds, model_name)
    evaluate_model(trained_model, test_ds)


if __name__ == "__main__":
    main()
