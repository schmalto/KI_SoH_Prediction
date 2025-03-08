from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, Input, BatchNormalization
import keras
import tensorflow as tf
import os
from src.model.model_train_helpers import PlotLossAccuracy, evaluate_model, generate_training_data, modelCheckpoint

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# from matplotlib.ticker import MaxNLocator
# from matplotlib import pyplot as plt


def create_cnn_model():
    inputs = Input(shape=(1 ,))
    x = inputs
    # Normalize the input feature
    # Reduced number of layers and units
    x = Dense(4096, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(256, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.3)(x)
    x = Dense(128, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dense(32, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dense(8, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    predictions = Dense(1)(x)
    model = Model(inputs=inputs, outputs=predictions)
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
                  loss='mean_squared_error')
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
    model_name = "cooling_branch_1"
    # Ensure that generate_training_data includes data normalization
    train_ds, val_ds, test_ds = generate_training_data(
        feature_type="cooling",
        label_type="to_break",
    )
    model = create_cnn_model()
    trained_model, history = train_model(model, train_ds, val_ds, model_name)
    evaluate_model(trained_model, test_ds)


if __name__ == "__main__":
    main()
