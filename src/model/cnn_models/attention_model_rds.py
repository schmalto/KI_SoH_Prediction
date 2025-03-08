import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # noqa
import tensorflow as tf
from tensorflow.keras.layers import (
    Input, Dense, Flatten, MultiHeadAttention,
    LayerNormalization, Dropout, Concatenate, Reshape, Add, Lambda, Conv1D  
)
from tensorflow.keras.models import Model
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
from src.model.model_train_helpers import (
    generate_combined_dataset,
    train_model,
    PlotLossAccuracy,
    modelCheckpoint
)


def transformer_model():
    input_dim = 100
    d_model = 64  # Dimension of the embedding space

    # Define inputs
    input_sequence = Input(shape=(input_dim,), name='input_sequence')
    vth = Input(shape=(input_dim, 1), name='vth')
    cooling = Input(shape=(1,), name='cooling')

    # Reshape input_sequence
    x = Reshape((input_dim, 1))(input_sequence)
    x = Dense(d_model)(x)

    # Positional encoding
    positions = tf.range(start=0, limit=input_dim, delta=1)
    position_embedding = tf.keras.layers.Embedding(input_dim=input_dim, output_dim=d_model)
    position_encoded = position_embedding(positions)

    # Adding positional encoding to16 the input within a Lambda layer
    def add_position_encoding(inputs):
        x, pos_enc = inputs
        pos_enc = tf.expand_dims(pos_enc, axis=0)  # Shape (1, input_dim, d_model)
        pos_enc = tf.tile(pos_enc, [tf.shape(x)[0], 1, 1])  # Tile to match batch size
        return x + pos_enc

    x = Lambda(add_position_encoding)([x, position_encoded])

    # Transformer Encoder Block for x
    attention_output = MultiHeadAttention(num_heads=8, key_dim=d_model)(x, x)
    x = Add()([x, attention_output])
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Dense(d_model, activation='relu')(x)
    x = LayerNormalization(epsilon=1e-6)(x)
    x = Flatten()(x)


    # Fully connected layers
    combined = Dense(128, activation='relu')(x)
    combined = Dense(64, activation='relu')(combined)

    # Output layer
    pred = Dense(1)(combined)

    # Define the model
    model = Model(inputs=[input_sequence], outputs=pred)
    mape = tf.keras.losses.MeanAbsolutePercentageError()

    model.compile(optimizer='adam', loss=mape)

    return model


def train_model(model, train_ds, val_ds, model_name):
    plotLoss = PlotLossAccuracy(model_name)
    checkpoint = modelCheckpoint(model_name)
    n_epochs = 500
    early_stopping = EarlyStopping(
        monitor='val_loss', patience=20, restore_best_weights=True)
    # reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=10)
    history = model.fit(train_ds, validation_data=val_ds,
                        epochs=n_epochs, batch_size=8,
                        callbacks=[early_stopping, plotLoss, checkpoint])
    return model, history


def setup_model():
    model_name = "attention_model_rdson"
    # Ensure that generate_training_data includes data normalization
    train_ds, val_ds, test_ds = generate_combined_dataset(("rdson_sampled_all",))
    model = transformer_model()
    trained_model, history = train_model(model, train_ds, val_ds, model_name)


def main():
    setup_model()


if __name__ == '__main__':
    main()
