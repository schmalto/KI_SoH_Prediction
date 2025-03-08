from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, Input, BatchNormalization, GRU
import tensorflow as tf
import os
from tuner_helpers import tuner_start, load_best_trials


os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_model(hp):
    inputs = Input(shape=(9940, 1))
    x = inputs
    # Build a lstm model with tunable hyperparameters
    for i in range(hp.Int('lstm_layers', 1, 4)):
        x = GRU(hp.Int(f'lstm_{i}_units', 1, 10, step=1), return_sequences=True)(x)
        if hp.Boolean(f'add_batchnorm_{i}', default=True):
            x = BatchNormalization()(x)
        x = Dropout(
            hp.Float(f'dropout_{i}', min_value=0.05, max_value=0.5, step=0.1))(x)
    x = GRU(hp.Int('lstm_final_units', 1, 10, step=1), return_sequences=True)(x)
    predictions = Dense(1)(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
    model.summary()
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model




def main():
    project_names = ['cycle_lstm']
    tuner_start(project_names[0], build_model)
    # load_best_trials()


if __name__ == '__main__':
    main()
