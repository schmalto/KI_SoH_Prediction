from keras.layers import Dense, Dropout, Flatten, Conv1D, MaxPool1D, Input, BatchNormalization
import keras
import tensorflow as tf
import os
import keras_tuner as kt
from  model.model_train_helpers import generate_training_data

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def build_model(hp):
    inputs = Input(shape=(9940, 1))
    x = inputs
    for i in range(hp.Int('conv_layers', 1, 4)):
        x = Conv1D(hp.Int(f'conv_{i}_units', 16, 128, step=16), 3, activation='relu', strides=(1))(x)
        if hp.Boolean(f'add_batchnorm_{i}', default=True):
            x = BatchNormalization()(x)
        if hp.Boolean(f'add_maxpool_{i}', default=True):
            x = MaxPool1D(2, name=f'conv_{i}')(x)
        x = Dropout(
                hp.Float(f'dropout_{i}', min_value=0.05, max_value=0.5, step=0.1))(x)
    for i in range(hp.Int('conv_layers_2', 1, 4)):
        x = Conv1D(hp.Int(f'conv_2_{i}_units', 8, 32,
                       step=8), 3, activation='relu', strides=(1))(x)
        x = Dropout(
            hp.Float(f'dropout_{i}', min_value=0.1, max_value=0.5, step=0.1))(x)

    x = Flatten()(x)
    for i in range(hp.Int('dense_layers', 1, 3)):
        x = Dense(hp.Int(f'dense_{i}_units', 32, 128, step=16), activation='relu')(x)
        x = Dense(hp.Int(f'dense_{i}_2_units', 32, 128,
                  step=16), activation='relu')(x)
        x = Dropout(hp.Float(f'dropout_dense_{i}', min_value=0.05, max_value=0.5, step=0.1))(x)
        x = Dense(hp.Int(f'dense_{i}_3_units', 8, 32,
                  step=8), activation='relu')(x)
    predictions = Dense(1)(x)
    model = tf.keras.models.Model(inputs=inputs, outputs=predictions)
    model.summary()
    model.compile(optimizer='adam', loss='mean_absolute_error')
    return model


def generate_tuner(project_name, overwrite=False):
    tuner = kt.GridSearch(
        build_model,
        objective='val_loss',
        max_trials=1000,
        executions_per_trial=2,
        directory='src/model/tuner',
        project_name=project_name,
        overwrite=overwrite
    )
    return tuner


def search_best_model(tuner, train_ds, val_ds):
    tuner.search_space_summary()
    tuner.search(train_ds, validation_data=val_ds, epochs=10)
    best_models = tuner.get_best_models()[:5]
    for model in best_models:
        model.summary()
    return model

def load_best_trials(project_name):
    tuner = generate_tuner(project_name, overwrite=False)
    tuner.results_summary()
    return 0

def tuner_start(project_name):
    train_ds, val_ds, _ = generate_training_data()
    tuner = generate_tuner(project_name)
    best_model = search_best_model(tuner, train_ds, val_ds)
    return best_model


def main():
    project_names = ['tune_model', 'tune_model_2', 'tune_model_3', 'augmented_data', 'cycle_to_break_1', 'cycle_to_break_2']
    tuner_start('cycle_to_break_2')
    # load_best_trials()


if __name__ == '__main__':
    main()
