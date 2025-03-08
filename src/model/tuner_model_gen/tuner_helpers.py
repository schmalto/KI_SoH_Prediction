from model.model_train_helpers import generate_training_data
import keras_tuner as kt


def generate_tuner(project_name, build_model, overwrite=False):
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


def tuner_start(project_name, build_model):
    train_ds, val_ds, _ = generate_training_data()
    tuner = generate_tuner(project_name, build_model)
    best_model = search_best_model(tuner, train_ds, val_ds)
    return best_model
