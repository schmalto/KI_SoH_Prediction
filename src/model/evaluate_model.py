from src.model.model_train_helpers import plot_model_predictions



def main():
    plot_model_predictions("multi_input_random_samples_complex_all_n100_2", ["rdson_sampled_all", "cooling_to_break"], "DUT_88")


if __name__ == '__main__':
    main()
