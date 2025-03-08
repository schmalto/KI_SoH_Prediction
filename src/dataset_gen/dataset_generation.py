import os
import cupy as cp
import numpy as np
from matplotlib import pyplot as plt
from tqdm import tqdm
import glob
from pathlib import Path
from sklearn.preprocessing import minmax_scale
from termcolor import colored
from src.dataset_gen.plot_seq import plot_data
import random
import time

# Define base directory relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_DIR = os.path.join(BASE_DIR, "data")
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")
SRC_PLOTS_DIR = os.path.join(BASE_DIR, "src", "plots")

# Constants for sampling
DEFAULT_SAMPLE_SIZE = 50
VISUALIZATION_ENABLED = False


def get_all_folders(base_dir=None):
    base_dir = base_dir or DATA_DIR
    folders = glob.glob(base_dir + '/*/')
    folders = [os.path.basename(os.path.normpath(folder))
               for folder in folders]
    return folders


def format_curves_max(files):
    files_sorted = sorted(
        files,
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])
    )
    vth = [np.load(file) for file in files_sorted]
    ret_vth = np.concatenate(vth, axis=0)
    return ret_vth


def format_curves(files):

    vth = []

    # Sort files based on the integer suffix in the filename
    files_sorted = sorted(
        files,
        key=lambda x: int(os.path.splitext(os.path.basename(x))[0].split('_')[-1])
    )

    min_size = np.inf  # Initialize minimum size to infinity
    n_files = len(files_sorted)

    for idx, file in enumerate(files_sorted):
        # Load the data as a CuPy array
        try:
            data = cp.load(file)
            # data = np.asnumpy(data)
        except Exception as e:
            print(f"Error loading {file}: {e}")
            continue
        if idx == 0:
            data_shape = data.shape[0]
        # Update min_size based on all files except the last one
        if idx != n_files - 1:
            if data.shape[1] < min_size:
                min_size = data.shape[1]
        else:
            # For the last file, pad or truncate to match min_size
            if data.shape[1] < min_size:
                pad_width = ((0, 0), (0, min_size - data.shape[1]))
                data = cp.pad(data, pad_width, 'constant', constant_values=0)
            else:
                data = data[:, :min_size]
        # Ensure the first dimension is data_shape by padding or truncating
        if data.shape[0] < data_shape:
            pad_len = data_shape - data.shape[0]
            data = cp.pad(data, ((0, pad_len), (0, 0)), 'constant', constant_values=0)
        elif data.shape[0] > data_shape:
            data = data[:data_shape, :]

        if data.shape[1] > min_size:
            data = data[:, :min_size]
        elif data.shape[1] < min_size:
            pad_width = ((0, 0), (0, min_size - data.shape[1]))
            data = cp.pad(data, pad_width, 'constant', constant_values=0)

        vth.append((data))

    # Slice each curve to min_size and collect into temp_curve
    temp_curve = [el[:, :min_size] for el in vth]

    # Concatenate all curves along the first axis
    if temp_curve:
        ret_vth = cp.concatenate(temp_curve, axis=0)
    else:
        ret_vth = cp.array([])

    # Convert the final CuPy array back to a NumPy array before returning
    return cp.asnumpy(ret_vth)


def get_all_cooling(folder):
    files = glob.glob(folder + '/*cooling_*.npy')
    cooling = format_curves(files)
    return cooling


def get_all_vdson(folder):
    files = glob.glob(folder + '/*vdson_*.npy')
    vdson = format_curves(files)
    print(vdson.shape)
    return vdson


def get_all_vth(folder, vth_type):
    if vth_type not in ['up', 'down']:
        raise ValueError('Invalid vth type')
    files = glob.glob(folder + '/*vth_' + str(vth_type) + '_*.npy')
    vth = format_curves(files)
    return vth


def get_all_complete(folder):
    files = glob.glob(folder + '/*complete_*.npy')
    complete = format_curves(files)
    return complete


def get_all_wo_cooling(folder):
    files = glob.glob(folder + '/*wocooling_*.npy')
    wo_cooling = format_curves(files)
    return wo_cooling


def get_all_types(folder, pattern):
    files = glob.glob(folder + f'/*{pattern}_*.npy')
    ret = format_curves_max(files)
    return ret


def get_all_rdson(folder):
    files = glob.glob(folder + '/*rdson_*.npy')
    ret_rdson = format_curves(files)
    print(ret_rdson.shape)

    return ret_rdson


def rolling_mean_filter(x, window_size):
    """
    :param x: Array to filter
    :param window_size: Window size of rolling average
    :return: filtered array
    """
    z = np.convolve(x, np.ones(window_size) / window_size, mode='valid')
    return z


def calc_rul(p_cycles, low=False):
    """
    Calculation of the RUL curve based on the rdson LOW values.
    :param p_cycles: list of cycles with rdson values
    :param low: if True, the LOW values are used for the calculation, else the peak values are used
    :return: list of cycles of RUL values
    """

    # if low:
    #     rdson = get_lows(p_cycles, n_window=50)
    # else:
    #    rdson = get_peaks(p_cycles, n_window=50)
    average_window = p_cycles[2000][15:20]
    full_RUL = np.average(average_window)
    broken_RUL = full_RUL * 5  # 20% increase in RUL ^= broken device
    n_cycles = normalize_rul(p_cycles, full_RUL, broken_RUL)
    f_cycles = np.where(n_cycles > 1, 1, n_cycles)
    f_cycles = np.where(f_cycles < 0, 0, f_cycles)
    try:
        first_one = f_cycles.tolist().index(1)
    except ValueError:
        print(colored('No broken RUL found in this dataset', 'red'))
        return f_cycles
    first_one = f_cycles.tolist().index(1)
    f_cycles[first_one:] = 1
    return f_cycles


def normalize_rul(p_data, p_full_rul, p_broken_rul):
    """
    Normalizes the RUL values between 0 and 1 based on the full and broken RUL values.
    :param p_data: data to normalize
    :param p_full_rul: value of full RUL
    :param p_broken_rul: value of broken RUL
    :return: normalized data
    """
    scale = p_broken_rul - p_full_rul
    n_data = np.array(((p_data - p_full_rul) / scale))
    return n_data


def get_peaks(p_data, n_window=10):
    """
    Calculate the peaks of the rdson curve
    :param p_data: rdson curve
    :return: peaks of the curve
    """
    peaks = np.array([])
    for i in range(len(p_data)):
        max_values = np.argsort(p_data[i])[-n_window:]
        avg = np.average(p_data[i][max_values])
        peaks = np.append(peaks, avg)
    return np.array(peaks)


def get_lows(p_data, n_window=10):
    """
    Calculate the lows of the rdson curve
    :param p_data: rdson curve
    :return: lows of the curve
    """
    lows = np.array([])
    for i in range(len(p_data)):
        min_values = np.argsort(p_data[i])[:n_window]
        avg = np.average(p_data[i][min_values])
        lows = np.append(lows, avg)
    return np.array(lows)


def generate_tim_straight(p_data, startpoint=175):
    tim_curve = p_data[startpoint:]
    time_axis = np.sqrt(np.arange(len(tim_curve)))
    A = np.vstack([time_axis, np.ones(len(time_axis))]).T
    m, c = np.linalg.lstsq(A, tim_curve, rcond=None)[0]
    tim_straight = m * time_axis + c
    return tim_straight


def generate_dut_straight(p_data, endpoint=50):
    dut_curve = p_data[:endpoint]
    time_axis = np.sqrt(np.arange(len(dut_curve)))
    A = np.vstack([time_axis, np.ones(len(time_axis))]).T
    m, c = np.linalg.lstsq(A, dut_curve, rcond=None)[0]
    dut_straight = m * time_axis + c
    return dut_straight


def generate_cooling_curve_straight(p_data):
    all_temperatures = []
    for el in p_data[::1]:
        tim = generate_tim_straight(el)
        dut = generate_dut_straight(el)
        # x_axis = np.sqrt(np.arange(0, len(el), 1))
        # plt.plot(x_axis, el)
        # plt.plot(x_axis[-len(tim):], tim, '--', label='TIM', linewidth=2, color='r')
        # plt.plot(x_axis[:len(dut)], dut, '--', label='DUT', linewidth=2, color='y')
        # plt.xlabel('$\sqrt{\mathrm{t}}$ [$\sqrt{\mathrm{ms}}$ ]')
        # plt.ylabel('$V_{\mathrm{ds}}$ $[ V ]$')
        # plt.legend()
        # plt.savefig(os.path.join(SRC_PLOTS_DIR, 'cooling_curve.pdf'))
        # plt.show()
        all_temperatures.append(dut[0] - tim[-1])
    temperature_delta = np.array(all_temperatures) / 0.0025
    return temperature_delta


def show_data_with_cycle_labels(p_data, cycle_length, n=5):
    start = time.time()
    data = p_data[::100]
    data_smoothed = rolling_mean_filter(data, cycle_length)
    smoothed_init_value = data_smoothed[cycle_length * 50]
    # smoothed_brake_value = smoothed_init_value * 1.2
    smoothed_brake_value = 2.5
    breaking_index = np.argmax(data_smoothed > smoothed_brake_value)

    breaking_point = int((breaking_index / (cycle_length)) * 100)
    # x = np.arange(0, len(data))[:2700000]
    # x = (x / (cycle_length*2)) * 100
    # plt.plot(x, data_smoothed[:2700000])
    # plt.axvline(x=breaking_point, color='r')
    # plt.ylabel('$R_{\mathrm{DS,on}}$ $[ \Omega ]$')
    # plt.xlabel('Cycle')
    # plt.grid()
    # plt.savefig(os.path.join(PLOTS_DIR, 'smoothed_data.pdf'))
    return breaking_point


def get_breaking_cycle():
    """
    Let the user pick the breaking cycle from the plot.
    """
    print('Please select the breaking cycle')
    breaking_cycle = int(input())
    # breaking_cycle = 7931
    return breaking_cycle


def broken_labels(p_data):
    """
    Show the data as a plot
    Let the user pick the breaking cycle
    Every cycle before the breaking cycle is labeled as the difference to the breaking cycle
    Every cycle after the breaking cycle is labeled as the negative difference to the breaking cycle
    :param p_data: rdson curve
    :return: labels for the data
    """
    cycle_length = p_data.shape[1]
    breaking_cycle = show_data_with_cycle_labels(
        np.concatenate(p_data, axis=0), cycle_length)
    print("Breaking: ", breaking_cycle)
    if breaking_cycle == -1:
        print('No breaking cycle found')
        return cp.zeros(len(p_data))

    labels = breaking_cycle - cp.arange(len(p_data))
    return cp.asnumpy(labels)


def broken_labels_seq(p_data, sequence_length, features):
    cycle_length = p_data.shape[0]
    breaking_cycle = show_data_with_cycle_labels(np.concatenate(
        p_data, axis=0), sequence_length * cycle_length, n=1)
    # breaking_cycle = get_breaking_cycle()
    if breaking_cycle == -1:
        print('No breaking cycle found')
        return np.zeros(len(features))
    labels = np.zeros(len(features))
    for i in range(len(features)):
        # if i < breaking_cycle:
        labels[i] = breaking_cycle - i
        # else:
        #    labels[i] = i - breaking_cycle
    return labels


def sampling(cycles, method):
    """
    Sample time series data using different sampling strategies.

    Args:
        cycles: Time series data with shape (n_cycles, cycle_length)
        method: Sampling method, one of ['beginning', 'end', 'random', 'all', 'evenly', 'trng']

    Returns:
        Sampled time series with reduced dimensionality
    """
    if len(cycles) == 0:
        return np.array([])

    # Get cycle length from the data
    cycle_length = cycles.shape[1] if cycles.shape[1] > 0 else 0
    sample_size = DEFAULT_SAMPLE_SIZE

    # Pre-allocate array for true random sampling ('trng')
    new_cycles = np.zeros((len(cycles), sample_size))

    try:
        if method == 'beginning':
            # Sample points from beginning and end of cycles
            start_point = max(1, int(cycle_length * 0.01))
            start_end = max(start_point + 1, int(cycle_length * 0.1))
            end_point = min(cycle_length - 1, int(cycle_length * 0.99))
            end_start = max(start_end, min(end_point, int(cycle_length * 0.9)))

            begin = cycles[:, start_point:start_end]
            end = cycles[:, end_start:end_point]
            return np.concatenate((begin, end), axis=1)

        elif method == 'end':
            # Sample points from the end of cycles
            end_size = min(sample_size, cycle_length)
            end_start = max(0, cycle_length - end_size)
            return cycles[:, end_start:cycle_length]

        elif method == 'random':
            # Random sampling from beginning and end of cycles
            half_size = sample_size // 2

            # Sample from beginning
            begin_indices = random.sample(
                range(0, max(1, int(cycle_length * 0.2))),
                min(half_size, max(1, int(cycle_length * 0.2)))
            )
            begin_indices = np.sort(np.array(begin_indices))

            # Sample from end
            end_indices = random.sample(
                range(min(cycle_length-1, int(cycle_length * 0.8)), cycle_length),
                min(half_size, cycle_length - int(cycle_length * 0.8))
            )
            end_indices = np.sort(np.array(end_indices))

            return np.concatenate((cycles[:, begin_indices], cycles[:, end_indices]), axis=1)

        elif method == 'all':
            # Random sampling across entire cycle
            indices = random.sample(
                range(0, cycle_length),
                min(sample_size, cycle_length)
            )
            indices = np.sort(np.array(indices))
            return cycles[:, indices]

        elif method == 'evenly':
            # Evenly spaced samples
            indices = np.linspace(0, cycle_length-1, sample_size, endpoint=True, dtype=int)
            return cycles[:, indices]

        elif method == 'trng':
            # True random sampling - different for each cycle
            for i, cycle in enumerate(cycles):
                indices = random.sample(
                    range(0, cycle_length),
                    min(sample_size, cycle_length)
                )
                new_cycles[i] = cycle[np.array(indices)]
            return new_cycles

        else:
            raise ValueError(f"Invalid sampling method: {method}")

    except Exception as e:
        print(colored(f"Error in sampling method '{method}': {str(e)}", "red"))
        # Fall back to evenly spaced sampling
        indices = np.linspace(0, cycle_length-1, sample_size, endpoint=True, dtype=int)
        return cycles[:, indices]


def plot_sampling_visualization(original_cycle, sampled_indices, method):
    """
    Visualize the sampling method by showing which points were selected.

    Args:
        original_cycle: The original time series cycle
        sampled_indices: The indices that were sampled
        method: The sampling method name
    """
    if not VISUALIZATION_ENABLED:
        return

    plt.figure(figsize=(12, 6))

    x_axis = np.arange(len(original_cycle)) / 10
    plt.plot(x_axis, original_cycle, color='k', label='Original')

    # Create mask for sampled points
    mask = np.zeros_like(original_cycle, dtype=bool)
    mask[sampled_indices] = True

    # Plot sampled points
    plt.scatter(
        x_axis[mask],
        original_cycle[mask],
        color='lime',
        marker='*',
        s=150,
        label='Sampled'
    )

    plt.title(f'Sampling Method: {method}')
    plt.ylabel('$R_\mathrm{DS,on}$ [$\Omega$]')
    plt.xlabel('Time [ms]')
    plt.legend()
    plt.grid()

    save_path = os.path.join(SRC_PLOTS_DIR, f'cycle_sampled_{method}.pdf')
    plt.savefig(save_path)
    plt.close()


def generate_sampled_set(folder, feature_type='rdson', label_type='to_break',
                         sampling_method='beginning', max_data=False):
    """
    Generate a dataset with sampled features from the given folder.

    Args:
        folder: Source folder for data
        feature_type: Type of feature to extract ('rdson', 'vdson', 'vth', etc.)
        label_type: Type of label to generate ('to_break', 'cycle')
        sampling_method: Method to use for sampling
        max_data: Whether to use maximum data values

    Returns:
        tuple: (sampled_cycles, labels) as numpy arrays
    """
    # Dictionary to map feature types for consistent handling
    feature_type_map = {
        'vth': 'vth_up',  # Map 'vth' to 'vth_up' for backwards compatibility
    }

    # Get the mapped feature type or use the original
    mapped_feature_type = feature_type_map.get(feature_type, feature_type)

    # Check for unsupported feature types
    unsupported_types = ['vth_down', 'cooling']
    if feature_type in unsupported_types:
        raise ValueError(f"Unsupported feature type: {feature_type}")

    # Generate the dataset
    cycles, labels = generate_set(
        folder,
        p_feature_type=mapped_feature_type,
        p_label_type=label_type,
        max_data=max_data
    )

    # Apply sampling if we have data
    if len(cycles) > 0:
        sampled_cycles = sampling(cycles, sampling_method)
        return sampled_cycles, labels
    else:
        print(colored(f"No data found for {feature_type} in {folder}", "yellow"))
        return np.array([]), labels


def generate_max_data(p_feature_type):
    """
    Generate maximum data for a specific feature type.

    Args:
        p_feature_type: Feature type to extract

    Returns:
        tuple: (features, labels) as numpy arrays
    """
    folders = os.path.join(DATA_DIR, 'MaxData')
    features = get_all_types(folders, p_feature_type)
    labels = get_all_types(folders, 'norm_deg')
    return features, labels


def generate_set(p_folder, p_offset=0, p_feature_type='rdson', p_label_type='to_break', max_data=False):
    """
    Generate a dataset from the given folder.

    Args:
        p_folder: Folder containing the source data
        p_offset: Offset for cycle counting
        p_feature_type: Feature type to extract
        p_label_type: Label type to generate
        max_data: Whether to use maximum data values

    Returns:
        tuple: (features, labels) as numpy arrays
    """
    if max_data:
        features, labels = generate_max_data(p_feature_type)
        return features, labels

    # Validate feature and label types
    valid_feature_types = ['rdson', 'rul', 'cooling', 'cooling_curve', 'seq',
                           'vth_up', 'vth_down', 'wocooling', 'complete', 'vdson']
    if p_feature_type not in valid_feature_types:
        raise ValueError(f'Invalid feature type: {p_feature_type}. '
                         f'Must be one of {valid_feature_types}')

    valid_label_types = ['to_break', 'cycle']
    if p_label_type not in valid_label_types:
        raise ValueError(f'Invalid label type: {p_label_type}. '
                         f'Must be one of {valid_label_types}')

    # Load VDSon data as a common reference
    cycles = get_all_vdson(p_folder)
    features = []

    # Extract features based on type
    if p_feature_type == 'cooling':
        cooling_curves = get_all_cooling(p_folder)
        features = generate_cooling_curve_straight(cooling_curves)
    elif p_feature_type == 'rul':
        raise ValueError('RUL not implemented')
    elif p_feature_type == 'rdson':
        # Convert from CuPy to NumPy if needed
        try:
            cycles_normalized = minmax_scale(cycles.get(), feature_range=(0, 1))
        except AttributeError:
            # If cycles is already a NumPy array
            pass
        features = cycles
    elif p_feature_type == 'wocooling':
        features = get_all_wo_cooling(p_folder)
    elif p_feature_type == 'vth_up':
        features = get_all_vth(p_folder, 'up')
    elif p_feature_type == 'vth_down':
        features = get_all_vth(p_folder, 'down')
    elif p_feature_type == 'cooling_curve':
        cooling_curves = get_all_cooling(p_folder)
        features = cooling_curves
    elif p_feature_type == 'vdson':
        features = get_all_vdson(p_folder)
    elif p_feature_type == 'seq':
        sequence_length = 20
        features = [cycles[i:i+sequence_length].flatten()
                    for i in range(0, len(cycles), sequence_length)]
        if len(features) > 1 and features[-1].shape != features[-2].shape:
            features[-1] = np.pad(
                features[-1],
                (0, features[-2].shape[0] - features[-1].shape[0]),
                'edge'
            )

        # For seq feature type, the labels are initially just sequential indices
        labels = np.arange(1, len(features))

    # Generate labels based on type
    if p_label_type == 'to_break':
        if p_feature_type == 'seq':
            labels = broken_labels_seq(cycles, sequence_length, features)
            idx = np.where(labels > 0)[0]
        else:
            labels = broken_labels(cycles)
            features = np.array(features)
            idx = np.where(labels > 0)[0]

        # Filter to only include positive labels
        features = np.array(features)
        if len(idx) > 0:  # Only filter if we found positive labels
            features = features[idx]
            labels = labels[idx]
    elif p_label_type == 'cycle':
        start = p_offset
        end = len(features) + p_offset
        labels = np.arange(start, end)

    return features, labels


def generate_dataset_numpy(p_folder, base_dir=None, offset=0, feature_type='rdson',
                           label_type='to_break', sampling=False, sampling_type='beginning',
                           max_data=False):
    """
    Generate and save a dataset in NumPy format.

    Args:
        p_folder: Source folder name
        base_dir: Base directory for source data
        offset: Offset for cycle counting
        feature_type: Type of feature to extract
        label_type: Type of label to generate
        sampling: Whether to apply sampling
        sampling_type: Method to use for sampling
        max_data: Whether to use maximum data values

    Returns:
        int: 0 on success
    """
    base_dir = base_dir or DATA_DIR
    load_folder = os.path.join(base_dir, p_folder)

    # Determine output folder path
    if sampling:
        temp = f'{feature_type}_sampled_{sampling_type}/'
        save_folder = os.path.join(DATASETS_DIR, temp, p_folder)
    else:
        save_folder = os.path.join(DATASETS_DIR, f'{feature_type}_{label_type}', p_folder)

    # Create output directory if it doesn't exist
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    try:
        # Generate dataset with or without sampling
        if sampling:
            cycles, labels = generate_sampled_set(
                load_folder,
                feature_type=feature_type,
                label_type=label_type,
                sampling_method=sampling_type,
                max_data=max_data
            )
        else:
            cycles, labels = generate_set(
                load_folder,
                p_offset=offset,
                p_feature_type=feature_type,
                p_label_type=label_type,
                max_data=max_data
            )

        # Save dataset
        cycles_path = os.path.join(save_folder, 'cycles.npy')
        labels_path = os.path.join(save_folder, 'labels.npy')

        np.save(cycles_path, cycles)
        np.save(labels_path, labels)

        print(colored(f"Dataset saved to {save_folder}", "green"))
    except Exception as e:
        print(colored(f"Error generating dataset: {e}", "red"))

    return 0


def generate_all_feature_sets(base_dir=None, feature_type='rdson'):
    """
    Generates all datasets for all devices
    :param base_dir: Data directory, where ALL raw data is located
    :param feature_type: rdson or rul
    :return:
    """
    base_dir = base_dir or DATA_DIR
    folders = get_all_folders(base_dir=base_dir)
    for folder in tqdm(folders):
        if not os.listdir(os.path.join(base_dir, folder)):
            continue
        else:
            if folder == 'DUT_TS':
                generate_dataset_numpy(
                    folder, offset=1023, feature_type=feature_type)
                continue
            generate_dataset_numpy(folder, offset=0, feature_type=feature_type)


def normalized_data(p_data):
    data = np.array(p_data)
    maximum = np.max(data)
    if maximum == 0:
        return data
    data_norm = data / maximum
    return data_norm


def flatten_1d(data):
    """
    Little helper function to flatten 2D arrays with variable size TODO: Improve performance
    :param data:
    :return:
    """
    ret = []
    for (i, x) in enumerate(data):
        ret.append(x)
    return ret


def test_load():
    data = np.load(os.path.join(DATASETS_DIR, "rdson_to_break", "TO_54", "labels.npy"))
    # data = np.concatenate(data, axis=0)
    plt.plot(data)
    plt.show()


def main():
    """Main function to generate all datasets."""
    sets = ['TO_54', 'TO_32', 'TO_88']

    # Uncomment to generate maximum data
    # generate_dataset_numpy("", feature_type="uds_max",
    #    label_type="to_break_max", max_data=True)

    # Generate datasets for each device using 'all' sampling
    for dataset in tqdm(sets, desc="Generating datasets"):
        generate_dataset_numpy(
            dataset,
            feature_type="rdson",
            label_type="to_break",
            sampling=True,
            sampling_type='all'
        )


if __name__ == "__main__":
    main()
