"""
Channel conversion module for processing measurement data from different devices.
Handles loading, processing, and saving of time series data from various sources.
"""
import os
import glob
from pathlib import Path
import time

# Third-party imports
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
from termcolor import colored
import scipy.io as sio
from sklearn.preprocessing import minmax_scale
import cupy as cp

# Configure matplotlib
matplotlib.use('tkagg')  # Can be changed to 'svg' for figure saving

# Define base paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATA_INPUT_DIR = os.path.join(os.path.dirname(BASE_DIR), "sAPC")
DATA_OUTPUT_DIR = os.path.join(BASE_DIR, "data")
PLOTS_DIR = os.path.join(BASE_DIR, "plots")

# Ensure directories exist
for directory in [DATA_OUTPUT_DIR, PLOTS_DIR]:
    os.makedirs(directory, exist_ok=True)


# Logging functions
def output_error(msg):
    """Print error message in red."""
    print(colored(f"[Error] {msg}", 'red'))


def output_debug(msg):
    """Print debug message in yellow."""
    print(colored(f"[Debug] {msg}", 'yellow'))


def output_info(msg):
    """Print info message in green."""
    print(colored(f"[Info] {msg}", 'green'))


# File handling functions
def get_txt_files(folder, base_dir=None):
    """Get sorted list of text files in the specified folder."""
    base_dir = base_dir or DATA_INPUT_DIR
    folder_path = os.path.join(base_dir, folder, 'txt')
    files = glob.glob(os.path.join(folder_path, '*.txt'))
    return sorted(files)


def get_mat_files(folder, base_dir=None, max_data_flag=False, segmented=True):
    """Get sorted list of MAT files in the specified folder.

    Args:
        folder: Folder name to search in
        base_dir: Base directory (defaults to DATA_INPUT_DIR)
        max_data_flag: Whether to get max data files
        segmented: Whether the data is segmented

    Returns:
        For segmented data: tuple of (current, voltage) file paths
        Otherwise: list of file paths
    """
    base_dir = base_dir or DATA_INPUT_DIR
    folder_path = os.path.join(base_dir, folder)

    if not max_data_flag and segmented:
        save_dir = os.path.join(folder_path, 'mat')
        current = sorted(glob.glob(os.path.join(save_dir, 'I_*.mat')))
        uds = sorted(glob.glob(os.path.join(save_dir, 'Uds_*.mat')))
        return current, uds
    elif max_data_flag:
        return sorted(glob.glob(os.path.join(folder_path, '*.mat')))
    else:  # not segmented
        save_dir = os.path.join(folder_path, 'mat')
        return sorted(glob.glob(os.path.join(save_dir, '*.mat')))


# Data loading functions
def read_mat(file_path):
    """Read MATLAB .mat file and return its contents."""
    try:
        return sio.loadmat(file_path)
    except Exception as e:
        output_error(f"Failed to read MAT file {file_path}: {e}")
        return None


def load_data(file):
    """Load data from a MAT file."""
    return read_mat(file)


def load_txt(file):
    """Load data from a text file using CuPy."""
    try:
        return cp.loadtxt(file, delimiter=',')
    except Exception as e:
        output_error(f"Failed to load text file {file}: {e}")
        return None


def load_data_non_seg(file):
    """Load non-segmented data from MAT file."""
    mat = read_mat(file)
    return mat['dat'] if mat and 'dat' in mat else None


def load_segmented_data(file, start_time=-0.01, end_time=np.inf):
    """Load segmented data from MAT file with time constraints.

    Args:
        file: Path to MAT file
        start_time: Start time for filtering data
        end_time: End time for filtering data

    Returns:
        Tuple of (channel1_data, channel2_data) if channel 2 exists
        Otherwise just channel1_data
    """
    mat = read_mat(file)
    if not mat:
        return None

    r_Ch1 = mat.get('Channel_1')
    r_Ch2 = mat.get('Channel_2')

    if not r_Ch1:
        output_error(f"No Channel_1 data found in {file}")
        return None

    try:
        p_start_index, p_end_index = calculate_start_end_indices(
            r_Ch1, start_time, end_time)

        Ch1 = conversion(r_Ch1, p_start_index, p_end_index)
        if r_Ch2 is not None:
            Ch2 = conversion(r_Ch2, p_start_index, p_end_index)
            return Ch1, Ch2
        return Ch1
    except Exception as e:
        output_error(f"Error processing segmented data: {e}")
        return None


# Processing functions
def convert_max_data():
    """Convert and process maximum data files."""
    load_folder = os.path.join(DATA_INPUT_DIR, 'MaxData')
    save_folder = os.path.join(DATA_OUTPUT_DIR, 'MaxData')

    files = get_mat_files(load_folder, max_data_flag=True)
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    for i, file in enumerate(tqdm(files, desc="Processing max data files")):
        if i == 0:
            # Skip first file (as in original code)
            continue

        data = read_mat(file)
        if not data or 'export_data' not in data:
            output_error(f"Invalid data format in {file}")
            continue

        export_data = data['export_data']

        # Extract different data columns
        norm_deg = export_data[:, 0]
        rds_max = export_data[:, 1]
        usd_max = export_data[:, 2]
        usd_min = export_data[:, 3]
        uth_up = export_data[:, 4]
        uth_down = export_data[:, 5]
        uds_max = export_data[:, 6]
        ids_max = export_data[:, 7]

        # Save each curve type
        data_to_save = {
            'norm_deg': norm_deg,
            'rds_max': rds_max,
            'usd_max': usd_max,
            'usd_min': usd_min,
            'uth_up': uth_up,
            'uth_down': uth_down,
            'uds_max': uds_max,
            'ids_max': ids_max
        }

        for curve_type, curve_data in data_to_save.items():
            save_curve(file, save_folder, curve_data, curve_type=curve_type, index=i)


def plot_current(data):
    """Plot current data sample."""
    start = 250 * 10
    end = 1500 * 10

    x_axis = np.arange(len(data[0])) / 10

    plt.figure(figsize=(10, 6))
    plt.plot(x_axis[start:end], data[100][start:end])
    plt.grid()
    plt.xlabel('$t$ [ms]')
    plt.ylabel('Current')
    plt.savefig(os.path.join(PLOTS_DIR, 'i_sapc.pdf'))
    plt.close()


def chunk_array(arr, chunk_size):
    """Split array into chunks of specified size."""
    n_chunks = arr.shape[0] // chunk_size
    return np.array([arr[i * chunk_size:(i + 1) * chunk_size] for i in range(n_chunks)])


def convert_non_segmented_data(folder):
    """Convert non-segmented data for the specified folder."""
    sampling = 5  # Sample reduction factor
    output_info(f"Converting non-segmented data for {folder}...")

    i_files = get_mat_files(folder, segmented=False)
    if not i_files:
        output_error(f"No files found for {folder}")
        return

    # Load first file to determine dimensions
    first_file_data = load_data_non_seg(i_files[0])
    if first_file_data is None:
        return

    len_mess = first_file_data.shape[0]
    no_files = len(i_files)
    ten_percent_of_mess = max(1, no_files // 6)  # Ensure at least 1

    # Pre-allocate arrays
    sample_size = (len_mess // sampling) + 1
    uds = np.zeros((no_files, sample_size), dtype=np.float32)
    ids_2 = np.zeros((no_files, sample_size), dtype=np.float32)

    save_folder = os.path.join(DATA_OUTPUT_DIR, folder)
    Path(save_folder).mkdir(parents=True, exist_ok=True)

    i = 0
    k = 0
    for file in tqdm(i_files, desc=f"Processing files for {folder}"):
        data = load_data_non_seg(file)
        if data is None:
            continue

        # Extract voltage and current data
        uds_t = data[:, 2]
        uds[i, :] = uds_t[::sampling]

        ids_t = data[:, 4]
        ids_2[i, :] = ids_t[::sampling]

        i += 1
        if i % ten_percent_of_mess == 0:
            save_formatted_data(file, uds, ids_2, k, save_folder, segmented=False)
            i = 0
            k += 1


def calculate_rdson_non_seg(voltage, current, c_threshold=22):
    """Calculate R_DS(on) for non-segmented data.

    Args:
        voltage: Voltage data array
        current: Current data array
        c_threshold: Current threshold for filtering

    Returns:
        R_DS(on) array
    """
    try:
        voltage_cp = cp.array(voltage)
        current_cp = cp.array(current)
        rdson_list = []
        max_len = 0

        for i in range(len(voltage_cp)):
            v = voltage_cp[i]
            c = current_cp[i]

            # Apply current threshold
            mask = c > c_threshold
            v_masked = v[mask]
            c_masked = c[mask]

            # Apply additional non-zero voltage filter
            mask_v = v_masked != 0
            v_masked = v_masked[mask_v]
            c_masked = c_masked[mask_v]

            # Calculate resistance where current is non-zero
            filtered_r = cp.where(c_masked != 0, v_masked / c_masked, cp.zeros_like(v_masked))
            rdson_list.append(filtered_r)

            # Track maximum length
            if filtered_r.shape[0] > max_len:
                max_len = filtered_r.shape[0]

        # Create uniform-sized output array
        rdson_array = np.zeros((len(rdson_list), max_len), dtype=voltage[0].dtype)

        # Copy data back to CPU
        for i, arr_gpu in enumerate(rdson_list):
            arr_cpu = cp.asnumpy(arr_gpu)
            rdson_array[i, :arr_cpu.shape[0]] = arr_cpu

        return rdson_array

    except Exception as e:
        output_error(f"Error calculating R_DS(on): {e}")
        return np.array([])


def convert_segmented_data(folder, segmented=True):
    """Convert segmented data files to processed data.

    Args:
        folder: Folder containing the data files
        segmented: Whether the data is segmented
    """
    output_info(f"Converting segmented data for {folder}...")

    try:
        i_files, u_files = get_mat_files(folder, segmented=True)
        if not u_files:
            output_error(f"No files found for {folder}")
            return

        save_folder = os.path.join(DATA_OUTPUT_DIR, folder)
        Path(save_folder).mkdir(parents=True, exist_ok=True)

        for i, file in enumerate(tqdm(u_files, desc=f"Processing files for {folder}")):
            v_data_conv = load_segmented_data(file, start_time=-np.inf, end_time=np.inf)
            if v_data_conv is not None:
                save_formatted_data(file, v_data_conv[1], v_data_conv[0], i+1, save_folder)

    except Exception as e:
        output_error(f"Error converting segmented data for {folder}: {e}")


def save_formatted_data(file, uds, ids, index, save_folder, segmented=True):
    """Process and save various curve types extracted from the data.

    Args:
        file: Original data file path
        uds: Voltage data
        ids: Current data
        index: File index
        save_folder: Output folder path
        segmented: Whether the data is segmented
    """
    try:
        # Calculate R_DS(on)
        rdson = calculate_rdson_non_seg(uds, ids)

        # Extract various curve types
        cooling_curve, v_th_down, v_th_up, wo_cooling, complete, vdson = get_curves(uds)

        # Save all curves
        curves_to_save = {
            'rdson': rdson,
            'vth_down': v_th_down,
            'vth_up': v_th_up,
            'cooling': cooling_curve,
            'complete': complete,
            'wocooling': wo_cooling,
            'vdson': vdson
        }

        for curve_type, curve_data in curves_to_save.items():
            save_curve(file, save_folder, curve_data, curve_type=curve_type, index=index)

    except Exception as e:
        output_error(f"Error saving formatted data: {e}")


def const_array_size(arr, size, axis=1):
    """Resize array to specified size along given axis."""
    if arr.shape[axis] < size:
        pad_width = [(0, 0), (0, 0)]
        pad_width[axis] = (0, size - arr.shape[axis])
        return np.pad(arr, pad_width, 'constant')
    else:
        slices = [slice(None), slice(None)]
        slices[axis] = slice(0, size)
        return arr[tuple(slices)]


def new_min_length(curve, min_length):
    """Calculate new minimum length between curve length and current min_length."""
    return min(min_length, len(curve))


def format_curve(curve, min_length):
    """Format list of curves to uniform length arrays.

    Args:
        curve: List of curve data arrays
        min_length: Minimum length to use

    Returns:
        CuPy array with uniform sized curves
    """
    n = len(curve)
    if n == 0:
        return cp.array([])

    # Pre-allocate output array
    out = cp.zeros((n, min_length), dtype=cp.float32)

    # Copy data, truncating or padding as needed
    for i, el in enumerate(curve):
        el_cp = cp.asarray(el, dtype=cp.float32)
        length = min(min_length, el_cp.shape[0])
        out[i, :length] = el_cp[:length]

    return out


def get_rdson(cycles_i, cycles_u):
    """Calculate R_DS(on) from current and voltage cycles.

    Args:
        cycles_i: Current cycles data
        cycles_u: Voltage cycles data

    Returns:
        R_DS(on) curves
    """
    rdson_curves = []
    rdson_curves_length_min = np.inf

    for j in range(len(cycles_u)):
        try:
            # Find indices where current exceeds threshold
            idx = np.where(cycles_i[j] > 22)

            i = cycles_i[j][idx]
            u = cycles_u[j][idx]

            # Update minimum length for consistent arrays
            if j != len(cycles_u)-1:
                rdson_curves_length_min = new_min_length(i, rdson_curves_length_min)
                rdson_curves_length_min = new_min_length(u, rdson_curves_length_min)
                rdson = np.divide(u, i, out=np.zeros_like(u), where=i != 0)
            else:
                rdson = np.divide(u, i, out=np.zeros_like(u), where=i != 0)
                rdson = const_array_size(rdson, rdson_curves_length_min)

            rdson_curves.append(rdson)

        except IndexError:
            output_debug(f"IndexError in get_rdson for cycle {j}")

    # Format curves to uniform size
    return format_curve(rdson_curves, rdson_curves_length_min)


def get_curves(cycles):
    """Extract various curve types from the cycles data.

    Returns:
        Tuple of (cooling_curves, v_th_down_curves, v_th_up_curves, 
                 complete_curves, wo_cooling_curves, vdson_curves)
    """
    # Initialize lists for different curve types
    cooling_curves = []
    v_th_up_curves = []
    v_th_down_curves = []
    complete_curves = []
    wo_cooling_curves = []
    vdson_curves = []

    # Initialize minimum lengths
    cooling_length_min = np.inf
    v_th_up_length_min = np.inf
    v_th_down_length_min = np.inf
    wo_cooling_curves_length_min = np.inf
    complete_curves_length_min = np.inf
    vdson_curves_length_min = np.inf

    # Process each cycle
    for cycle in cycles:
        try:
            # Split cycle into segments
            idx = split_cycle(cycle)

            # Extract segments
            wo_cooling = cycle[:idx[5]]
            v_th_up = cycle[idx[0]:idx[1]]
            v_th_down = cycle[idx[2]:idx[4]]
            vdson = cycle[idx[4]:idx[5]]
            complete = cycle

            # Update minimum lengths
            v_th_up_length_min = new_min_length(v_th_up, v_th_up_length_min)
            v_th_down_length_min = new_min_length(v_th_down, v_th_down_length_min)
            wo_cooling_curves_length_min = new_min_length(wo_cooling, wo_cooling_curves_length_min)
            complete_curves_length_min = new_min_length(complete, complete_curves_length_min)
            vdson_curves_length_min = new_min_length(vdson, vdson_curves_length_min)

            # Append to respective lists
            v_th_up_curves.append(v_th_up)
            v_th_down_curves.append(v_th_down)
            wo_cooling_curves.append(wo_cooling)
            complete_curves.append(complete)
            vdson_curves.append(vdson)

            # Process cooling curve
            cooling = cycle[idx[5]:]
            cooling = cooling[10:]  # Discard first samples
            cooling_length_min = new_min_length(cooling, cooling_length_min)
            cooling_curves.append(cooling)

        except IndexError:
            continue

    # Format all curves to uniform lengths
    cooling_curves = format_curve(cooling_curves, cooling_length_min)
    v_th_up_curves = format_curve(v_th_up_curves, v_th_up_length_min)
    v_th_down_curves = format_curve(v_th_down_curves, v_th_down_length_min)
    wo_cooling_curves = format_curve(wo_cooling_curves, wo_cooling_curves_length_min)
    complete_curves = format_curve(complete_curves, complete_curves_length_min)
    vdson_curves = format_curve(vdson_curves, vdson_curves_length_min)

    return (cooling_curves, v_th_down_curves, v_th_up_curves,
            wo_cooling_curves, complete_curves, vdson_curves)


def save_curve(file, save_folder, data, curve_type='rdson', index=1):
    """Save curve data to a NumPy file.

    Args:
        file: Original data file path
        save_folder: Output folder path
        data: Curve data to save
        curve_type: Type of curve
        index: File index
    """
    save_name = os.path.join(save_folder, f"{curve_type}_{index}.npy")
    try:
        np.save(save_name, data)
    except Exception as e:
        output_error(f"Failed to save curve: {e}")


def decide_offset(p_data):
    """Determine if data has an offset.

    Returns:
        True if data appears to be in absolute values, False otherwise
    """
    try:
        s_data = p_data[0][0].flatten()
        r_values = np.random.choice(s_data, min(20, len(s_data)))
        after_decimal = np.absolute(np.modf(r_values)[0])
        return np.sum(after_decimal) == 0.0
    except Exception:
        return False


def scipy_flatten(p_data):
    """Flatten nested arrays from scipy.io.loadmat."""
    try:
        return p_data[0][0][0][0]
    except (IndexError, TypeError):
        output_error("Failed to flatten data structure")
        return None


def conversion(channel_data, start_index, end_index):
    """Convert channel data using scale and offset.

    Args:
        channel_data: Raw channel data
        start_index: Start index for data extraction
        end_index: End index for data extraction

    Returns:
        Converted data array
    """
    try:
        conv_data = []
        YInc = scipy_flatten(channel_data['YInc'])
        YOrg = scipy_flatten(channel_data['YOrg'])
        Data = channel_data[0][0][2][0]
        add_offset = decide_offset(Data)

        for el in Data:
            # Handle the specific nested data structure
            flattened = el.flatten()[0][0].flatten()
            cell = np.array(flattened)[start_index:end_index + 2]

            # Apply conversion if needed
            if add_offset:
                conv_temp = (YInc * cell) + YOrg
            else:
                conv_temp = cell

            conv_data.append(conv_temp)

        return np.array(conv_data)

    except Exception as e:
        output_error(f"Error in data conversion: {e}")
        return np.array([])


def calculate_start_end_indices(Ch1, start_time, end_time):
    """Calculate start and end indices based on time constraints.

    Args:
        Ch1: Channel 1 data
        start_time: Start time
        end_time: End time

    Returns:
        Tuple of (start_index, end_index)
    """
    try:
        NumPoints = scipy_flatten(Ch1['NumPoints'])
        XInc = scipy_flatten(Ch1['XInc'])
        XOrg = scipy_flatten(Ch1['XOrg'])

        # Generate time array
        time = XInc * np.arange(1, NumPoints + 1) + XOrg

        # Find indices within time range
        indices = np.where((time >= start_time) & (time <= end_time))[0]
        indices = indices[np.where((indices >= 0) & (indices < NumPoints))[0]]

        if len(indices) == 0:
            output_error("No indices found within specified time range")
            return 0, NumPoints - 1

        return int(indices[0]), int(indices[-1])

    except Exception as e:
        output_error(f"Error calculating indices: {e}")
        return 0, 0


def decode_channel(mat_file):
    """Decode channel data from MAT file."""
    data = load_segmented_data(mat_file, start_time=-np.inf, end_time=np.inf)
    return np.concatenate(data[1], axis=0) if data is not None else np.array([])


def split_cycle(data):
    """Split cycle data based on significant changes.

    Args:
        data: Cycle data array

    Returns:
        Array of split indices
    """
    try:
        # Normalize and filter data
        n_data = minmax_scale(data, feature_range=(0, 10))
        filtered = n_data * n_data * n_data

        # Calculate differences and apply thresholding
        diff = np.diff(filtered)
        diff = diff * diff * diff * diff * diff
        threshold = 200000

        diff = np.append(diff, np.inf)
        indexes_all = np.where(np.abs(diff) > threshold)[0]

        # Find sufficiently spaced indices
        n_arr = np.diff(indexes_all)
        id_x = np.where(n_arr > 500)[0]

        return indexes_all[id_x]

    except Exception as e:
        output_error(f"Error splitting cycle: {e}")
        return np.array([0, 10, 20, 30, 40, 50])  # Default fallback values


def debug_functions():
    """Collection of debug functions for development."""
    # Uncomment to use specific debug functions
    # three_channel_debug()
    # show_rdson()
    pass


def three_channel_debug():
    """Debug function for visualizing channel data."""
    data = np.load(os.path.join(DATA_OUTPUT_DIR, 'TO_4/Uds_cooling.npy'))
    time_axis = np.arange(len(data[0]))

    plt.figure(figsize=(10, 6))
    for el in data[::10]:
        plt.plot(time_axis, el)
    plt.grid()
    plt.title("Channel Debug Visualization")
    plt.xlabel("Time")
    plt.ylabel("Value")
    plt.show()


def show_rdson():
    """Debug function for visualizing R_DS(on) data."""
    data = np.load(os.path.join(DATA_OUTPUT_DIR, 'TO_54/Uds_rdson_3.npy'))
    data = np.concatenate(data, axis=0)

    plt.figure(figsize=(10, 6))
    plt.plot(data)
    plt.grid()
    plt.title("R_DS(on) Visualization")
    plt.xlabel("Sample")
    plt.ylabel("R_DS(on)")
    plt.show()


def main():
    """Main function to run the channel conversion process."""
    start_time = time.time()

    # Process specified datasets
    datasets = ['TO_54', 'TO_32']  # Add or remove datasets as needed

    for dataset in datasets:
        output_info(f"Processing dataset: {dataset}")
        convert_segmented_data(dataset)

    # Uncomment to process additional features
    # convert_max_data()

    elapsed_time = time.time() - start_time
    output_info(f"Processing completed in {elapsed_time:.2f} seconds")


if __name__ == '__main__':
    main()
