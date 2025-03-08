import os
import numpy as np
import tensorflow as tf
import glob
from pathlib import Path
from matplotlib import pyplot as plt
from termcolor import colored
from tqdm import tqdm

# Suppress TensorFlow logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# Define base directory relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
TRAINING_SETS_DIR = os.path.join(BASE_DIR, "training_sets")

# Constants
DEFAULT_AUGMENTATIONS = 3
FIXED_LENGTH_FEATURE = 50  # Default fixed length for time series features

# Set seed for reproducibility
np.random.seed(42)


class TimeSeriesAugmenter:
    """Class for time series augmentation methods"""

    @staticmethod
    def jittering(time_series, sigma=0.01):
        """Add random Gaussian noise to the time series.

        Args:
            time_series: Input time series data
            sigma: Standard deviation of the noise

        Returns:
            Augmented time series with added noise
        """
        noise = np.random.normal(loc=0.0, scale=sigma, size=time_series.shape)
        return time_series + noise

    @staticmethod
    def scaling(time_series, sigma=0.005):
        """Scale the time series by a random factor.

        Args:
            time_series: Input time series data
            sigma: Scale factor range (1±sigma)

        Returns:
            Scaled time series
        """
        scaling_factor = np.random.uniform(1 - sigma, 1 + sigma)
        return time_series * scaling_factor

    @staticmethod
    def time_warping(time_series, max_warp=0.2):
        """Apply time warping by shifting time series indices.

        Args:
            time_series: Input time series data
            max_warp: Maximum warp amount

        Returns:
            Time-warped time series
        """
        indices = np.arange(time_series.shape[0])
        warp_amount = np.random.uniform(-max_warp, max_warp)
        warp = indices + warp_amount
        idx = np.clip(warp, 0, time_series.shape[0] - 1).astype(int)
        return np.take(time_series, idx)

    @staticmethod
    def window_slicing(time_series, slice_ratio=0.9):
        """Slice a window from the time series and symmetrically pad to original length.

        Args:
            time_series: Input time series data
            slice_ratio: Ratio of the slice length to original length

        Returns:
            Sliced and padded time series
        """
        original_len = time_series.shape[0]
        slice_len = int(original_len * slice_ratio)
        start = np.random.randint(0, original_len - slice_len)
        sliced_series = time_series[start: start + slice_len]
        return np.pad(sliced_series, (0, original_len - slice_len), mode="symmetric")

    @classmethod
    def augment_time_series(cls, time_series, cooling_curve=False):
        """Apply a combination of augmentation techniques.

        Args:
            time_series: Input time series data
            cooling_curve: Whether the data is a cooling curve

        Returns:
            Augmented time series
        """
        if time_series.ndim == 0:
            return cls.jittering(time_series)

        if cooling_curve:
            # For cooling curves, apply sequential transformations to the same series
            ts = cls.jittering(time_series)
            ts = cls.scaling(ts)
            ts = cls.time_warping(ts)
            return ts
        else:
            # For regular time series, apply transformations to each sequence
            augmented_series = []
            for seq in time_series:
                ts = cls.jittering(seq)
                ts = cls.scaling(ts)
                ts = cls.time_warping(ts)
                ts = cls.window_slicing(ts)
                augmented_series.append(ts)
            return augmented_series


def normalize(time_series):
    """Normalize time series by the mean of its top values.

    Args:
        time_series: Input time series data

    Returns:
        Normalized time series
    """
    max_ind = np.argpartition(time_series, -10)[-10:]
    max_avg = np.mean(time_series[max_ind])
    return time_series / max_avg if max_avg != 0 else time_series


def debug_augmentation(original_ts, augmented_ts, sample_rate=100):
    """Visualize original and augmented time series.

    Args:
        original_ts: Original time series
        augmented_ts: Augmented time series
        sample_rate: Sample every nth point for visualization
    """
    plt.figure(figsize=(12, 6))

    aug_con = np.concatenate(augmented_ts, axis=0)
    plt.plot(aug_con[::sample_rate], label="Augmented")
    plt.plot(original_ts[::sample_rate], label="Original")

    plt.title("Original vs Augmented Time Series")
    plt.legend()
    plt.grid(True)
    plt.show()


def create_cooling_dataset(time_series, labels, iterations=0):
    """Generate dataset for cooling curves with augmentation.

    Args:
        time_series: Input time series data
        labels: Corresponding labels
        iterations: Number of augmentation iterations

    Returns:
        TensorFlow dataset
    """
    def data_generator():
        # Yield original time series samples
        for i in range(len(time_series)):
            ts = time_series[i]
            lbl = labels[i] if len(labels) > 1 else labels
            yield [ts], [lbl]

        # Generate and yield augmented samples
        for _ in range(iterations):
            for i in range(len(time_series)):
                ts = time_series[i]
                lbl = labels[i] if len(labels) > 1 else labels
                aug_ts = TimeSeriesAugmenter.augment_time_series(ts, cooling_curve=True)
                yield [aug_ts], [lbl]

    # Define the output types and shapes
    output_signature = (
        tf.TensorSpec(shape=(None,), dtype=tf.float32),
        tf.TensorSpec(shape=(1,), dtype=tf.int32),
    )

    return tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )


def create_dataset(time_series, labels, iterations=DEFAULT_AUGMENTATIONS, cooling_curve=False):
    """Generate dataset with augmentation.

    Args:
        time_series: Input time series data
        labels: Corresponding labels
        iterations: Number of augmentation iterations
        cooling_curve: Whether the data represents cooling curves

    Returns:
        TensorFlow dataset
    """
    def data_generator():
        # Yield original time series samples
        for i in range(len(time_series) - 1):
            try:
                ts = time_series[i]
                lbl = labels[i] if len(labels) > 1 else labels
                yield ts, [lbl]
            except IndexError as e:
                print(colored(f"Error processing sample {i}: {e}", "red"))
                continue

        # Generate and yield augmented samples
        for _ in range(iterations):
            for i in range(len(time_series)):
                try:
                    ts = time_series[i]
                    lbl = labels[i] if len(labels) > 1 else labels
                    aug_ts = TimeSeriesAugmenter.augment_time_series(ts, cooling_curve=cooling_curve)
                    yield aug_ts, [lbl]
                except Exception as e:
                    print(colored(f"Error augmenting sample {i}: {e}", "red"))
                    continue

    # Define the output types and shapes based on data type
    if cooling_curve:
        output_signature = (
            tf.TensorSpec(shape=(1,), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.int32),
        )
    else:
        output_signature = (
            tf.TensorSpec(shape=(None,), dtype=tf.float32),
            tf.TensorSpec(shape=(1,), dtype=tf.int32),
        )

    return tf.data.Dataset.from_generator(
        data_generator,
        output_signature=output_signature
    )


def serialize_example(time_series, label):
    """Serialize a single example for TFRecord.

    Args:
        time_series: Time series data
        label: Corresponding label

    Returns:
        Serialized example
    """
    # Convert to numpy and flatten
    time_series = time_series.numpy().flatten()

    # Create feature dictionary
    feature = {
        "time_series": tf.train.Feature(float_list=tf.train.FloatList(value=time_series)),
        "label": tf.train.Feature(float_list=tf.train.FloatList(value=label))
    }

    # Create Example protocol buffer
    example_proto = tf.train.Example(features=tf.train.Features(feature=feature))
    return example_proto.SerializeToString()


def generate_dataset(folder, augmentations=DEFAULT_AUGMENTATIONS, cooling_curve=False):
    """Generate dataset from files in folder.

    Args:
        folder: Path to folder containing cycles.npy and labels.npy
        augmentations: Number of augmentation iterations
        cooling_curve: Whether the data represents cooling curves

    Returns:
        TensorFlow dataset
    """
    try:
        # Load time series and labels
        time_series_path = os.path.join(folder, "cycles.npy")
        labels_path = os.path.join(folder, "labels.npy")

        if not os.path.exists(time_series_path) or not os.path.exists(labels_path):
            print(colored(f"Missing data files in {folder}", "red"))
            return None

        time_series = np.load(time_series_path)
        labels = np.load(labels_path)

        # Create appropriate dataset
        if cooling_curve:
            return create_cooling_dataset(time_series, labels, iterations=augmentations)
        else:
            return create_dataset(time_series, labels, iterations=augmentations, cooling_curve=cooling_curve)

    except Exception as e:
        print(colored(f"Error generating dataset from {folder}: {e}", "red"))
        return None


def create_augmented_dataset(modus="rdson", augs=DEFAULT_AUGMENTATIONS):
    """Create augmented dataset and save as TFRecord.

    Args:
        modus: Dataset mode/type
        augs: Number of augmentation iterations
    """
    # Determine if dealing with cooling curves
    cooling_curve = any(keyword in modus.split("_") for keyword in ["max", "cooling"])

    # Find all dataset folders for this mode
    folders = glob.glob(os.path.join(DATASETS_DIR, modus, "*"))
    if not folders:
        print(colored(f"No folders found for mode: {modus}", "yellow"))
        return

    # Prepare output path
    output_path = os.path.join(TRAINING_SETS_DIR, modus, "dataset.tfrecord")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Process all folders and write to TFRecord
    with tf.io.TFRecordWriter(output_path) as writer:
        for folder in tqdm(folders, desc=f"Processing {modus}"):
            ds = generate_dataset(folder, augmentations=augs, cooling_curve=cooling_curve)
            if ds is None:
                continue

            for time_series, label in ds:
                try:
                    example = serialize_example(time_series, label)
                    writer.write(example)
                except Exception as e:
                    print(colored(f"Error serializing example: {e}", "red"))
                    continue


def pad_time_series(time_series, max_len=1792):
    """Pad time series to a fixed length.

    Args:
        time_series: List of time series
        max_len: Maximum length to pad to

    Returns:
        List of padded time series
    """
    return [np.pad(ts, (0, max_len - len(ts)), mode="constant") for ts in time_series]


def generate_evaluation_dataset(device, modus):
    """Generate evaluation dataset for a specific device.

    Args:
        device: Device identifier (e.g., "TO_88")
        modus: Dataset mode/type
    """
    # Determine if dealing with cooling curves
    cooling_curve = "cooling" in modus.split("_")

    # Prepare folder and output paths
    folder = os.path.join(DATASETS_DIR, modus, device)
    if not os.path.exists(folder):
        print(colored(f"Folder not found: {folder}", "red"))
        return

    output_path = os.path.join(TRAINING_SETS_DIR, "evaluation", modus, "dataset.tfrecord")
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    # Process dataset with no augmentations
    with tf.io.TFRecordWriter(output_path) as writer:
        ds = generate_dataset(folder, augmentations=0, cooling_curve=cooling_curve)
        if ds is None:
            return

        print(f"Generating evaluation dataset for {device} ({modus})...")
        for time_series, label in ds:
            example = serialize_example(time_series, label)
            writer.write(example)


def main():
    """Main function to generate all required datasets."""
    # Basic datasets without augmentation
    create_augmented_dataset(modus="rdson_sampled_all", augs=0)
    create_augmented_dataset(modus="vth_sampled_all", augs=0)

    # Cooling dataset with augmentation
    create_augmented_dataset(modus="cooling_to_break", augs=2)

    # Uncomment to generate evaluation datasets
    # generate_evaluation_dataset("TO_88", modus="rdson_sampled_all")
    # generate_evaluation_dataset("TO_88", modus="cooling_to_break")
    # generate_evaluation_dataset("TO_88", modus="vth_sampled_all")
    # generate_evaluation_dataset("TO_32", modus="rdson_sampled_all")

    # Uncomment to generate other max datasets
    # datasets = [
    #     "ids_to_break_max", "rds_to_break_max", "uds_max_to_break_max",
    #     "usd_min_to_break_max", "usd_max_to_break_max", "uth_down_to_break_max",
    #     "uth_up_to_break_max"
    # ]
    # for data in tqdm(datasets, desc="Processing max datasets"):
    #     create_augmented_dataset(data, augs=0)


if __name__ == "__main__":
    main()
