"""
Convert MATLAB data files to TensorFlow datasets.

This script handles the end-to-end process of:
1. Converting segmented MAT files to processed data
2. Generating numpy datasets with features and labels
3. Creating augmented TensorFlow datasets
"""
import os
import argparse
from pathlib import Path
from termcolor import colored
from tqdm import tqdm

from src.dataset_gen.channel_conv import convert_segmented_data
from src.dataset_gen.dataset_generation import generate_dataset_numpy
from src.dataset_gen.augmentation import create_augmented_dataset

# Define available options
AVAILABLE_FEATURES = [
    'rdson', 'rul', 'cooling', 'cooling_curve', 'seq', 
    'vth_up', 'vth_down', 'complete', 'wocooling'
]
AVAILABLE_LABELS = ['to_break', 'cycle']
DEFAULT_DATASETS = ['TO_54', 'TO_32', 'TO_88']


def process_dataset(dataset, feature_type, label_type, augmentations=2):
    """
    Process a single dataset through the full pipeline.
    
    Args:
        dataset: Dataset identifier (e.g., 'TO_54')
        feature_type: Type of feature to extract
        label_type: Type of label to create
        augmentations: Number of augmentations to create
    
    Returns:
        bool: Success status
    """
    try:
        print(colored(f"Processing dataset: {dataset}", "green"))
        
        # Step 1: Convert segmented MAT data to processed data
        print(f"Converting segmented data for {dataset}...")
        convert_segmented_data(dataset)
        
        # Step 2: Generate numpy dataset with features and labels
        print(f"Generating {feature_type} features with {label_type} labels...")
        generate_dataset_numpy(dataset, feature_type=feature_type, label_type=label_type)
        
        return True
    except Exception as e:
        print(colored(f"Error processing dataset {dataset}: {e}", "red"))
        return False


def create_datasets(datasets=None, feature_type='cooling', label_type='to_break', augmentations=2):
    """
    Create datasets for specified parameters.
    
    Args:
        datasets: List of dataset identifiers
        feature_type: Type of feature to extract
        label_type: Type of label to create
        augmentations: Number of augmentations to create
    """
    # Validate inputs
    if feature_type not in AVAILABLE_FEATURES:
        print(colored(f"Invalid feature type: {feature_type}", "red"))
        print(f"Available features: {', '.join(AVAILABLE_FEATURES)}")
        return
        
    if label_type not in AVAILABLE_LABELS:
        print(colored(f"Invalid label type: {label_type}", "red"))
        print(f"Available labels: {', '.join(AVAILABLE_LABELS)}")
        return
    
    # Use default datasets if none specified
    if datasets is None or len(datasets) == 0:
        datasets = DEFAULT_DATASETS
    
    # Process each dataset
    successful_datasets = []
    failed_datasets = []
    
    for dataset in tqdm(datasets, desc="Processing datasets"):
        success = process_dataset(dataset, feature_type, label_type)
        if success:
            successful_datasets.append(dataset)
        else:
            failed_datasets.append(dataset)
    
    # Create combined augmented dataset if any were successful
    if successful_datasets:
        # Construct the dataset mode from feature and label types
        mode = f"{feature_type}_{label_type}"
        print(colored(f"Creating augmented dataset with mode: {mode}", "green"))
        create_augmented_dataset(modus=mode, augs=augmentations)
        print(colored("Dataset creation complete!", "green"))
    
    # Report results
    if successful_datasets:
        print(colored(f"Successfully processed datasets: {', '.join(successful_datasets)}", "green"))
    if failed_datasets:
        print(colored(f"Failed to process datasets: {', '.join(failed_datasets)}", "red"))


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Convert MAT files to TensorFlow datasets")
    
    parser.add_argument("--datasets", "-d", nargs="+", default=DEFAULT_DATASETS,
                      help="List of datasets to process")
    parser.add_argument("--feature", "-f", default="cooling", choices=AVAILABLE_FEATURES,
                      help="Feature type to extract")
    parser.add_argument("--label", "-l", default="to_break", choices=AVAILABLE_LABELS,
                      help="Label type to create")
    parser.add_argument("--augmentations", "-a", type=int, default=2,
                      help="Number of augmentations to create")
                      
    return parser.parse_args()


def main():
    """Main entry point."""
    # For command-line usage, uncomment:
    # args = parse_arguments()
    # create_datasets(
    #     datasets=args.datasets,
    #     feature_type=args.feature,
    #     label_type=args.label,
    #     augmentations=args.augmentations
    # )
    
    # Default behavior to match original script
    create_datasets(
        datasets=['TO_54'],
        feature_type='cooling',
        label_type='to_break',
        augmentations=2
    )


if __name__ == '__main__':
    main()
