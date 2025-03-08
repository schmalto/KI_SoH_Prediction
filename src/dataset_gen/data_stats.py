"""
Statistical analysis module for power MOSFET parameter correlation.

This module provides functions to:
1. Load and combine maximum parameter data from various datasets
2. Calculate correlation between electrical parameters
3. Generate heatmap visualizations of parameter correlations
"""

# Standard library imports
import os
from pathlib import Path

# Third-party imports
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter
from termcolor import colored

# Local imports
from src.dataset_gen.dataset_generation import generate_max_data, get_all_cooling, generate_cooling_curve_straight

# Define base directory relative to this file
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASETS_DIR = os.path.join(BASE_DIR, "datasets")
PLOTS_DIR = os.path.join(BASE_DIR, "plots", "statistics")

# Create plots directory if it doesn't exist
os.makedirs(PLOTS_DIR, exist_ok=True)


def get_all_data():
    """
    Load and combine data from all available datasets.
    
    Returns:
        pandas.DataFrame: DataFrame containing all parameters with aligned indices
    """
    try:
        # Load maximum parameter data
        ids_max, deg = generate_max_data('ids_max')
        rds_max, _ = generate_max_data('rds_max')
        uds_max, _ = generate_max_data('uds_max')
        usd_max, _ = generate_max_data('usd_max')
        uth_down, _ = generate_max_data('uth_down')
        uth_up, _ = generate_max_data('uth_up')
        
        # Load temperature data from each dataset
        temperature = np.array([])
        sets = ['TO_54', 'TO_32', 'TO_88']
        
        for device in sets:
            try:
                folder_path = os.path.join(DATASETS_DIR, 'cooling_to_break', device)
                cycles_path = os.path.join(folder_path, 'cycles.npy')
                
                if not os.path.exists(cycles_path):
                    print(colored(f"Warning: Cycles file not found for {device}", "yellow"))
                    continue
                    
                temp = np.load(cycles_path)
                temperature = np.append(temperature, temp)
            except Exception as e:
                print(colored(f"Error loading temperature data for {device}: {e}", "red"))
        
        # Ensure all arrays have the same length
        min_length = min(
            uth_up.shape[0], 
            deg.shape[0],
            len(temperature)
        )
        
        # Align all arrays to the minimum length
        features = np.array(temperature)[:min_length]
        ids_max = ids_max[:min_length]
        rds_max = rds_max[:min_length]
        uds_max = uds_max[:min_length]
        usd_max = usd_max[:min_length]
        uth_down = uth_down[:min_length]
        uth_up = uth_up[:min_length]
        deg = deg[:min_length]
        
        # Create a DataFrame with all variables
        data = pd.DataFrame({
            'Degradation': deg,
            'ids_max': ids_max,
            'rds_max': rds_max,
            'uds_max': uds_max,
            'usd_max': usd_max,
            'uth_down': uth_down,
            'uth_up': uth_up,
            'temperature': features,
        })
        
        return data
        
    except Exception as e:
        print(colored(f"Error generating dataset: {e}", "red"))
        return pd.DataFrame()


def display_corr_matrix_heatmap(corr_matrix, save=False, filename='correlation_matrix.pdf'):
    """
    Display a heatmap of the correlation matrix.
    
    Args:
        corr_matrix: Pandas DataFrame containing correlation values
        save: Whether to save the figure to file
        filename: Filename for saved figure
    """
    plt.figure(figsize=(12, 10))
    
    # Define LaTeX formatted labels for parameters
    xticklabels = [
        'Degradation', 
        '$ \hat{\mathrm{I}}_{\mathrm{DS}} $', 
        '$ \hat{\mathrm{R}}_{\mathrm{DS}} $', 
        '$ \hat{\mathrm{U}}_{\mathrm{DS}} $', 
        '$ \hat{\mathrm{I}}_{\mathrm{SD}} $', 
        '$ \hat{\mathrm{U}}_{\mathrm{th, down}} $', 
        '$ \hat{\mathrm{U}}_{\mathrm{th, up}} $', 
        '$ \Delta\mathrm{T} $'
    ]
    
    # Configure seaborn for LaTeX rendering
    sns.set_theme(rc={'text.usetex': True})
    
    # Create heatmap
    ax = sns.heatmap(
        corr_matrix, 
        annot=True, 
        cmap='rocket_r', 
        xticklabels=xticklabels, 
        yticklabels=xticklabels,
        fmt='.2f',
        linewidths=0.5,
        annot_kws={"size": 12}
    )
    
    # Configure appearance
    ax.tick_params(axis='x', labelsize=17, rotation=45)
    ax.tick_params(axis='y', labelsize=17)
    ax.set_title("Parameter Correlation Matrix", fontsize=20, pad=20)
    
    # Add colorbar label
    cbar = ax.collections[0].colorbar
    cbar.ax.set_ylabel("Absolute Correlation", fontsize=16, labelpad=15)
    
    plt.tight_layout()
    
    # Save figure if requested
    if save:
        save_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(colored(f"Figure saved to {save_path}", "green"))
    
    plt.show()


def analyze_feature_importance(data):
    """
    Analyze the importance of each feature with respect to degradation.
    
    Args:
        data: DataFrame containing parameter data
        
    Returns:
        DataFrame sorted by correlation with degradation
    """
    # Get correlation with degradation
    degradation_corr = data.corr()['Degradation'].abs().sort_values(ascending=False)
    
    # Create feature importance DataFrame
    importance_df = pd.DataFrame({
        'Feature': degradation_corr.index,
        'Correlation with Degradation': degradation_corr.values
    })
    
    # Remove the degradation row itself
    importance_df = importance_df[importance_df['Feature'] != 'Degradation']
    
    return importance_df


def plot_feature_importance(importance_df, save=False, filename='feature_importance.pdf'):
    """
    Plot feature importance based on correlation with degradation.
    
    Args:
        importance_df: DataFrame with feature importance data
        save: Whether to save the figure
        filename: Filename for saved figure
    """
    plt.figure(figsize=(10, 6))
    
    # Create horizontal bar plot
    sns.barplot(
        data=importance_df,
        y='Feature',
        x='Correlation with Degradation',
        palette='viridis'
    )
    
    plt.title('Feature Importance for Degradation Prediction', fontsize=16)
    plt.xlabel('Absolute Correlation with Degradation', fontsize=14)
    plt.ylabel('Parameter', fontsize=14)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()
    
    if save:
        save_path = os.path.join(PLOTS_DIR, filename)
        plt.savefig(save_path, bbox_inches='tight', dpi=300)
        print(colored(f"Feature importance plot saved to {save_path}", "green"))
    
    plt.show()


def main():
    """Main function to run the statistical analysis."""
    print(colored("Loading and analyzing parameter data...", "green"))
    
    # Load all data
    data = get_all_data()
    
    if data.empty:
        print(colored("Error: Failed to load parameter data", "red"))
        return
        
    # Calculate correlation matrix
    corr_matrix = data.corr().abs()
    
    # Display correlation information
    print("\nCorrelation Matrix:")
    print(corr_matrix)
    
    # Display feature importance
    importance = analyze_feature_importance(data)
    print("\nFeature Importance:")
    print(importance)
    
    # Generate visualizations
    print(colored("\nGenerating correlation heatmap...", "green"))
    display_corr_matrix_heatmap(corr_matrix, save=True)
    
    print(colored("\nGenerating feature importance plot...", "green"))
    plot_feature_importance(importance, save=True)


if __name__ == "__main__":
    main()