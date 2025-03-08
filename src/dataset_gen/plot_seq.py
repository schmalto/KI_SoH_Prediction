"""
Utility module for creating technical drawing style plots of sequence data.
"""
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator
from termcolor import colored


def plot_data(x, y, sequence_points, title="Time Series with Technical Dimensions",
              xlabel="Time", ylabel="Value", save_path=None, figsize=(10, 6), dpi=300,
              line_color='blue', marker_color='black', grid=True):
    """
    Create a technical drawing style plot with dimension lines for sequence points.

    Args:
        x: X-axis data points
        y: Y-axis data points
        sequence_points: List of x positions to mark with dimension lines
        title: Plot title
        xlabel: X-axis label
        ylabel: Y-axis label
        save_path: Path to save the figure (if None, figure is displayed but not saved)
        figsize: Figure size as (width, height) in inches
        dpi: Resolution for saved figure
        line_color: Color of the main data line
        marker_color: Color of dimension markers
        grid: Whether to display grid lines

    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    try:
        # Input validation
        if len(x) != len(y):
            raise ValueError("x and y must have the same length")
        if len(sequence_points) < 2:
            raise ValueError("At least 2 sequence points are required for dimensions")

        # Create figure
        fig, ax = plt.subplots(figsize=figsize)

        # Plot main data
        ax.plot(x, y, label="Data", color=line_color, linewidth=1.5)

        # Calculate positioning for dimension lines
        y_min, y_max = np.min(y), np.max(y)
        y_range = y_max - y_min
        dimension_height = y_max + 0.15 * y_range  # Place dimension lines above data
        marker_height = 0.1 * y_range  # Height of vertical markers

        # Draw vertical markers and horizontal dimension lines
        for i in range(len(sequence_points)-1):
            start_point = sequence_points[i]
            end_point = sequence_points[i+1]

            # Draw dimension line
            ax.hlines(y=dimension_height, xmin=start_point, xmax=end_point,
                      colors=marker_color, linestyles='--', linewidth=0.8)

            # Draw vertical markers
            ax.vlines(x=start_point, ymin=dimension_height - marker_height,
                      ymax=dimension_height + marker_height/2, colors=marker_color, linewidth=1)
            ax.vlines(x=end_point, ymin=dimension_height - marker_height,
                      ymax=dimension_height + marker_height/2, colors=marker_color, linewidth=1)

            # Add dimension text
            segment_length = end_point - start_point
            mid_point = start_point + segment_length / 2
            ax.text(mid_point, dimension_height + marker_height,
                    f"{segment_length:.2f}", ha='center', va='bottom',
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none', boxstyle='round,pad=0.2'))

        # Customize the plot
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.legend(loc='best')

        # Set grid styling
        if grid:
            ax.grid(True, linestyle=':', alpha=0.4)
            ax.grid(which='minor', linestyle=':', alpha=0.2)
            ax.xaxis.set_minor_locator(AutoMinorLocator())
            ax.yaxis.set_minor_locator(AutoMinorLocator())

        # Add a slight margin to show all elements
        plt.margins(y=0.25)
        plt.tight_layout()

        # Save figure if path provided
        if save_path:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(os.path.abspath(save_path)), exist_ok=True)
            plt.savefig(save_path, dpi=dpi, bbox_inches='tight')
            print(colored(f"Figure saved to {save_path}", "green"))

        return fig

    except Exception as e:
        print(colored(f"Error creating plot: {e}", "red"))
        return None


def create_sequence_visualization(data, sequence_length, output_path=None):
    """
    Create a visualization of how data is split into sequences.

    Args:
        data: Time series data to visualize
        sequence_length: Length of each sequence
        output_path: Path to save the generated figure

    Returns:
        matplotlib.figure.Figure: The created figure object
    """
    x = np.arange(len(data))

    # Calculate sequence points
    sequence_points = [i * sequence_length for i in range(len(data) // sequence_length + 1)]

    # Create the visualization
    fig = plot_data(
        x, data,
        sequence_points,
        title="Time Series Segmentation Visualization",
        xlabel="Sample Index",
        ylabel="Value",
        save_path=output_path,
        line_color='royalblue'
    )

    return fig


def main():
    """Example usage of plotting functions."""
    # Generate sample data
    np.random.seed(42)
    x = np.linspace(0, 10, 200)
    y = np.sin(x) + 0.1 * np.random.randn(len(x))

    # Define some sequence points
    sequence_points = [0, 50, 100, 150, 200]

    # Create and show the plot
    plot_data(x, y, sequence_points,
              title="Sample Technical Drawing",
              xlabel="Time (s)",
              ylabel="Amplitude",
              save_path="technical_drawing.png")

    plt.show()


if __name__ == "__main__":
    main()
