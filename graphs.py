from matplotlib.ticker import MaxNLocator
import matplotlib.pyplot as plt
import numpy as np

def graph_plot(enabled, folder_path, graph_title, x_data, y_data, x_label, y_label, y_max):
    """
    Create and save a line graph display the passed in data.
    The chart is saved as a PNG image in the specified folder.

    Args:
        enabled (bool):
            Whether graph generation is enabled.

        folder_path (str):
            Directory where the graph image will be saved.

        graph_title (str):
            Title of the graph and filename of the output image.

        x_data (float):
            .

        y_data (float):
            .

        x_label (str):
            Label for the x-axis.

        y_label (str):
            Label for the y-axis.

        y_max (float):
            Maximum allowed y-axis limit.

    Returns:
        None

    Notes:
    """

    # The enabled parameter tracks whether or not the given graph is enabled in settings
    # If it is not enabled, don't graph it
    if not enabled:
        return

    # If there is only one data point make the graph a bar graph instead of a line graph
    if len(x_data) == 1:
        graph_plot_point(enabled, folder_path, graph_title, 0, y_data[0], "", y_label, y_max)
        return

    # Graph setup
    _, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(graph_title)

    # Plot data
    ax.plot(list(range(1, len(y_data) + 1)), y_data)

    # Set limits for y values
    ax.set_ylim(max(0, min(y_data) - 1), min(y_max + 0.025, max(y_data) + 1))

    # Ensure tick marks on x axis are only ever whole numbers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    # Save graph locally
    plt.savefig(f"{folder_path}/{graph_title}.png")
    plt.close()

"""
Purpose: Creates and saves a bar chart containing a single bar
Input: Enabled - boolean value representing whether or not the graph should be made
    folder_path - string containing the file path of the folder to save the graph to
    graph_title - string containing a title for the graph
    x_data / y_data - numeric data to graph on given axis
    x_label / y_label - string containing name for given axis
    y_max - maximum value for y bars
Output: None
Notes: A bar graph is used instead of a line graph to make it clearer what the value is
"""
def graph_plot_point(enabled, folder_path, graph_title, x_value, y_value, x_label, y_label, y_max):
    """
    Create and save a bar chart containing a single data point.
    The chart is saved as a PNG image in the specified folder.

    Args:
        enabled (bool):
            Whether graph generation is enabled.

        folder_path (str):
            Directory where the graph image will be saved.

        graph_title (str):
            Title of the graph and filename of the output image.

        x_value (float):
            X-axis value for the bar position.

        y_value (float):
            Height of the bar.

        x_label (str):
            Label for the x-axis.

        y_label (str):
            Label for the y-axis.

        y_max (float):
            Maximum allowed y-axis limit.

    Returns:
        None

    Notes:
        A minimum bar height of 0.005 is enforced to ensure visibility for very small values.
        A bar graph is used instead of a line graph to make it clearer what the value is
    """

    # The enabled parameter tracks whether or not the given graph is enabled in settings
    # If it is not enabled, don't graph it
    if not enabled:
        return

    # Graph setup
    fig, ax = plt.subplots(figsize=(4, 6))
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(graph_title)

    # Plot data
    ax.bar([x_value], max(0.005, y_value))
    
    # Setup x and y limits, tick marks
    ax.set_xlim(-1e-12, 1e-12)

    # Disables x ticks
    ax.set_xticks([])
    
    # Keep y limits close to y value, but make sure it is within the range (0, y_max)
    ax.set_ylim(max(0, y_value - 1), min(y_max, y_value + 1))

    # Save image locally in given folder with given title
    plt.tight_layout()
    plt.savefig(f"{folder_path}/{graph_title}.png")
    plt.close()