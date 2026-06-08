import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def graph_plot(
        enabled: bool,
        folder_path: str,
        graph_title: str,
        x_label: str,
        y_label: str,
        x_data: list[float],
        y_data: list[float],
        y_max: float
    ) -> None:
    """
    Create and save a line graph displaying the passed in data.
    The chart is saved as a PNG image in the specified folder.
    """

    if not enabled:
        return

    # If there is only one data point make the graph a bar graph instead of a line graph
    if len(x_data) == 1:
        graph_plot_point(enabled, folder_path, graph_title, "", y_label, y_data[0], y_max)
        return

    _, ax = plt.subplots()
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(graph_title)

    ax.plot(x_data, y_data)

    # Keep y limits close to y value, but make sure it is within the range (0, y_max)
    ax.set_ylim(max(0, min(y_data) - 1), min(y_max + 0.025, max(y_data) + 1))

    # Ensure tick marks on x axis are only ever whole numbers
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))

    plt.savefig(f"{folder_path}/{graph_title}.png")
    plt.close()

def graph_plot_point(
        enabled: bool,
        folder_path: str,
        graph_title: str,
        x_label: str,
        y_label: str,
        y_value: float,
        y_max: float
    ) -> None:
    """
    Create and save a bar chart containing a single data point.
    The chart is saved as a PNG image in the specified folder.

    A minimum bar height of 0.005 is enforced to ensure visibility for very small values.
    A bar graph is used instead of a line graph to make it clearer what the value is.
    """

    if not enabled:
        return

    fig, ax = plt.subplots(figsize=(4, 6))    
    ax.set_xlabel(x_label)
    ax.set_ylabel(y_label)
    plt.title(graph_title)

    ax.bar(0, max(0.005, y_value))

    # Disables x ticks because there is no x value to be displayed
    ax.set_xticks([])
    
    # Keep y limits close to y value, but make sure it is within the range (0, y_max)
    ax.set_ylim(max(0, y_value - 1), min(y_max, y_value + 1))

    # The image size of the graph has been modified, so tight_layout ensures the image saved matches the size of the graph well
    plt.tight_layout()

    plt.savefig(f"{folder_path}/{graph_title}.png")
    plt.close()