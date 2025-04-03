"""
Creates histograms of various variables
"""
import os
import re
from typing import Tuple
import pandas as pd
import matplotlib.pyplot as plt

from .load_data import load_data


def create_histogram(
        hist_data: pd.DataFrame,
        column: str,
        hist_filename: str,
        size: Tuple[int, int]=(10, 5)
) -> None:
    """
    Creates and saves a histogram for the specified column.
    :param hist_data: The dataframe of all the data.
    :param column: The column to generate a histogram for.
    :param hist_filename: The name of the file for saving the plot.
    :param size: The size of the figure.
    """
    # Extract the hist_data
    series = hist_data[column]

    # Plot settings
    plt.figure(figsize=size)
    plt.hist(series, bins='auto', edgecolor='black', alpha=0.7)
    plt.xlabel(column)
    plt.ylabel("Frequency")
    plt.title(f"Distribution of {column}")
    plt.grid(True)

    # Ensure directory exists and save
    os.makedirs(os.path.dirname(hist_filename), exist_ok=True)
    plt.savefig(hist_filename)
    plt.close()


def make_histograms():
    """
    Creates histograms of various variables
    """
    data = load_data()
    columns_to_plot = [
        "Intake Air Temperature(°F)",
        "Miles Per Gallon(Instant)(mpg)",
        "Speed (OBD)(mph)",
        "Engine Load(Absolute)(%)",
        "Engine RPM(rpm)",
        "Fuel used (inst)",
    ]

    # Include all three target columns in preprocessing
    data = data[columns_to_plot]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    columns_to_plot.remove("Fuel used (inst)")

    # Convert temperature to integer
    data["Intake Air Temperature(°F)"] = data["Intake Air Temperature(°F)"].astype(int)

    for col in columns_to_plot:
        # Sanitize hist_filename: lowercase, replace spaces with underscores, remove special chars
        sanitized = re.sub(r'\W', '', col).replace('_', '').lower()
        filename = f"figures/histogram_{sanitized}.png"
        create_histogram(data, col, filename)

    # Pull out fuel used separately due to its outliers.
    sanitized = re.sub(r'\W', '', "Fuel used (inst)").replace('_', '').lower()
    filename = f"figures/histogram_{sanitized}.png"
    fuel_used_df = data[data["Fuel used (inst)"] < 0.0028]
    create_histogram(fuel_used_df, "Fuel used (inst)", filename, size=(10, 10))

