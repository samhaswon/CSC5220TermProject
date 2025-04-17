"""
Plots RPM against MPG
"""
import pandas as pd
# pylint: disable=relative-beyond-top-level
from .basic_plot import basic_plot
from .load_data import load_data


def make_rpm_mpg_plot():
    """
    Plots RPM against MPG
    """
    data = load_data()
    data = data[["Engine RPM(rpm)", "Miles Per Gallon(Instant)(mpg)"]]
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    data = data[data["Miles Per Gallon(Instant)(mpg)"] < 512]

    # Convert RPM to an integer for better grouping
    data["Engine RPM(rpm)"] = data["Engine RPM(rpm)"].astype(int)

    # Group by RPM and compute average MPG
    grouped = data.groupby("Engine RPM(rpm)")["Miles Per Gallon(Instant)(mpg)"].mean()

    basic_plot(
        grouped.index,
        grouped.values,
        "Average MPG",
        "Engine RPM",
        "Average Miles Per Gallon (mpg)",
        "RPM vs. MPG",
        "figures/rpm_vs_mpg.png"
    )
