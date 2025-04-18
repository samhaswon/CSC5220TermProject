"""
Finds the optimal throttle position for efficient acceleration.
"""
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def load_data(data_dir):
    """
    Loads CSV hist_data from `data_dir`
    :param data_dir: The directory with the cleaned hist_data CSVs.
    :returns: A pandas DataFrame with selected, cleaned columns.
    """
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    dataframes = [pd.read_csv(f) for f in all_files]
    data = pd.concat(dataframes, ignore_index=True)

    # Clean column names and values
    data.columns = data.columns.str.strip()
    data = data.map(lambda x: str(x).strip() if isinstance(x, str) else x)

    # Select relevant input features and target
    input_features = [
        "Engine RPM(rpm)", "Intake Air Temperature(°F)", "Altitude",
        "Relative Throttle Position(%)", "Speed (OBD)(mph)", "Grade"
    ]
    target_features = ["Fuel used (inst)"]
    all_features = input_features + target_features

    data = data[all_features]
    data.replace('-', np.nan, inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce')
    data = data.dropna()
    return data


def analyze_throttle_efficiency(df_i, speed_bin=5.0, throttle_bin=5.0):
    """
    Finds most fuel-efficient throttle setting for accelerating to a given speed.
    :param df_i: DataFrame from `load_data`
    :param speed_bin: Bin width in mph for speed
    :param throttle_bin: Bin width in % for throttle
    :returns: DataFrame of best throttle by speed bin
    """
    df_i = df_i.copy()

    # Compute speed delta to detect acceleration
    df_i["speed_diff"] = df_i["Speed (OBD)(mph)"].diff()

    # Acceleration by at lest 1.5 MPH/s
    df_acc = df_i[df_i["speed_diff"] > 1.5].copy()

    # Filter for near-level ground with a 1% grade
    df_acc = df_acc[df_acc["Grade"].abs() < 0.01]

    # Filter coasting and idling data
    df_acc = df_acc[df_acc["Fuel used (inst)"] > 0.00050192]

    # It takes ~20% throttle to maintain speed, so make that a minimum
    df_acc = df_acc[df_acc["Relative Throttle Position(%)"] > 20.0]

    # Bin speeds and throttle
    df_acc["speed_bin"] = (df_acc["Speed (OBD)(mph)"] // speed_bin) * speed_bin
    df_acc["throttle_bin"] = (
            (df_acc["Relative Throttle Position(%)"] // throttle_bin) * throttle_bin
    )

    # Group and average fuel usage
    grouped = (
        df_acc
        .groupby(["speed_bin", "throttle_bin"])["Fuel used (inst)"]
        .mean()
        .reset_index(name="avg_fuel_rate")
    )

    # Find the throttle bin with the lowest fuel usage per speed bin
    best = (
        grouped
        .loc[grouped.groupby("speed_bin")["avg_fuel_rate"].idxmin()]
        .sort_values("speed_bin")
        .reset_index(drop=True)
    )

    return best


def plot_best_throttle_curve(best_df):
    """
    Plots the most fuel-efficient throttle setting vs. speed.
    :param best_df: DataFrame with speed_bin, throttle_bin, avg_fuel_rate
    """
    plt.figure(figsize=(10, 6))
    plt.plot(best_df["speed_bin"], best_df["throttle_bin"], marker='o')
    plt.xlabel("Speed (mph)")
    plt.ylabel("Most Efficient Throttle Position (%)")
    plt.title("Most Fuel-Efficient Acceleration Throttle Position by Speed")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("./figures/acceleration.png")
    plt.show()


if __name__ == '__main__':
    # Load the data
    DATA_DIR = "./cleaned_data"
    df = load_data(DATA_DIR)

    # Figure out the most efficient throttle positions
    best_throttle_by_speed = analyze_throttle_efficiency(df)

    print("Most fuel-efficient throttle settings while accelerating:")
    print(best_throttle_by_speed)

    # Plot it.
    plot_best_throttle_curve(best_throttle_by_speed)
