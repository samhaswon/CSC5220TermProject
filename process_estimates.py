"""
Processes the estimates from RouteE
"""
import json
import os

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score


if __name__ == '__main__':
    # Dataframe things
    all_files = [
        os.path.join("cleaned_data", f)
        for f in os.listdir("cleaned_data")
        if f.endswith('.csv')
    ]
    dataframes = [pd.read_csv(f) for f in all_files]

    numeric_columns = [
        "Altitude", "Bearing", "Air Fuel Ratio(Measured)(:1)",
        "Engine Load(%)", "Engine RPM(rpm)", "Intake Air Temperature(°F)",
        "Relative Throttle Position(%)", "Speed (OBD)(mph)", "Grade",
        "Fuel used (inst)", "Fuel used (trip)(gal)"
    ]

    # Trim space from column names
    for df in dataframes:
        df.columns = df.columns.str.strip()

    # Trim other spaces from rows
    dataframes = [
        df.map(lambda x: str(x).strip() if isinstance(x, str) else x)
        for df in dataframes
    ]

    # Get the fuel used for each trip
    df_used = []
    for df in dataframes:
        # fuel_used = np.sum(df["Fuel used (inst)"])
        df_used.append(df["Fuel used (trip)(gal)"].iloc[-1])

    # RouteE things
    with open("estimates.json", "r", encoding="utf-8") as json_file:
        estimates = json.load(json_file)
    used = []
    for estimate in estimates:
        used.append(estimate['routee_estimate'])

    df_used_total = sum(df_used)
    used_total = sum(used)

    # Summary
    print(f"{sum(used)=}")
    print(f"{sum(df_used)=}")
    print(f"Off by {abs(df_used_total - used_total)} gallons")
    print(f"Percent error: {abs((used_total - df_used_total) / df_used_total) * 100:.4f}%")
    print(f"R²: {r2_score(df_used, used)}")
