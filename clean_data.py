"""
Clean the data from Torque
"""
import os
import time
import numpy as np
import pandas as pd


# Set the minimum number of rows required for a file to be processed
MIN_ROWS = 15

# Define input and output directories
INPUT_FOLDER = "torqueLogs"
OUTPUT_FOLDER = "cleaned_data"


def is_a_float(potential_float: str) -> bool:
    """
    Attempts to convert the input to a floating point, returning the success of this.
    :param potential_float: The value to attempt to convert.
    :return: Success of conversion.
    """
    try:
        float(potential_float)
        return True
    except ValueError:
        return False


if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    start = time.perf_counter()
    # Define the columns to keep (a trimmed version)
    columns_to_keep = [
        "GPS Time", "Device Time", "Longitude", "Latitude",
        "GPS Speed (Meters/second)", "Altitude", "Bearing", "G(potiential_float)", "G(y)",
        "G(z)", "G(calibrated)", "Air Fuel Ratio(Measured)(:1)", "Engine Load(%)",
        "Engine Load(Absolute)(%)", "Engine RPM(rpm)", "Fuel used (trip)(gal)",
        "GPS Altitude(ft)", "GPS vs OBD Speed difference(mph)",
        "Intake Air Temperature(°F)", "Miles Per Gallon(Instant)(mpg)",
        "Relative Throttle Position(%)", "Speed (GPS)(mph)", "Speed (OBD)(mph)",
        "Trip average MPG(mpg)"
    ]
    # pylint: disable=invalid-name
    max_fuel_used = -1
    # Just to see how many we have
    num_data_points = 0
    # Process each CSV file
    for filename in os.listdir(INPUT_FOLDER):
        if filename.endswith(".csv"):
            file_path = os.path.join(INPUT_FOLDER, filename)
            df = pd.read_csv(file_path)

            # Trim spaces from column names
            df.columns = df.columns.str.strip()

            # Keep only the required columns
            try:
                df = df[columns_to_keep]
            except KeyError as ke:
                print(f"Error for file: {filename}")
                print(ke)

            # Drop duplicate rows based on "GPS Time"
            df = df.drop_duplicates(subset=["GPS Time"]).reset_index()

            # Clean rows where the columns appear again for some reason
            df = df[df["Fuel used (trip)(gal)"].apply(is_a_float)]

            # Skip files that don't meet the row threshold
            if len(df) < MIN_ROWS:
                print(f"Skipping {filename} (too few rows: {len(df)})")
                continue

            # Replace missing numbers with zeros
            df["Fuel used (trip)(gal)"] = df["Fuel used (trip)(gal)"].replace('-', '0')

            # Calculate the fuel used for the last second
            try:
                fuel_next = df["Fuel used (trip)(gal)"].iloc[:-1].astype(float)
            except Exception as ex:
                print(ex)
                raise
            # If more than an ounce of fuel is used in the first row, duplicate the first value
            if fuel_next.iloc[0] > 0.008:
                fuel_next = pd.concat((pd.Series(fuel_next.iloc[0]), fuel_next), ignore_index=True)
            else:
                fuel_next = pd.concat((pd.Series(float(0)), fuel_next), ignore_index=True)
            try:
                df["Fuel used (inst)"] = np.abs(
                    fuel_next - df["Fuel used (trip)(gal)"].astype(float)
                )
            except Exception as ex:
                print(ex)
                raise
            if (x := max(df["Fuel used (inst)"])) > max_fuel_used:
                max_fuel_used = x

            df = df[df["GPS Time"] != '-']
            df.reset_index()

            # Save the cleaned hist_data
            output_path = os.path.join(OUTPUT_FOLDER, filename)
            df.to_csv(output_path, index=False)
            num_data_points += len(df["Latitude"])
            print(f"Processed {filename}: {len(df)} rows saved.")

    end = time.perf_counter()
    print("Cleaning complete.")
    print(f"Max fuel used: {max_fuel_used}")
    print(f"Number of data points: {num_data_points}")
    print(f"Time: {end - start:.4f}s")
