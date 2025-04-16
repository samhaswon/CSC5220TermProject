"""
Clean the data from Torque
"""
from datetime import datetime
import os
import time
from typing import Tuple, Union

import numpy as np
import pandas as pd
from requests_cache import CachedSession


# Setup cache (expires after 30 days)
session = CachedSession('weather_cache', expire_after=86400 * 30)

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


def moving_average(arr: np.array, window_size: int = 599) -> np.array:
    """
    Calculates the moving average of an array over *n* elements.
    :param arr: The NumPy array to average across.
    :param window_size: The window for averaging.
    :returns: The averaged array.
    """
    if window_size % 2 == 0:
        raise ValueError("Window size must be odd to maintain the same element count.")

    kernel = np.ones(window_size) / window_size
    smoothed = np.convolve(arr, kernel, mode='same')
    smoothed = np.round(smoothed, decimals=2)
    return smoothed


def parse_datetime(dt_str: str) -> Tuple[str, int]:
    """
    Parses the Torque datetime string into a date string and the hour.
    :param dt_str: The datetime to be parsed.
    :return: Tuple of date string (for the API) and the hour (for the index).
    """
    # Split string and remove the timezone (EDT)
    parts = dt_str.split()
    cleaned_dt_str = " ".join(parts[:4] + parts[5:])  # Removes the timezone part

    # Convert to datetime object
    dt_obj = datetime.strptime(cleaned_dt_str, "%a %b %d %H:%M:%S %Y")

    return dt_obj.strftime("%Y-%m-%d"), dt_obj.hour  # Returns date and hour


def get_weather(lat: str, lon: str, dt_str: str) -> Union[float, None]:
    """
    Gets weather data for the given location and time.
    :param lat: The latitude.
    :param lon: The longitude.
    :param dt_str: The datetime string of the time.
    :return: The temperature or None.
    """
    date, hour = parse_datetime(dt_str)

    # Open-Meteo API URL
    url = (f"https://archive-api.open-meteo.com/v1/archive?latitude={lat}&longitude={lon}&"
           f"start_date={date}&end_date={date}&hourly=temperature_2m")

    response = session.get(url)  # Cached request
    if response.status_code == 200:
        data = response.json()
        try:
            return data["hourly"]["temperature_2m"][hour]  # Get the closest hour
        except (KeyError, IndexError):
            return None  # Handle missing data
    return None  # API failure


def add_weather(input_df: pd.DataFrame, max_calls: int = 10_000) -> pd.DataFrame:
    """
    Adds the weather data to the given CSV.
    :param input_df: The dataframe to add weather data to.
    :param max_calls: The maximum number of API calls to do for this file.
    :returns: None.
    """

    call_count = 0
    row_num = 0
    temperature_series = pd.Series(dtype=float, name="Temperature (°C)")

    for _, row in input_df.iterrows():
        if row_num % 600 != 0:
            row_num += 1
            continue
        if call_count >= max_calls:
            print("Reached API limit, stopping.")
            break  # Stop if we hit the daily limit

        weather = get_weather(row["Latitude"], row["Longitude"], row["GPS Time"])
        if weather is not None:
            weather_list = [weather for _ in range(600)]
            temperature_series = pd.concat((temperature_series, pd.Series(weather_list)))
            call_count += 1
            if call_count % 100 == 0:  # Avoid rate limits
                time.sleep(1)
        row_num += 1
    temperature_series = np.array(temperature_series)
    temperature_series = moving_average(temperature_series)
    temperature_series = temperature_series[:(len(input_df["Latitude"]) - len(temperature_series))]
    input_df["Temperature (°C)"] = temperature_series
    return input_df


if __name__ == '__main__':
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    start = time.perf_counter()
    # Define the columns to keep (a trimmed version)
    columns_to_keep = [
        "GPS Time", "Device Time", "Longitude", "Latitude",
        "GPS Speed (Meters/second)", "Altitude", "Bearing", "G(x)", "G(y)",
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
            # Find the max used
            if (x := max(df["Fuel used (inst)"])) > max_fuel_used:
                max_fuel_used = x

            # Drop columns without time.
            df = df[df["GPS Time"] != '-']
            df.reset_index()

            # Calculate the grade (slope) of the road.
            df = df[df["Altitude"] != "-"]
            df["Altitude"] = df["Altitude"].apply(pd.to_numeric, errors='coerce')
            df["Speed (OBD)(mph)"] = df["Speed (OBD)(mph)"].apply(pd.to_numeric, errors='coerce')
            df.reset_index()
            # We calculate the change in altitude (rise) for the second,
            # converting to feet.
            # Then we find the distance for the past second with the speed,
            # converting to ft/s.
            # Divide the rise by the run to get the ratio representing grade.
            df["Grade"] = (
                    (df["Altitude"].diff() * 3.280839895) /  # Δm -> Δft
                    (df["Speed (OBD)(mph)"] * 5280 / 3600)   # MPH -> ft/s
            )
            df["Grade"] = df["Grade"].fillna(0)
            # Filter outliers.
            # Grade shouldn't be more than about 12% (0.12),
            # but that is still an outlier for this data.
            df["Grade"] = df["Grade"].clip(lower=-0.12, upper=0.12)
            # Take the derivative of "Grade"
            df["Grade Break"] = df["Grade"].diff()
            df["Grade Break"] = df["Grade Break"].fillna(0)

            df = add_weather(df)

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
