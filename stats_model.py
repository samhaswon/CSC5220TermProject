import os

import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from scipy.optimize import curve_fit


def torque(cd, a, rho, v, crr, m, grade, r):
    """
    Calculates the torque in newton-meters.
    :param cd: The coefficient of drag.
    :param a: (A) the frontal area in m^2.
    :param rho: Air density in kg/m^3.
    :param v: Velocity in m/s.
    :param crr: The coefficient of rolling resistance.
    :param m: The mass of the vehicle.
    :param grade: The fraction that represents the grade angle.
    :param r: The radius of the tire.
    :returns: Torque required (Nm).
    """
    g = 9.80665  # m/s
    t = (0.5 * cd * a * rho * v ** 2 + crr * m * g + m * g * grade) * r
    return t


def fuel_volume(v_engine, n, m_air, afr):
    """
    Calculates the fuel volume based on displacement, rpm and air.
    :param v_engine: The displacement volume of the engine.
    :param n: The RPM.
    :param m_air: The mass of the air comint into the engine in kg/min.
    :param afr: The air-fuel ratio.
    :returns: Fuel volume (L/min)
    """
    gasoline_density = 0.74  # kg/L
    volume = ((v_engine / 2) * n * m_air) / (afr * gasoline_density)
    return volume


def bsfc(fuel, t, n):
    """
    Calculates the brake-specific fuel consumption.
    :param fuel: The weight of fuel used.
    :param t: The torque required.
    :param n: The RPM.
    """
    if t == 0:
        t = 1E-9
    if n == 0:
        n = 1E-9
    bsfc_val = fuel / ((t * n) / 9550)
    return bsfc_val


def v_fuel(v_engine, n, m_air, afr, cd, a, rho, v, crr, m, grade, r):
    """
    :param v_engine: The engine's volume.
    :param n: The RPM.
    :param m_air: The mass of the air comint into the engine in kg/min.
    :param afr: The air-fuel ratio.
    :param cd: The coefficient of drag.
    :param a: (A) the frontal area in m^2.
    :param rho: Air density in kg/m^3.
    :param v: Velocity in m/s.
    :param crr: The coefficient of rolling resistance.
    :param m: The mass of the vehicle.
    :param grade: The fraction that represents the grade angle.
    :param r: The radius of the tire.
    """
    fuel_volume_value = fuel_volume(v_engine, n, m_air, afr)
    gasoline_density = 0.74  # kg/L
    t = torque(cd, a, rho, v, crr, m, grade, r)
    bsfc_value = bsfc(fuel_volume_value, t, n)
    v_fuel_value = (bsfc_value * t * n) / (9550 * gasoline_density)
    return v_fuel_value


def air_density_std(altitude_m, temp_c):
    """
    Calculate air density at given altitude and temperature,
    using the International Standard Atmosphere up to 11 km.
    :param altitude_m: Geopotential altitude in meters.
    :param temp_c: Ambient temperature in °C (at that altitude).
    :returns: Density in kg/m^3
    """
    # constants
    p0 = 101325.0  # sea-level standard pressure, Pa
    T0 = 288.15  # sea-level standard temp, K
    L = 0.0065  # temperature lapse rate, K/m
    g0 = 9.80665  # gravitational acceleration, m/s²
    R = 287.05  # specific gas constant for dry air, J/(kg·K)

    # convert inputs
    T = temp_c + 273.15  # K
    h = altitude_m  # m

    # pressure via barometric formula (troposphere)
    p = p0 * (1 - L * h / T0) ** (g0 / (R * L))

    # density from ideal gas law
    rho = p / (R * T)
    return rho


def calculate_fuel(x, cd, a, crr, air_r):
    """
    :param n: The RPM.
    :param temp: The temperature (°F).
    :param altitude: The altitude.
    :param throttle: The throttle position.
    :param cd: The coefficient of drag.
    :param a: (A) the frontal area in m^2.
    :param v: The velocity (MPH).
    :param crr: The coefficient of rolling resistance.
    :param air_r: Constant of air flow.
    :param grade: The fraction that represents the grade angle.
    """
    n, temp, altitude, throttle, v, grade = x
    air_density = air_density_std(altitude, (temp - 32) * 5 / 9)
    air_flow_rate = (0.0018 * throttle * cd) / air_r
    fuel_used = v_fuel(
        v_engine=1.8,  # L
        n=n,
        m_air=air_flow_rate,
        afr=14.7,
        cd=0.27,
        a=a,
        rho=air_density,
        v=v / 2.237,  # MPH -> m/s
        crr=crr,
        m=1370,  # 2820 lbs + junk -> kg = 1370
        grade=grade,
        r=0.315976  # P205/55R16
    )
    return fuel_used / 3.785  # L -> gal


def load_data(data_dir):
    """
    Loads CSV hist_data from `DATA_DIR`
    :param data_dir: The directory with the cleaned hist_data CSVs.
    :returns: The numpy arrays of the hist_data.
    """
    all_files = [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.endswith('.csv')]
    dataframes = [pd.read_csv(f) for f in all_files]
    data = pd.concat(dataframes, ignore_index=True)

    # Strip column names of any leading/trailing whitespace
    data.columns = data.columns.str.strip()

    # Strip spaces from all values and replace non-numeric values with NaN
    data = data.map(lambda x: str(x).strip() if isinstance(x, str) else x)

    # Select relevant input features and target variables
    input_features = [
        "Engine RPM(rpm)", "Intake Air Temperature(°F)", "Altitude",
        "Relative Throttle Position(%)", "Speed (OBD)(mph)", "Grade"
    ]
    target_features = ["Fuel used (inst)"]
    all_features = input_features + target_features
    # Extract only what we want
    data = data[all_features]

    # Make the strings into numbers, replacing missing hist_data with nans.
    data.replace('-', np.nan, inplace=True)
    # Convert the string hist_data to float64 for some math
    data = data.apply(pd.to_numeric, errors='coerce')
    # Drop rows with missing hist_data
    data = data.dropna()

    # Cast to float32 for training
    x = data[input_features].astype(np.float32).values
    y = data[target_features].astype(np.float32).values
    return x, y


if __name__ == '__main__':
    X, Y = load_data("./cleaned_data")

    def objective_func(x, cd, a, crr, air_r):
        """
        Perform curve fitting to estimate parameters (cd, a, crr, air_r)
        """
        return np.array([calculate_fuel(xi, cd, a, crr, air_r) for xi in x.T])


    # Fit the model to the data
    params, covariance = curve_fit(
        objective_func,
        X.T,
        Y.flatten(),
        p0=[100, 2.13677, 0.01, 1.0]
    )

    # Extract optimized parameters
    cd, a, crr, air_r = params
    print(f"Optimized parameters: cd={cd}, a={a}, crr={crr}, air_r={air_r}")

    # Predict fuel usage
    predictions = objective_func(X.T, *params)
    Y = Y.flatten()
    mse = mean_squared_error(Y, predictions)
    r2 = r2_score(Y, predictions)
    print(f"MSE: {mse}\n"
          f"R²: {r2}")
