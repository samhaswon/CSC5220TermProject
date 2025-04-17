import os
import math
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor

# ------------------------------
# DATA LOADING -----------------
# ------------------------------

def load_data(data_dir):
    """
    Loads CSV hist_data from `data_dir` and returns NumPy arrays.
    """
    all_files = [os.path.join(data_dir, f)
                 for f in os.listdir(data_dir) if f.endswith('.csv')]
    df_list = [pd.read_csv(f) for f in all_files]
    data = pd.concat(df_list, ignore_index=True)

    data.columns = data.columns.str.strip()
    data = data.applymap(lambda x: str(x).strip() if isinstance(x, str) else x)

    inputs = ["Engine RPM(rpm)", "Intake Air Temperature(°F)", "Altitude",
              "Relative Throttle Position(%)", "Speed (OBD)(mph)", "Grade"]
    target = ["Fuel used (inst)"]

    data = data[inputs + target]
    data.replace('-', np.nan, inplace=True)
    data = data.apply(pd.to_numeric, errors='coerce').dropna()

    X = data[inputs].astype(np.float32).values
    y = data[target].astype(np.float32).values
    return X, y

# ------------------------------
# PHYSICAL MODEL ---------------
# ------------------------------

def predict_physical(x: np.ndarray) -> np.ndarray:
    """Compute instantaneous fuel flow from physics model."""
    MASS, CD, AREA = 1370.0, 0.27, 2.20
    CRR, TORQUE_MAX, BSFC = 0.010, 115.0, 250.0
    RHO_F, G = 0.745, 9.81

    rpm, Tf, h, thr, spd, grd = x.T
    v = spd * 0.44704
    Tk = (Tf - 32)*5/9 + 273.15
    rho = 1.225*(288.15/Tk)*np.exp(-h/8434)
    theta = np.arctan(grd)

    p_max = (TORQUE_MAX * rpm * 2*math.pi/60)/1000    # kW
    p_eng = (thr/100)*p_max                         # kW

    m_dot = BSFC * p_eng / 3600 / 1000               # kg/s
    v_dot = m_dot / RHO_F                            # L/s
    return v_dot.reshape(-1,1)

# ------------------------------
# EVALUATION -------------------
# ------------------------------

def evaluate(y_true, y_pred):
    return mean_squared_error(y_true, y_pred), r2_score(y_true, y_pred)

# ------------------------------
# ML BASELINES -----------------
# ------------------------------

def train_ml(X, y):
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=42)
    # Linear
    lr = LinearRegression().fit(Xtr, ytr)
    y_lr = lr.predict(Xte)
    mse_lr, r2_lr = evaluate(yte, y_lr)
    # Random Forest
    rf = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    rf.fit(Xtr, ytr.ravel())
    y_rf = rf.predict(Xte).reshape(-1,1)
    mse_rf, r2_rf = evaluate(yte, y_rf)

    print("--- ML BASELINES ---")
    print(f"Linear Regression => MSE: {mse_lr:.6e}, R²: {r2_lr:.4f}")
    print(f"Random Forest     => MSE: {mse_rf:.6e}, R²: {r2_rf:.4f}")
    return Xtr, Xte, ytr, yte, rf

# ------------------------------
# HYBRID PHYSICS+RESIDUAL -----
# ------------------------------

def train_hybrid(Xtr, Xte, ytr, yte, rf_resid):
    # Residuals on train
    y_phys_tr = predict_physical(Xtr)
    resid_tr = ytr - y_phys_tr
    # Fit residual model
    rf_resid.fit(Xtr, resid_tr.ravel())
    # Predict hybrid
    y_phys_te = predict_physical(Xte)
    resid_pred = rf_resid.predict(Xte).reshape(-1,1)
    y_hybrid = y_phys_te + resid_pred
    mse_h, r2_h = evaluate(yte, y_hybrid)
    print("--- HYBRID MODEL ---")
    print(f"MSE: {mse_h:.6e}, R²: {r2_h:.4f}")

# ------------------------------
# MAIN -------------------------
# ------------------------------

def main(data_dir="./cleaned_data"):
    X, y = load_data(data_dir)
    # Physics
    y_p = predict_physical(X)
    mse_p, r2_p = evaluate(y, y_p)
    print("--- PHYSICS MODEL ---")
    print(f"MSE: {mse_p:.6e}, R²: {r2_p:.4f}")

    # ML
    Xtr, Xte, ytr, yte, rf_base = train_ml(X, y)

    # Hybrid
    train_hybrid(Xtr, Xte, ytr, yte, rf_base)

if __name__ == '__main__':
    main()
