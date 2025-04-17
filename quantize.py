"""
Get the stats of the model on the whole of the dataset.
"""
import time

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from data_loading.vehicle_dataset import VehicleDataset

# Load dataset
DATA_DIR = "cleaned_data"
SEQUENCE_LENGTH = 10
dataset = VehicleDataset(
    DATA_DIR,
    sequence_length=SEQUENCE_LENGTH,
    do_weather=True
)
dataloader = DataLoader(
    dataset,
    batch_size=1024,
    shuffle=False
)

# Define model parameters
INPUT_SIZE = 10  # Number of input features
HIDDEN_SIZE = 64
NUM_LAYERS = 6
OUTPUT_SIZE = 1  # Predicting 1 variable

print(f"Params\n"
      f"{INPUT_SIZE=}\n"
      f"{HIDDEN_SIZE=}\n"
      f"{NUM_LAYERS=}\n"
      f"{OUTPUT_SIZE=}\n")

# Initialize model
device = torch.device("cpu")
model = torch.load("checkpoint.pth", weights_only=False, map_location=device)
model = torch.quantization.quantize_dynamic(
    model,
    {torch.nn.LSTM, torch.nn.Linear},
    dtype=torch.qint8
)
model = model.to(device)

start = time.perf_counter()
model.eval()
eval_targets = []
eval_predictions = []

# Get model predictions
progress_bar = tqdm(total=len(dataloader), desc="Evaluation")
# Eval, so don't bother with gradients
with torch.no_grad():
    for inputs, targets in dataloader:
        # Tensors to device
        inputs, targets = inputs.to(device), targets.to(device)

        # Predict it
        outputs = model(inputs)

        # Save outputs and targets
        eval_targets.append(targets.cpu().numpy())
        eval_predictions.append(outputs.detach().cpu().numpy())
        progress_bar.update(1)
progress_bar.close()

eval_targets = np.vstack(eval_targets).squeeze(1)
eval_predictions = np.vstack(eval_predictions).squeeze(1)

r2_score = r2_score(eval_targets, eval_predictions)
mse = mean_squared_error(eval_targets, eval_predictions)
mae = mean_absolute_error(eval_targets, eval_predictions)
sse = np.sum((eval_targets - eval_predictions) ** 2)
sae = np.sum(np.abs(eval_targets - eval_predictions))
absolute_error = np.sum(  # ∑ err
    np.abs(
        # | |a| - |b| |
        np.abs(
            eval_predictions *
            # Un-norm the data
            (dataset.fuel_range[1] - dataset.fuel_range[0]) + dataset.fuel_range[0]
        ) -
        np.abs(
            eval_targets *
            # Un-norm the data
            (dataset.fuel_range[1] - dataset.fuel_range[0]) + dataset.fuel_range[0]
        )
    )
)
total_difference = (
    # |a| - |b|
    np.sum(
        np.abs(
            eval_predictions *
            # Un-norm the data
            (dataset.fuel_range[1] - dataset.fuel_range[0]) + dataset.fuel_range[0]
        )
    ) -
    np.sum(
        np.abs(
            eval_targets *
            # Un-norm the data
            (dataset.fuel_range[1] - dataset.fuel_range[0]) + dataset.fuel_range[0]
        )
    )
)

fuel_total = np.sum(
    eval_targets *
    # Un-norm the data
    (dataset.fuel_range[1] - dataset.fuel_range[0]) + dataset.fuel_range[0]
)
fuel_total_hat = np.sum(
    eval_predictions *
    # Un-norm the data
    (dataset.fuel_range[1] - dataset.fuel_range[0]) + dataset.fuel_range[0]
)

print(
    f"Absolute Errors: {absolute_error:.6f} gallons\n"
    f"MAE: {mae:.6f}\n"
    f"MSE: {mse:.6f}\n"
    f"R² score: {r2_score:.6f}\n"
    f"Sum of Absolute Errors (SAE): {sae:.6f}\n"
    f"Sum Squared Errors (SSE): {sse:.6f}\n"
    f"Total difference: {total_difference:.6f} gallons, | predicted total - actual total |\n"
    f"Fuel total: {fuel_total:.4f} gallons\n"
    f"Fuel total hat: {fuel_total_hat:.4f} gallons\n"
    f"Total error percent: {abs((fuel_total_hat - fuel_total) / fuel_total) * 100:.4f}%"
)
