from fvcore.nn import FlopCountAnalysis
import torch

from mpg_rnn.fuel_mpg_rnn import FuelMPGRNN


# Define model parameters
INPUT_SIZE = 10  # Number of input features
HIDDEN_SIZE = 64
NUM_LAYERS = 6
OUTPUT_SIZE = 1  # Predicting 1 variable

model = FuelMPGRNN(
    input_size=INPUT_SIZE,
    hidden_size=HIDDEN_SIZE,
    num_layers=NUM_LAYERS,
    output_size=OUTPUT_SIZE
)
input_tensor = torch.randn((1, 10, 10))

lstm_flops = ((4 * (INPUT_SIZE + HIDDEN_SIZE) * HIDDEN_SIZE) * 10 +  # Layer 1
              4 * (HIDDEN_SIZE * 2) * HIDDEN_SIZE * 10 * 5           # Layers 2-6
              )

flops = FlopCountAnalysis(model, input_tensor)
print(f"Total FLOPs: {flops.total() + lstm_flops:,} ({lstm_flops=})")
