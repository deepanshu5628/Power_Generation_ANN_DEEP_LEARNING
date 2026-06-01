# Power Plant Energy Output Prediction

A deep learning project that predicts the net hourly electrical energy output (PE) of a combined cycle power plant using an Artificial Neural Network (ANN) built with PyTorch.

## Dataset

`powerplant_data.csv` — Contains sensor readings from a power plant.

| Feature | Description |
|--------|-------------|
| AT | Ambient Temperature |
| V | Exhaust Vacuum |
| AP | Ambient Pressure |
| RH | Relative Humidity |
| PE | Net Electrical Energy Output (target) |

## Model Architecture

A fully connected ANN with:
- Input layer: 4 features
- Hidden layer 1: 4 → 6 neurons (ReLU)
- Hidden layer 2: 6 → 6 neurons (ReLU)
- Output layer: 6 → 1 neuron

## Training

- Loss function: MSELoss
- Optimizer: Adam
- Epochs: 100
- Batch size: 32
- Train/Test split: 80/20

## Files

| File | Description |
|------|-------------|
| `PowerPlant.ipynb` | Main notebook with full pipeline |
| `powerplant_data.csv` | Dataset |
| `best_model.pt` | Saved model weights |

## Requirements

```
numpy
pandas
scikit-learn
torch
```

## Usage

1. Run all cells in `PowerPlant.ipynb`
2. The trained model is saved to `best_model.pt`
3. Load the model for inference:

```python
model = ANN()
model.load_state_dict(torch.load("best_model.pt"))
model.eval()
```

