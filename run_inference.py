import torch
import pandas as pd
import pickle
import os
import numpy as np

# -------- PATH SETUP --------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(BASE_DIR, "Data/testmultisensor_faults.xlsx")
MODEL_PATH = os.path.join(BASE_DIR, "best_model.pt")
SCALER_PATH = os.path.join(BASE_DIR, "scaler.pkl")
CFG_PATH = os.path.join(BASE_DIR, "cfg.pkl")

DEVICE = "cpu"

# -------- LOAD FILES --------
scaler = pickle.load(open(SCALER_PATH, "rb"))
cfg = pickle.load(open(CFG_PATH, "rb"))

# -------- LOAD DATA --------
df = pd.read_excel(DATA_PATH)

# -------- BASIC FEATURE (TEMPORARY) --------
# Replace this later with your real feature engineering
X = df.select_dtypes(include=[np.number]).values

X = scaler.transform(X)
X = torch.tensor(X, dtype=torch.float32).unsqueeze(0).to(DEVICE)

# -------- MODEL (minimal dummy structure) --------
class SimpleModel(torch.nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.fc = torch.nn.Linear(input_dim, 2)
    def forward(self, x):
        return self.fc(x)

model = SimpleModel(X.shape[-1])
model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE))
model.eval()

# -------- PREDICT --------
with torch.no_grad():
    out = model(X)
    pred = out.argmax(-1)

print("Prediction:", pred)
