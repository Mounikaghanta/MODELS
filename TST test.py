
# ============================================================
# CELL 2 — INFERENCE
# Upload: test_no_labels.xlsx, testdt_predictionsonly.xlsx
# ============================================================

import numpy as np, pandas as pd, torch, torch.nn as nn, pickle, warnings
import matplotlib.pyplot as plt
warnings.filterwarnings('ignore')

DEVICE   = 'cuda' if torch.cuda.is_available() else 'cpu'
SAVE_DIR = '/content/tst_ckpt'

cfg        = pickle.load(open(f'{SAVE_DIR}/cfg.pkl',   'rb'))
scaler     = pickle.load(open(f'{SAVE_DIR}/scaler.pkl','rb'))
feat_names = pickle.load(open(f'{SAVE_DIR}/feats.pkl', 'rb'))
W, S, F    = cfg['W'], cfg['S'], cfg['F']
ROLL_W     = cfg['ROLL_W']

SENSOR_COLS = ['T (degC)', 'Tdew (degC)', 'rh (%)']
DT_COLS     = ['T_pred',   'Td_pred',     'RH_pred']
TIMESTAMP   = 'Date Time'

# Rebuild TST (identical to Cell 1)
class PositionalEncoding(nn.Module):
    def __init__(self, d, max_len=512, drop=0.1):
        super().__init__()
        self.drop = nn.Dropout(drop)
        pe  = torch.zeros(max_len, d)
        pos = torch.arange(0, max_len).unsqueeze(1).float()
        div = torch.exp(torch.arange(0, d, 2).float() * (-np.log(10000.0) / d))
        pe[:, 0::2] = torch.sin(pos * div)
        pe[:, 1::2] = torch.cos(pos * div)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x): return self.drop(x + self.pe[:, :x.size(1)])

class TST(nn.Module):
    def __init__(self, n_feat, d, n_heads, n_layers, d_ff, drop):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(n_feat, d), nn.LayerNorm(d))
        self.pe   = PositionalEncoding(d, drop=drop)
        enc = nn.TransformerEncoderLayer(d, n_heads, d_ff, drop,
                                          batch_first=True, norm_first=True)
        self.enc  = nn.TransformerEncoder(enc, n_layers, enable_nested_tensor=False)
        self.head = nn.Sequential(
            nn.LayerNorm(d), nn.Linear(d, d//2), nn.GELU(),
            nn.Dropout(drop), nn.Linear(d//2, 2)
        )
    def forward(self, x):
        return self.head(self.enc(self.pe(self.proj(x))))

model = TST(F, cfg['D'], cfg['H'], cfg['L'], cfg['FF'], cfg['DO']).to(DEVICE)
model.load_state_dict(torch.load(f'{SAVE_DIR}/best_model.pt', map_location=DEVICE))
model.eval()
print('Model loaded ✔')

# Load & feature-engineer test data (same pipeline as training)
sens = pd.read_excel('/content/testmultisensor_faults no labels.xlsx')
dt   = pd.read_excel('/content/dt_test_predictionsonly.xlsx')
sens[TIMESTAMP] = pd.to_datetime(sens[TIMESTAMP], dayfirst=True)
dt[TIMESTAMP]   = pd.to_datetime(dt[TIMESTAMP])

df = pd.merge(
    sens[[TIMESTAMP] + SENSOR_COLS],
    dt[[TIMESTAMP]   + DT_COLS],
    on=TIMESTAMP, how='inner'
).sort_values(TIMESTAMP).reset_index(drop=True)

def build_features(df, sensor_cols, dt_cols, roll_w):
    cols = []
    for sc, dc in zip(sensor_cols, dt_cols):
        res = df[sc] - df[dc]
        cols += [res.values, res.abs().values, (res**2).values,
                 res.rolling(roll_w, min_periods=1).mean().values,
                 res.rolling(roll_w, min_periods=1).std().fillna(0).values,
                 res.abs().rolling(roll_w, min_periods=1).max().values,
                 df[sc].diff().fillna(0).values]
    return np.stack(cols, axis=1).astype(np.float32)

X_raw = build_features(df, SENSOR_COLS, DT_COLS, ROLL_W)
X     = scaler.transform(X_raw).astype(np.float32)
print(f'Test timesteps: {len(X):,}')

# Predict: overlapping windows, vote per timestep
# Each timestep collects votes from all windows that include it
vote_sum   = np.zeros((len(X), 2), dtype=np.float32)
vote_count = np.zeros(len(X), dtype=np.int32)

starts = list(range(0, len(X) - W + 1, S))
if len(X) - W not in starts: starts.append(max(0, len(X) - W))

wins_t = torch.tensor(np.stack([X[s:s+W] for s in starts]), dtype=torch.float32)

with torch.no_grad():
    for i in range(0, len(wins_t), 512):
        logits = model(wins_t[i:i+512].to(DEVICE))        # (B, W, 2)
        probs  = torch.softmax(logits, dim=-1).cpu().numpy()
        for j, s in enumerate(starts[i:i+512]):
            vote_sum[s:s+W]   += probs[j]
            vote_count[s:s+W] += 1

vote_count = vote_count.clip(min=1)
norm_probs = vote_sum / vote_count[:, None]
preds      = np.argmax(norm_probs, axis=1)
p_faulty   = norm_probs[:, 1]

out = pd.DataFrame({
    'Date Time':       df[TIMESTAMP].values,
    'predicted_label': preds,
    'predicted_class': np.where(preds==1, 'faulty', 'clean'),
    'p_faulty':        np.round(p_faulty, 4),
    'p_clean':         np.round(norm_probs[:,0], 4)
})
out.to_csv('TST predicted_labels.csv', index=False)
print(f' predicted_labels.csv saved')
print(out['predicted_class'].value_counts().to_string())
