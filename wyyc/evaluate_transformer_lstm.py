import os
import time
import torch
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import joblib

from model import TransformerLSTM
from config import *

# 1. 加载数据
data_path = 'data/all_data_water_tp_new.csv'
data = pd.read_csv(data_path)

# 2. 特征与目标
features = ['PAC1','PAC2','PAM1','PAM2','TC1_FILTER_EFFTP', 'TC1_SDS_EFFTP',
            'influent_Ammonia', 'influent_COD', 'influent_PH', 'OxidationDitch_MLSS']
target_col = 'Effluent_TP'  # 目标列

# 只保留需要的列
data = data.dropna(subset=features + [target_col])
X = data[features].values.astype(np.float32)
y = data[target_col].values.astype(np.float32)

# 3. 标准化
scaler = joblib.load(SCALER_PATH_TP)
X_scaled = np.zeros_like(X)
for col_idx in range(X.shape[1]):
    col_data = X[:, col_idx].reshape(-1, 1)
    X_scaled[:, col_idx] = scaler[col_idx].transform(col_data).flatten()

# 4. 构造序列数据
seq_length = SUQUENCE_LENGTH
X_seq = []
y_seq = []
for i in range(len(X_scaled) - seq_length - OUTPUT_FEATURES + 1):
    X_seq.append(X_scaled[i:i+seq_length])
    y_seq.append(y[i+seq_length:i+seq_length+OUTPUT_FEATURES])
X_seq = np.array(X_seq)
y_seq = np.array(y_seq)

# 5. 加载模型
input_size = len(features)
model = TransformerLSTM(
    input_dim=input_size,
    hidden_dim=HIDDEN_LAYERS,
    output_dim=OUTPUT_FEATURES,
    num_layers=TRANSFORMER_LAYERS,
    d_model=TRANSFORMER_D_MODEL,
    nhead=TRANSFORMER_NHEAD
)
model.load_state_dict(torch.load('models/train_outTransformerLSTM_TP_14.pth', map_location='cpu'))
model.eval()

# 6. 推理与评测
batch_size = 64
num_batches = int(np.ceil(len(X_seq) / batch_size))
preds = []
start_time = time.time()
with torch.no_grad():
    for i in range(num_batches):
        batch_x = X_seq[i*batch_size:(i+1)*batch_size]
        batch_x_tensor = torch.tensor(batch_x, dtype=torch.float32)
        batch_pred = model(batch_x_tensor).numpy()
        preds.append(batch_pred)
preds = np.concatenate(preds, axis=0)
end_time = time.time()

# 7. 评估指标
# 只评估第一个输出步长（如只关心t+1）
y_true = y_seq[:, 0]
y_pred = preds[:, 0]
mse = mean_squared_error(y_true, y_pred)
mae = mean_absolute_error(y_true, y_pred)
r2 = r2_score(y_true, y_pred)

# 8. 推理速度
inference_time = end_time - start_time
avg_time_per_sample = inference_time / len(X_seq)

# 9. 模型参数量
param_count = sum(p.numel() for p in model.parameters())

# 10. 输出评测报告
print('==== Transformer+LSTM模型自动化评测报告 ====')
print(f'样本数: {len(X_seq)}')
print(f'MSE: {mse:.6f}')
print(f'MAE: {mae:.6f}')
print(f'R2: {r2:.4f}')
print(f'推理总耗时: {inference_time:.2f} 秒')
print(f'平均每个样本耗时: {avg_time_per_sample*1000:.2f} 毫秒')
print(f'模型参数量: {param_count}')
print('==========================================') 