import os
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from torch.utils.data import TensorDataset
from sklearn.model_selection import GridSearchCV, train_test_split
from joblib import dump
import torch
import torch .nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from config import *
from model import TransformerLSTM

# 设备指定
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# 确保模型存储目录存在
os.makedirs("models", exist_ok=True)


def load_data(file_path):
    data = pd.read_csv(file_path, parse_dates=['_time'])
    data.set_index('_time', inplace=True)
    return data


def preprocess_data(data, seq_length):
    # 选择特征和目标变量
    features = data.drop(columns=['2#SurfaceAerator', '1#SurfaceAerator', 'HighTempEle', 'NaClOEle']).values
    target = data['Effluent_TP'].values
    scaler_path = SCALER_PATH_TP
    scalers = {}

    # 对每个特征单独进行标准化
    features_scaled = np.zeros_like(features)  # 创建一个与 features 形状相同的数组用于存储标准化后的数据
    for i in range(features.shape[1]):
        scaler = StandardScaler()
        features_scaled[:, i] = scaler.fit_transform(features[:, [i]].reshape(-1, 1)).flatten()
        scalers[i] = scaler  # 保存每个特征的标准化器

    # 保存所有标准化器到一个文件
    joblib.dump(scalers, scaler_path)

    # 将数据转换为适合RNN的格式 (samples, timesteps, features)
    def create_sequences(data, target, seq_length, output_features=OUTPUT_FEATURES):
        xs, ys = [], []
        for i in range(len(data) - seq_length - output_features + 1):
            x = data[i:(i + seq_length)]
            y = target[i + seq_length:i + seq_length + output_features]  # 取output_features步
            xs.append(x)
            ys.append(y)
        return np.array(xs), np.array(ys)

    X, y = create_sequences(features_scaled, target, seq_length, OUTPUT_FEATURES)
    print(X.shape, y.shape)  # (样本数, seq_length, 特征数), (样本数, 6)

    # 转换为PyTorch张量
    X = torch.tensor(X, dtype=torch.float32).to(device)
    y = torch.tensor(y, dtype=torch.float32).to(device)  # 不再view(-1, 1)，直接保持(batch, 6)

    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # # 数据增强：加噪声
    # def augment_with_noise(X, noise_std=0.08):
    #     # 生成与X形状相同的噪声
    #     noise = torch.randn_like(X) * noise_std
    #     # 将噪声添加到X上
    #     return X + noise

    # # 只对训练集增强
    # aug_X = augment_with_noise(X_train, noise_std=0.08)
    # X_train = torch.cat([X_train, aug_X], dim=0)
    # y_train = torch.cat([y_train, y_train], dim=0)

    return X_train, X_test, y_train, y_test, scaler


def train_effluent_tn_model():
    """训练出水总氮模型"""
    try:
        data = pd.read_csv("data/all_data_water.csv")
        data = data.dropna()  # 去除缺失值

        target = "Effluent_TN"
        features = [
            "PAC1","PAC2","PAM1","PAM2","TC1_FILTER_EFFTP","TC1_SDS_EFFTP","influent_Ammonia",
            "influent_COD","influent_PH","OxidationDitch_MLSS"
        ]

        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        model_path = f"models/model_{target}.joblib"
        dump(model, model_path)
        print(f"Model for {target} trained and saved to {model_path}")
    except Exception as e:
        print(f"Error training model for {target}: {e}")
        raise


def train_effluent_tp_model():
    """训练出水总磷模型"""
    try:
        data = pd.read_csv("data/all_data_water.csv")
        data = data.dropna()  # 去除缺失值

        target = "Effluent_TP"
        features = [
            "PAC1","PAC2","PAM1","PAM2","TC1_FILTER_EFFTP","TC1_SDS_EFFTP","influent_Ammonia",
            "influent_COD","influent_PH","OxidationDitch_MLSS"
        ]

        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        model_path = f"models/model_{target}.joblib"
        dump(model, model_path)
        print(f"Model for {target} trained and saved to {model_path}")
    except Exception as e:
        print(f"Error training model for {target}: {e}")
        raise


def train_effluent_ammonia_model():
    """训练出水氨氮模型"""
    try:
        data = pd.read_csv("data/all_data_water.csv")
        data = data.dropna()  # 去除缺失值

        target = "Effluent_Ammonia"
        features = [
            "PAC1","PAC2","PAM1","PAM2","TC1_FILTER_EFFTP","TC1_SDS_EFFTP","influent_Ammonia",
            "influent_COD","influent_PH","OxidationDitch_MLSS"
        ]

        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        model_path = f"models/model_{target}.joblib"
        dump(model, model_path)
        print(f"Model for {target} trained and saved to {model_path}")
    except Exception as e:
        print(f"Error training model for {target}: {e}")
        raise


def train_effluent_cod_model():
    """训练出水 COD 模型"""
    try:
        data = pd.read_csv("data/all_data_water.csv")
        data = data.dropna()  # 去除缺失值

        target = "Effluent_COD"
        features = [
            "PAC1","PAC2","PAM1","PAM2","TC1_FILTER_EFFTP","TC1_SDS_EFFTP","influent_Ammonia",
            "influent_COD","influent_PH","OxidationDitch_MLSS"
        ]

        X = data[features]
        y = data[target]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)

        model_path = f"models/model_{target}.joblib"
        dump(model, model_path)
        print(f"Model for {target} trained and saved to {model_path}")
    except Exception as e:
        print(f"Error training model for {target}: {e}")
        raise


def trainTransformerLstmOutTpPred():
    file_path = 'data/all_data_water_tp_new.csv'
    data = load_data(file_path)

    # 数据预处理
    X_train, X_test, y_train, y_test, scaler = preprocess_data(data, SUQUENCE_LENGTH)

    # 创建数据加载器
    train_dataset = TensorDataset(X_train, y_train)
    test_dataset = TensorDataset(X_test, y_test)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE_TR, shuffle=True, drop_last=True)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE_TR, shuffle=False, drop_last=True)

    # 初始化Transformer-LSTM模型
    input_size = X_train.shape[2]
    print('X_train',X_train)
    model = TransformerLSTM(
        input_dim=input_size,
        hidden_dim=HIDDEN_LAYERS,
        output_dim=OUTPUT_FEATURES,
        num_layers=TRANSFORMER_LAYERS,
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD
    )
    model = model.to(device)

    # 定义损失函数和优化器
    # 修改损失函数，添加数值稳定性
    criterion = nn.MSELoss(reduction='mean')
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE_TR, eps=1e-8)

    # 添加学习率调度器
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)

    # 训练循环，加入早停机制
    best_val_loss = float('inf')
    best_model_state = None
    patience =  2  # 早停的耐心值
    epochs_no_improve = 0
    model.train()
    print(f'开始训练模型')
    for epoch in range(EPOCHS_TR):
        total_loss = 0
        for batch_x, batch_y in train_loader:
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)
            optimizer.zero_grad()

            # 检查输入数据
            if torch.isnan(batch_x).any() or torch.isnan(batch_y).any():
                print("警告：输入数据包含 NaN")
                continue

            outputs = model(batch_x)
            # 原始损失
            loss_main = criterion(outputs, batch_y)
            # 平滑损失：二阶差分，鼓励输出序列平滑
            if outputs.shape[1] > 2:
                smooth_loss = ((outputs[:, 2:] - 2 * outputs[:, 1:-1] + outputs[:, :-2]) ** 2).mean()
            else:
                smooth_loss = 0.0
            # 总损失 = 原损失 + 平滑损失权重*平滑损失
            smooth_weight = 0.1  # 可调整
            loss = loss_main + smooth_weight * smooth_loss

            # 检查损失值
            if torch.isnan(loss):
                print("警告：损失值为 NaN")
                continue

            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # 检查梯度
            for name, param in model.named_parameters():
                if param.grad is not None:
                    if torch.isnan(param.grad).any():
                        print(f"警告：{name} 的梯度包含 NaN")
                        continue

            optimizer.step()

            total_loss += loss.detach().item()
        avg_loss = total_loss / len(train_loader)
        scheduler.step(avg_loss)  # 更新学习率
        current_lr = scheduler.get_last_lr()[0]

        # 验证集评估
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for val_x, val_y in test_loader:
                val_x = val_x.to(device)
                val_y = val_y.to(device)
                val_outputs = model(val_x)
                v_loss = criterion(val_outputs, val_y)
                val_loss += v_loss.item()
        val_loss /= len(test_loader)
        model.train()

        print(f'Epoch {epoch+1}/{EPOCHS_TR} | Train Loss: {avg_loss:.4f} | Val Loss: {val_loss:.4f} | LR: {current_lr:.6f}')

        # 早停判断
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= patience:
                print(f'验证集损失连续{patience}个epoch未提升，提前停止训练。')
                break

    # 保存最佳模型
    if best_model_state is not None:
        torch.save(best_model_state, f'models/train_outTransformerLSTM_TP_{SUQUENCE_LENGTH}.pth')
        print(f'已保存最优模型 models/train_outTransformerLSTM_TP_{SUQUENCE_LENGTH}.pth')
    else:
        torch.save(model.state_dict(), f'models/train_outTransformerLSTM_TP_{SUQUENCE_LENGTH}.pth')
        print(f'已保存最终模型 models/train_outTransformerLSTM_TP_{SUQUENCE_LENGTH}.pth')


if __name__ == "__main__":
    # 训练所有模型
    #train_effluent_tn_model()
    #train_effluent_tp_model()
    #train_effluent_ammonia_model()
    #train_effluent_cod_model()
    trainTransformerLstmOutTpPred()
    