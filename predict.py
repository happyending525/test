# predict.py
import os

import joblib
import numpy as np
import pandas as pd
import torch
from joblib import load
from pydantic import BaseModel, Field

from config import *
from model import TransformerLSTM

from collections import deque
import time
from datetime import datetime, timedelta
from influxdb_client import InfluxDBClient
import logging
from typing import Optional
from config import PHOSPHORUS_CONTROLLER_CONFIG


# 定义输入数据结构
class InputData(BaseModel):
    PAC1: float = Field(None, alias="PAC1", description="聚合氯化铝1投加量（mg/L）")
    PAC2: float = Field(None, alias="PAC2", description="聚合氯化铝2投加量（mg/L）")
    PAM1: float = Field(None, alias="PAM1", description="聚丙烯酰胺1投加量（mg/L）")
    PAM2: float = Field(None, alias="PAM2", description="聚丙烯酰胺2投加量（mg/L）")
    TC1_FILTER_EFFTP: float = Field(None, alias="TC1_FILTER_EFFTP", description="TC1过滤器后出水总磷（mg/L）")
    TC1_SDS_EFFTP: float = Field(None, alias="TC1_SDS_EFFTP", description="TC1沉淀池后出水总磷（mg/L）")
    #effluent_tp: Optional[float] = Field(None, alias="EffTp", description="出水总磷浓度（mg/L）")
    influent_Ammonia: float = Field(None, alias="influent_Ammonia", description="进水氨氮浓度（mg/L）")
    influent_COD: float = Field(None, alias="influent_COD", description="进水化学需氧量（mg/L）")
    influent_PH: float = Field(None, alias="influent_PH", description="进水pH值", ge=0, le=14)
    OxidationDitch_MLSS: float = Field(None, alias="OxidationDitch_MLSS", description="氧化沟混合液悬浮固体浓度（mg/L）")
# 定义固定的特征顺序
FEATURES = ["PAC1","PAC2","PAM1","PAM2","TC1_FILTER_EFFTP","TC1_SDS_EFFTP","influent_Ammonia","influent_COD","influent_PH","OxidationDitch_MLSS"]

def load_model(target):
    """加载指定目标变量的模型"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    model_file = os.path.join(current_dir, "models", f"model_{target}.joblib")
    try:
        return load(model_file)
    except Exception as e:
        print(f"Error loading model {model_file}: {e}")
        raise


#----------------随机森林-------------------

def predict(model, input_data: InputData, target_name: str):
    """进行预测"""

    print(target_name)
    input_dict = input_data.model_dump()
    print(input_dict)
    input_df = pd.DataFrame([input_dict])
    features_to_use = [col for col in FEATURES if col != target_name]
    print(features_to_use)
    input_df = input_df[features_to_use]  # 确保特征顺序一致
    try:
        prediction = model.predict(input_df)[0]
        return {
            "code": 0,
            "version":"1.0.0",
            "data": [{
                f"predictOut{target_name}": round(prediction, 3),
                f"targetOut{target_name}": 0.275,
                f"targetOutRange": 0.025
            }]
        }
    except Exception as e:
        raise Exception(f"Error during prediction: {str(e)}")


#----------------transformer+lstm-------------------

def transformlstm_predict_water_tp(custom_data):
    features = ['pac1','pac2','pam1','pam2','TC1_FILTER_EFFTP', 'TC1_SDS_EFFTP', 'Effluent_TP',
                'influent_Ammonia', 'influent_COD', 'influent_PH', 'OxidationDitch_MLSS']


    scaler = joblib.load(SCALER_PATH_TP)
    # print(type(scaler))
    # print("custom_data",type(custom_data),custom_data)
    # X_scaled = scaler.transform(custom_data)
    custom_data = np.array(custom_data)
    n_samples, n_features = custom_data.shape
    X_scaled = np.zeros_like(custom_data, dtype=np.float32)  # 初始化标准化后的数据

    # 按列逐个应用对应的缩放器
    for col_idx in range(n_features):
        # 提取当前列数据（形状：(n_samples, 1)）
        col_data = custom_data[:, col_idx].reshape(-1, 1)
        # 使用该列对应的缩放器进行标准化
        scaled_col = scaler[col_idx].transform(col_data)
        # 将结果放回对应列
        X_scaled[:, col_idx] = scaled_col.flatten()
    # 创建序列数据
    seq_length = SUQUENCE_LENGTH
    X_sequence = []
    for i in range(len(X_scaled) - seq_length + 1):
        X_sequence.append(X_scaled[i:i + seq_length])
    X_sequence = np.array(X_sequence)

    if X_sequence.shape[0] == 0:
        X_sequence = X_scaled[-seq_length:].reshape(1, seq_length, -1)
    else:
        X_sequence = X_sequence[-1:]

    input_tensor = torch.tensor(X_sequence, dtype=torch.float32)

    # 加载transformer-LSTM模型
    input_size = len(features)
    print("input_size",input_size)
    model = TransformerLSTM(
        input_dim=input_size,
        hidden_dim=HIDDEN_LAYERS,
        output_dim=OUTPUT_FEATURES,
        num_layers=TRANSFORMER_LAYERS,
        d_model=TRANSFORMER_D_MODEL,
        nhead=TRANSFORMER_NHEAD
    )
    model.load_state_dict(torch.load('models/train_outTransformerLSTM_TP_14.pth'))
    model.eval()


    with torch.no_grad():
        prediction = model(input_tensor)

    return  prediction.squeeze().tolist()

#----------------智能控制-------------------
class PhosphorusSmartController:
    def __init__(self, config):
        self.config = config
        self._init_parameters()
    def _init_parameters(self):
        self.current_dose = self.config['init_dose']
        self.history = deque(maxlen=self.config['history_size'])
        self.last_adjustment = 0
    def update_control(self, measured_value, predicted_values=None):
        try:
            self.history.append((time.time(), measured_value))
            status = self._analyze_status(measured_value)
            trend = self._analyze_trend()
            self._smart_adjust(status, trend, predicted_values)
            return {
                'current_dose': self.current_dose,
                'status': status,
                'trend': trend,
                'timestamp': time.time()
            }
        except Exception as e:
            print(f"控制更新时发生错误: {str(e)}")
            return None
    #----------------分析状态-------------------
    def _analyze_status(self, value):
        if value < self.config['safe_min']:
            return "过低"
        elif value > self.config['emergency_th']:
            return "紧急"
        elif value > self.config['safe_max']:
            return "偏高"
        elif abs(value - self.config['target']) <= 0.02:
            return "理想"
        else:
            return "正常"
    #----------------分析趋势-------------------
    def _analyze_trend(self):
        if len(self.history) < self.config['trend_window']:
            return "未知"
        recent_values = [item[1] for item in list(self.history)[-self.config['trend_window']:]]
        changes = [recent_values[i+1] - recent_values[i] for i in range(len(recent_values)-1)]
        avg_change = sum(changes) / len(changes)
        if avg_change > 0.01:
            return "上升"
        elif avg_change < -0.01:
            return "下降"
        else:
            return "平稳"
    #----------------智能调整-------------------
    def _smart_adjust(self, status, trend, predicted_values):
        # 如果状态为"紧急"
        if status == "紧急":
            # 将当前剂量设置为最大剂量
            self.current_dose = self.config['max_dose']
            return
        # 如果预测值存在且长度大于等于2，且前两个预测值中有任何一个大于安全最大值
        if predicted_values and len(predicted_values) >= 2 and any(
                p > self.config['safe_max'] for p in predicted_values[:2]):
            # 将当前剂量增加基础步长的两倍，但不超过最大剂量
            self.current_dose = min(self.current_dose + self.config['base_step'] * 2, self.config['max_dose'])
            return
        # 根据状态和趋势确定调整量
        adjustment = self._determine_adjustment(status, trend)
        # 计算步长大小
        step = self._calculate_step_size(adjustment)
        # 应用调整量
        self._apply_adjustment(adjustment, step)
    #----------------确定调整-------------------
    def _determine_adjustment(self, status, trend):
        if status == "偏高":
            if trend == "下降":
                return -1
            return 1
        elif status == "过低":
            return -1
        elif status == "理想":
            if trend == "未知":
                return 0
            if self.current_dose > self.config['min_dose']:
                return -1
        elif status == "正常":
            if trend == "上升":
                return 1
            elif trend == "下降":
                return -1
        return 0
    def _calculate_step_size(self, adjustment):
        # 获取基础步长
        step = self.config['base_step']

        # 如果调整值与上一次调整值相同，则步长变为原来的1.5倍
        if adjustment == self.last_adjustment:
            step = int(step * 1.5)

        # 如果调整值与上一次调整值相反，则步长变为原来的0.7倍
        elif adjustment == -self.last_adjustment:
            step = int(step * 0.7)

        # 返回步长的最大值和最小值限制在5到50之间
        return max(5, min(50, step))
    def _apply_adjustment(self, adjustment, step):
        # 如果调整值为1
        if adjustment == 1:
            # 计算新的剂量，不超过最大剂量
            new_dose = min(self.current_dose + step, self.config['max_dose'])
            # 如果新的剂量与当前剂量不同
            if new_dose != self.current_dose:
                # 记录最后一次调整为正调整
                self.last_adjustment = 1
        # 如果调整值为-1
        elif adjustment == -1:
            # 计算新的剂量，不低于最小剂量
            new_dose = max(self.current_dose - step, self.config['min_dose'])
            # 如果新的剂量与当前剂量不同
            if new_dose != self.current_dose:
                # 记录最后一次调整为负调整
                self.last_adjustment = -1
        # 如果调整值不是1或-1
        else:
            # 新的剂量保持为当前剂量
            new_dose = self.current_dose
            # 记录最后一次调整为零调整
            self.last_adjustment = 0
        # 更新当前剂量
        self.current_dose = new_dose

controller = PhosphorusSmartController(PHOSPHORUS_CONTROLLER_CONFIG)