TARGET_OUT_TP = 0.275 #目标出水总磷
ERROR_MARGIN = 0.025  #目标出水总磷误差范围
MAX_PAC = 500 #上线PAC流量mg/L
MIN_PAC = 300 #下限PAC流量mg/L

SCALER_PATH_TP = "models/scaler_tp.joblib"
BATCH_SIZE_TR = 16          # 批次大小
HIDDEN_LAYERS = 16          # LSTM隐藏层维度
OUTPUT_FEATURES = 6         # 输出步长
TRANSFORMER_LAYERS = 1      # Transformer层数
LEARNING_RATE_TR = 0.001    # 初始学习率
TRANSFORMER_D_MODEL = 16
TRANSFORMER_NHEAD = 2
EPOCHS_TR = 50
SUQUENCE_LENGTH = 14        # 修正拼写错误
NORMAL_PAC_RATIO = 0.0390625  #正常pac浓度

PHOSPHORUS_CONTROLLER_CONFIG = {
    'target': 0.275,            # 目标值
    'safe_min': 0.170,          # 安全最小值
    'safe_max': 0.325,          # 安全最大值
    'emergency_th': 0.35,       # 紧急阈值
    'min_dose': 300,            # 最小投加量
    'max_dose': 490,            # 最大投加量
    'init_dose': 350,           # 初始投加量
    'history_size': 8,          # 历史数据大小
    'trend_window': 6,          # 趋势窗口大小
    'base_step': 15             # 基础步长
}