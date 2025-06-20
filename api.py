import time 
from collections import deque 
from influxdb_client import InfluxDBClient
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
from fastapi import FastAPI, HTTPException
from pyparsing import deque
from predict import FEATURES, predict, load_model, InputData, transformlstm_predict_water_tp, controller
from pydantic import BaseModel, Field
from datetime import datetime
import logging
from config import *
from typing import List
from typing import Optional
from config import PHOSPHORUS_CONTROLLER_CONFIG

app = FastAPI()
#----------------日志模块-------------------
# 获取在 schedule_tasks.py 中配置的日志记录器
logger = logging.getLogger("api")
# 禁用传播机制，防止日志被传播到根记录器
logger.propagate = False

def setup_logging():
    """设置日志记录器"""
    logger = logging.getLogger('daily_logger')
    logger.setLevel(logging.DEBUG)
    
    # 获取当天日期格式化为 YYYY-MM-DD
    today = datetime.now().strftime('%Y-%m-%d')
    log_filename = f"logger/{today}.log"
    
    # 如果logger已经有处理器，先清除所有处理器
    if logger.handlers:
        logger.handlers.clear()
    
    # 创建文件处理器，写入到当天的日志文件
    file_handler = logging.FileHandler(log_filename)
    file_handler.setLevel(logging.DEBUG)
    
    # 设置日志格式
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    
    # 添加处理器到日志记录器
    logger.addHandler(file_handler)
    return logger


#----------------随机森林-------------------


class InputDataNotime(BaseModel):
    PAC1: Optional[float] = Field(None, alias="PAC1", description="聚合氯化铝1投加量（mg/L）")
    PAC2: Optional[float] = Field(None, alias="PAC2", description="聚合氯化铝2投加量（mg/L）")
    PAM1: Optional[float] = Field(None, alias="PAM1", description="聚丙烯酰胺1投加量（mg/L）")
    PAM2: Optional[float] = Field(None, alias="PAM2", description="聚丙烯酰胺2投加量（mg/L）")
    TC1_FILTER_EFFTP: Optional[float] = Field(None, alias="FilterTp", description="TC1过滤器后出水总磷（mg/L）")
    TC1_SDS_EFFTP: Optional[float] = Field(None, alias="SdsTp", description="TC1沉淀池后出水总磷（mg/L）")
    #effluent_tp: Optional[float] = Field(None, alias="EffTp", description="出水总磷浓度（mg/L）")
    influent_Ammonia: Optional[float] = Field(None, alias="InfluentAmmonia", description="进水氨氮浓度（mg/L）")
    influent_COD: Optional[float] = Field(None, alias="InfluentCod", description="进水化学需氧量（mg/L）")
    influent_PH: Optional[float] = Field(None, alias="InfluentpH", description="进水pH值", ge=0, le=14)
    OxidationDitch_MLSS: Optional[float] = Field(None, alias="OxidationDitchMLSS", description="氧化沟混合液悬浮固体浓度（mg/L）")

# 加载所有模型
model_tn = load_model("Effluent_TN")
model_tp = load_model("Effluent_TP")
model_ammonia = load_model("Effluent_Ammonia")
model_cod = load_model("Effluent_COD")

@app.post("/predAll")
def predict_all(input_data: InputDataNotime):
    """进行四个目标变量的预测"""
    # 将输入数据转换为字典格式
    input_dict = input_data.model_dump()
    # 将字典格式的数据转换为DataFrame格式
    input_df = pd.DataFrame([input_dict])
    
    targets = ["Effluent_COD", "Effluent_Ammonia", "Effluent_TN", "Effluent_TP"]
    result = {}
    
    for target in targets:
        try:
            # 加载目标对应的模型
            model = load_model(target)
            # 获取除目标变量外的特征列表
            features_to_use = [col for col in FEATURES if col != target]
            # 从输入数据中提取出目标变量对应的特征
            input_df_target = input_df[features_to_use]  # 确保特征顺序一致
            # 使用模型进行预测
            prediction = model.predict(input_df_target)[0]
            # 将预测结果添加到结果字典中，并保留四位小数
            result.update({
                f"predict{target}": round(prediction, 4),
            })
        except Exception as e:
            # 如果某个目标预测失败，记录错误信息，继续其他预测
            # 将错误信息添加到结果字典中
            result.update({
                f"predict{target}": None,
                f"error_{target}": str(e)
            })
    
    return {
        "code": 0,
        "data": [result]
    }

@app.post("/predWyTp")
async def predict_effluent_tp(input_data: InputDataNotime):
    return predict(model_tp, input_data, "Effluent_TP")

@app.post("/predWyTn")
async def predict_effluent_tn(input_data: InputDataNotime):
    return predict(model_tn, input_data, "Effluent_TN")

@app.post("/predWyAmmonia")
async def predict_effluent_ammonia(input_data: InputDataNotime):
    return predict(model_ammonia, input_data, "Effluent_Ammonia")

@app.post("/predWyCod")
async def predict_effluent_cod(input_data: InputDataNotime):
    return predict(model_cod, input_data, "Effluent_COD")

#----------------transformer+lstm-------------------

INFLUX_CONFIG = {
    "token": "2p7Z7zczQwZkOS0cxXR6XcF63O8YQyMXI5QgyTery3u9NXaqOgWGJaD7cVyqOyxW5pB7OyIzWlrGgsTDMkfFWg==",
    "org": "wy",
    "bucket": "stp",
    "url": "https://wy-influxdb.tunnel.blueorigintech.com/" 
}


class InputData(BaseModel):
    time: str = Field(..., alias="time", description="时间戳（格式建议: YYYY-MM-DD HH:MM:SS）")
    pac1: Optional[float] = Field(None, alias="PAC1", description="聚合氯化铝1投加量（mg/L）")
    pac2: Optional[float] = Field(None, alias="PAC2", description="聚合氯化铝2投加量（mg/L）")
    pam1: Optional[float] = Field(None, alias="PAM1", description="聚丙烯酰胺1投加量（mg/L）")
    pam2: Optional[float] = Field(None, alias="PAM2", description="聚丙烯酰胺2投加量（mg/L）")
    tc1_filter_efftp: Optional[float] = Field(None, alias="FilterTp", description="TC1过滤器后出水总磷（mg/L）")
    tc1_sds_efftp: Optional[float] = Field(None, alias="SdsTp", description="TC1沉淀池后出水总磷（mg/L）")
    effluent_tp: Optional[float] = Field(None, alias="EffluentTp", description="出水总磷浓度（mg/L）")
    influent_ammonia: Optional[float] = Field(None, alias="InfluentAmmonia", description="进水氨氮浓度（mg/L）")
    influent_cod: Optional[float] = Field(None, alias="InfluentCod", description="进水化学需氧量（mg/L）")
    influent_ph: Optional[float] = Field(None, alias="InfluentpH", description="进水pH值", ge=0, le=14)
    oxidation_ditch_mlss: Optional[float] = Field(None, alias="OxidationDitchMLSS", description="氧化沟混合液悬浮固体浓度（mg/L）")
    PacRatio: Optional[float] = Field(None, alias="PacRatio", description="聚合氯化铝浓度")


def get_realtime_data(tag_name="MW836", lookback_minutes=5):
    """安全获取实时数据（默认回溯5分钟）"""
    try:
        # 动态生成时间范围（兼容未来时间）
        end_time = datetime.utcnow() 
        start_time = (end_time - timedelta(minutes=lookback_minutes)).isoformat() + "Z"
        
        # 优化后的Flux查询
        query = f'''
        from(bucket: "{INFLUX_CONFIG['bucket']}")
          |> range(start: {start_time})
          |> filter(fn: (r) => r._measurement == "mw" and r._field == "val" and r.pos  == "{tag_name}")
          |> last()
          |> limit(n:1)
        '''
        
        with InfluxDBClient(**INFLUX_CONFIG) as client:
            result = client.query_api().query(query) 
            
            # 防御性空值判断 
            if not result or len(result) == 0:
                return {"status": "error", "message": f"无 {tag_name} 的实时数据"}
            
            table = result[0]
            if len(table.records)  == 0:
                return {"status": "error", "message": "查询结果为空记录集"}
            
            record = table.records[0] 
            beijing_time = pd.to_datetime(record.get_time()).tz_convert('Asia/Shanghai') 
            
            return {
                "tag": tag_name,
                "value": record.get_value(), 
                "timestamp": beijing_time.strftime('%Y-%m-%d  %H:%M:%S'),
                "status": "success"
            }
            
    except Exception as e:
        return {"status": "error", "message": f"系统级错误: {str(e)}"}

@app.post("/predPacDosing")
async def predict_dosing(data: List[InputData]):
     # 初始化日志记录器
    logger = setup_logging()
    # 记录所有输入数据到日志
    input_data_list = [{
        "time": item.time,
        "PAC1": item.pac1,
        "PAC2": item.pac2,
        "PAM1": item.pam1,
        "PAM2": item.pam2,
        "FilterTp": item.tc1_filter_efftp,
        "SdsTp": item.tc1_sds_efftp,
        "EffluentTp": item.effluent_tp,
        "InfluentAmmonia": item.influent_ammonia,
        "InfluentCod": item.influent_cod,
        "InfluentpH": item.influent_ph,
        "OxidationDitchMLSS": item.oxidation_ditch_mlss,
        "PacRatio": item.PacRatio
    } for item in data]
    logger.info("Received data for /predPacDosing: %s", input_data_list)
    custom_data = np.array([
        [
            d.pac1,  # 聚合氯化铝1投加量（mg/L）
            d.pac2,  # 聚合氯化铝2投加量（mg/L）
            d.pam1,  # 聚丙烯酰胺1投加量（mg/L）
            d.pam2,  # 聚丙烯酰胺2投加量（mg/L）
            d.tc1_filter_efftp,  # TC1过滤器后出水总磷（mg/L）
            d.tc1_sds_efftp,  # TC1沉淀池后出水总磷（mg/L）
            d.effluent_tp,  # 出水总磷浓度（mg/L）
            d.influent_ammonia,#进水氨氮浓度（mg/L）
            d.influent_cod,  #  进水化学需氧量 (mg/L)
            d.influent_ph,  # 进水pH值", ge=0, le
            d.oxidation_ditch_mlss # 氧化沟混合液悬浮固体浓度（mg/L）
        ]
        for d in data
    ])
    
    # 浓度判断
    print("DATA", data[SUQUENCE_LENGTH-1].PacRatio)
    
    pacRatio = abs(2 - ( data[SUQUENCE_LENGTH-1].PacRatio/NORMAL_PAC_RATIO ))
   
    print("pacRatio",pacRatio)
    # 获取实时数据 
    # 正式环境
    tpdata = get_realtime_data(lookback_minutes=30)  # 扩展时间窗口 
    test_values = [tpdata['value']]
    # 测试环境
    # tpdata = data[13].effluent_tp
    # print("tpdata",tpdata)
    # test_values = [tpdata]

    
    prediction = transformlstm_predict_water_tp(custom_data)
    print("prediction",prediction)
    test_predictions = [
        prediction
    ]
    
    # 运行
    result = controller.update_control(test_values[0],  test_predictions[0])
        
    print(f"总磷: {test_values[0]:.3f} | 状态: {result['status']} | "
                  f"趋势: {result['trend']} | 加药量: {result['current_dose']}mg/L")

    dosing_strategy = {
        "predictList": [round(num, 4) for num in prediction],
        "targetOutTp": PHOSPHORUS_CONTROLLER_CONFIG['target'],
        "targetOutRange": 0.025,
        "adj3Pac": 0 ,
        "adj4Pac": round(result['current_dose']* pacRatio,2),
        "maxPac": PHOSPHORUS_CONTROLLER_CONFIG['max_dose'],
        "minPac": PHOSPHORUS_CONTROLLER_CONFIG['min_dose'],
        "version": "Beta 1.4.3"
    }
    # 记录输出数据到日志，只记录一次
    logger.info("Output from /predPacDosing: %s", dosing_strategy)
    return {
        "code": 0,
        "data": [dosing_strategy]
    }