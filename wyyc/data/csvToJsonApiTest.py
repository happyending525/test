import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import csv 
import json 
import time 
from datetime import datetime 
import requests

from config import SUQUENCE_LENGTH 
 
def process_csv_to_api(csv_path, batch_size=14, interval=2):
    """
    从CSV文件分批次读取数据并发送HTTP请求 
    
    参数：
    - csv_path: CSV文件路径 
    - batch_size: 每批数据条数（默认14）
    - interval: 请求间隔秒数（默认2）
    """
    # 读取CSV文件（自动跳过标题行）
    with open(csv_path, 'r', encoding='utf-8') as f:
        csv_reader = csv.DictReader(f)
        data = [row for row in csv_reader]  # data[0]对应CSV第2行 
    
    # 字段映射关系 
    field_mapping = {
        "_time": "time",
        "influent_Ammonia": "InfluentAmmonia",
        "influent_COD": "InfluentCod",
        "influent_PH": "InfluentpH",
        "Effluent_TP": "EffluentTp",
        "PAC1": "PAC1",
        "PAC2": "PAC2",
        "PAM1": "PAM1",
        "PAM2": "PAM2",
        "OxidationDitch_MLSS": "OxidationDitchMLSS", 
        "TC1_FILTER_EFFTP": "FilterTp",
        "TC1_SDS_EFFTP": "SdsTp"
    }
 
    start_index = 0  # 初始批次索引（对应CSV第2行）
    total_batches = 0 
    
    while start_index + batch_size <= len(data):
        # 1. 提取当前批次数据 
        batch_data = data[start_index : start_index + batch_size]
        
        # 2. 转换JSON结构 
        payload = []
        for row in batch_data:
            json_obj = {}
            for csv_field, json_field in field_mapping.items(): 
                value = row[csv_field]
                
                # 特殊处理时间字段 
                if csv_field == "_time":
                    try:
                        dt = datetime.fromisoformat(value) 
                        value = dt.strftime("%Y-%m-%d  %H:%M:%S")
                    except ValueError:
                        value = "2025-05-15 10:41:00"  # 默认当前时间 
                
                # 数值类型转换 
                try:
                    if csv_field != "_time":  # 时间字段不转换类型 
                        value = float(value) if '.' in value else int(value)
                except (ValueError, TypeError):
                    pass  # 保持字符串类型 
                
                json_obj[json_field] = value
            json_obj['PacRatio'] = 0.0390625 #增加浓度字段
            payload.append(json_obj) 
        print(payload)
        # 3. 发送HTTP请求 
        url = "http://127.0.0.1:8090/predPacDosing"
        headers = {"Content-Type": "application/json"}
        #url = "https://wy-stp.tunnel.blueorigintech.com/drug-api/predPacDosing"
        try:
            response = requests.post(url,  json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            print(response.json())
            print(f"[{datetime.now().strftime('%H:%M:%S')}]  批次 {start_index+1}-{start_index+batch_size} 请求成功 | 状态码: {response.status_code}") 
        except requests.exceptions.RequestException  as e:
            print(f"[{datetime.now().strftime('%H:%M:%S')}]  批次 {start_index+1}-{start_index+batch_size} 请求失败: {str(e)}")
        
        # 4. 更新索引与间隔等待 
        start_index += 1 
        total_batches += 1 
        time.sleep(interval) 
 
    print(f"处理完成，共发送 {total_batches} 个批次")
 
if __name__ == "__main__":
    process_csv_to_api("data/csvToJson.csv",SUQUENCE_LENGTH) 