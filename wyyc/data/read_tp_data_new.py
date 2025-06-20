from influxdb_client import InfluxDBClient
import pandas as pd

# InfluxDB 配置
token = "2p7Z7zczQwZkOS0cxXR6XcF63O8YQyMXI5QgyTery3u9NXaqOgWGJaD7cVyqOyxW5pB7OyIzWlrGgsTDMkfFWg=="
org = "wy"
bucket = "stp"
url = "https://wy-influxdb.tunnel.blueorigintech.com/"

# 查询时间范围
start_time = '2025-05-13T00:00:00Z'
stop_time = '2025-06-13T16:30:00Z'

# 要查询的点位及其别名
fields = {
    "MW2123": "MW2123",
    "MW2124": "MW2124",
    "MW2125": "MW2125",
    "MW2126": "MW2126",
    "MW836": "MW836",
    "MW832": "MW832",
    "MW837": "MW837",
    "MW829": "MW829",
    "MW664": "MW664",
    "TC1_FILTER_EFFTP": "MW2130",
    "TC1_SDS_EFFTP": "MW2129",
    "EM231124040001":"EM231124040001",
    "EM231124040002":"EM231124040002",
    "EM240614150001":"EM240614150001",
    "EM240614150002":"EM240614150002"
}

# 构造查询语句
query = f"""
from(bucket: "{bucket}")
  |> range(start: {start_time}, stop: {stop_time})
  |> filter(fn: (r) => r._measurement == "mw")
  |> filter(fn: (r) => r._field == "val")
  |> filter(fn: (r) => r.pos == "MW2123" or r.pos == "MW2124" or r.pos == "MW2125" or r.pos == "MW2126" 
  or r.pos == "MW836" or r.pos == "MW2130" or r.pos == "MW2129"   or r.pos == "MW832" or r.pos == "MW837" 
  or r.pos == "MW829" or r.pos == "MW664" or r.pos == "EM231124040001" or r.pos == "EM231124040002" 
  or r.pos == "EM240614150001" or r.pos == "EM240614150002")
  |> window(every: 30m)
  |> max()
  |> pivot(rowKey: ["_start"], columnKey: ["pos"], valueColumn: "_value")
  |> keep(columns: ["_start", "MW2123", "MW2124", "MW2125", "MW2126", "MW836","MW2130", "MW2129",  
  "MW832", "MW837", "MW829", "MW664", "EM231124040001", "EM231124040002", "EM240614150001", "EM240614150002"])
"""

# 创建 InfluxDB 客户端
with InfluxDBClient(url=url, token=token, org=org) as client:
    try:
        # 执行查询
        result = client.query_api().query(query=query)
        
        # 检查查询结果是否为空
        if not result:
            print("未查询到数据")
        
        # 将查询结果转换为 DataFrame
        data = []
        for table in result:
            for record in table.records:
                # 获取原始时间并转换为北京时间
                original_time = record.values.get("_start")
                beijing_time = pd.to_datetime(original_time) + pd.Timedelta(hours=8)
                
                data.append({
                    "_time": beijing_time,
                    "PAC1": record.values.get("MW2123", 0),
                    "PAC2": record.values.get("MW2125", 0),
                    "PAM1": record.values.get("MW2124", 0),
                    "PAM2": record.values.get("MW2126", 0),
                    "TC1_FILTER_EFFTP": record.values.get("MW2130", 0), #进水总磷 暂时代替MW2130
                    "TC1_SDS_EFFTP": record.values.get("MW2129", 0),
                    "Effluent_TP": record.values.get("MW836", 0),
                    "influent_Ammonia":  record.values.get("MW832", 0),
                    "influent_COD": record.values.get("MW664", 0),
                    "influent_PH": record.values.get("MW837", 0),
                    "OxidationDitch_MLSS": record.values.get("MW829", 0),
                    "2#SurfaceAerator": record.values.get("EM231124040001", 0),
                    "1#SurfaceAerator": record.values.get("EM231124040002", 0),
                    "HighTempEle": record.values.get("EM240614150001", 0),
                    "NaClOEle": record.values.get("EM240614150002", 0)
                })
        
        # 创建 DataFrame
        df = pd.DataFrame(data)
        
        # 将空值替换为零
        df = df.fillna(0)

        # 指定需要处理的列名
        columns_to_process = ['2#SurfaceAerator',
                              '1#SurfaceAerator',
                              'HighTempEle',
                              'NaClOEle']

        # 检查列名是否存在于 DataFrame 中
        columns_to_process = [col for col in columns_to_process if col in df.columns]

        if not columns_to_process:
            raise ValueError("指定的列名在 DataFrame 中不存在")

        # 选择需要处理的列
        selected_cols = df[columns_to_process]

        # 计算每一行与上一行的差值
        diff_result = selected_cols.diff()

        # 将差值覆盖到原始列
        df[columns_to_process] = diff_result

        # 丢弃第一行
        df = df.drop(0)

        # 查看结果
        print(df)

        # 保存为 CSV 文件
        df.to_csv('data/all_data_water_tp_new.csv', index=False)
        print("数据已保存为 all_data_water_tp_new.csv")
        
        df = pd.read_csv('data/all_data_water_tp_new.csv')
        cut_time = '2025-05-25 19:30:00+00:00'
        mask = pd.to_datetime(df['_time']) < pd.to_datetime(cut_time)
        # 互换PAM1和PAC2
        pam1_old = df.loc[mask, 'PAM1'].copy()
        df.loc[mask, 'PAM1'] = df.loc[mask, 'PAC2']
        df.loc[mask, 'PAC2'] = pam1_old
        df.to_csv('data/all_data_water_tp_new.csv', index=False)
    except Exception as e:
        print(f"查询失败: {e}")