# main.py
import uvicorn
from api import app, setup_logging

if __name__ == "__main__":
    # 训练所有模型（如果需要）
    # 启动 FastAPI 服务
    print("Starting the FastAPI server...")
    print("To make a prediction, send a POST request to the corresponding endpoint with the input data.")
    uvicorn.run(app, host="127.0.0.1", port=8090)
