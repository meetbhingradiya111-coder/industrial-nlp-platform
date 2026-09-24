import os
import subprocess
import time

# 1. Start FastAPI backend (main.py) on port 8000
backend_process = subprocess.Popen(
    ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
)

# Allow FastAPI backend 5 seconds to load PyTorch & models
time.sleep(5)

# 2. Launch Streamlit UI (app.py) on Hugging Face default port 7860
os.system(
    "streamlit run app.py --server.port=7860 --server.address=0.0.0.0"
)