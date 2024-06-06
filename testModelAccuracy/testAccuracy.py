import pandas as pd
import os
import algo
import colab_find_bounce_up
import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import warnings
from src.normalize import normalize_pose_landmarks

# 定義兩個資料夾的路徑
folders = ["./0-", "./0+"]

# 遍歷每個資料夾
for folder in folders:
    # 獲取資料夾內所有檔案
    for filename in os.listdir(folder):
        if filename.endswith('.csv'):  # 確認是CSV檔案
            file_path = os.path.join(folder, filename)  # 獲取檔案完整路徑
            df = pd.read_csv(file_path)  # 讀取CSV檔案
            result = df.values.tolist()
            left_ankle=df["49"].tolist()
            if(len(left_ankle)>=60):
                filled_result=colab_find_bounce_up.fill_ankle(left_ankle)
                jumpstart,jumpend=algo.find_inflection(filled_result)
                if(jumpend>60 and jumpstart>=15):
                    result=result[jumpstart-15:jumpend+1][:]
                else :
                    result=result[:jumpend+1][:]
        model_input = normalize_pose_landmarks(np.array(result, dtype=np.float32))
    # 2. convert to numpy float array
        model_input = model_input.astype(np.float32)
        # 3. convert input to tensor
        model_input = torch.Tensor(model_input)
        # # 4. add extra dimension
        model_input = torch.unsqueeze(model_input, dim=0)
                
            
            
