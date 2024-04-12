import pandas as pd
import numpy as np
from normalize import normalize_pose_landmarks


data_path = "/home/mcnlab/桌面/HAR using LSTM/swenbao_jump_data/train_data.csv"
info_path = "/home/mcnlab/桌面/HAR using LSTM/swenbao_jump_data/train_info.csv"

data = pd.read_csv(data_path, sep=',')
info = pd.read_csv(info_path, sep=',', header=None)

y =  []
# calculate the number of action classes and find the largest block
block_sizes = []
action_classes_num = 0
for row in info.iterrows():
    if pd.isna(row[1][3]):
        y += [row[1][1]] * row[1][2]
        action_classes_num += 1
        continue
    block_sizes.append(int(row[1][3]))
    
max_block = max(block_sizes)

x = data.values

x = np.delete(x, slice(2, None, 3), 1)
# 2. convert all NAN to -10
x = np.nan_to_num(x, copy=False, nan=0, posinf=None, neginf=None)
# 3. fill the blocks with padding
final_x = np.array([]).reshape(0, x.shape[1])
i_iter = 0
for block_size in block_sizes:
    block = np.full((max_block, x.shape[1]), -10)
    for i in range(block_size):
        block[i] = x[i_iter + i]
    i_iter += block_size
    final_x = np.concatenate((final_x, block), axis=0)

# for row in final_x[0:max_block+2]:
#     print(row)
    
pd.DataFrame(x).to_csv("origin.csv", index=False)
x = normalize_pose_landmarks(x)

x = np.nan_to_num(x, copy=False, nan=-10, posinf=None, neginf=None)
pd.DataFrame(x).to_csv("normalized.csv", index=False)
print(x[0:max_block])


a = np.reshape(final_x, (-1, 17, 2))
# print(a.reshape(-1, 34)[3:max_block+2])
# print(final_x)