
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt






def find_inflection(ankle_list) :
    # 计算斜率
    slope = np.gradient(ankle_list)

    # 寻找反折点
    inflection_points = []
    for i in range(1, len(slope)):
        if slope[i] * slope[i - 1] < 0:
            inflection_points.append(i)

    # 连接反折点并找到最大长度的线段
    max_length = 0
    start_index = None
    end_index = None

    
    print("total=",len(ankle_list))
    
    if len(inflection_points) >= 2:
        for i in range(1, len(inflection_points)):
            length = inflection_points[i] - inflection_points[i - 1]
            if length > max_length:
                max_length = length
                start_index = inflection_points[i - 1]
                middle_index = inflection_points[i]
                end_index = inflection_points[i+1]

    # 绘制图形（可选）
    
    print(f'Start index: {start_index}, Mid index: {middle_index},end index: {end_index}, Length: {max_length}')
    return end_index



