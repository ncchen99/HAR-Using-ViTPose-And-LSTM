import csv
import pandas as pd
import matplotlib.pyplot as plt
from scipy.interpolate import interp1d
from scipy.signal import butter, filtfilt



def fill_ankle( y_cord_list):
  x = list(range(1, len(y_cord_list) + 1))  # 定义 x 轴数据为帧数，从 1 开始递增到列表长度
  # 通過低通濾波器前的準備 : 用插植法填補空值
  for i, y_cord in enumerate(y_cord_list) :
    # 若有空值，讓空的這格跟前面一樣
    if y_cord == 100 :
      if i > 0:
        y_cord_list[i] = y_cord_list[i-1]
      else :
        y_cord_list[i] = 0
  # 低通濾波器前的參數
  cutoff_frequency = 1  # 截止頻率
  sampling_frequency = 10  # 采樣頻率
  # 通過低通濾波器處理資料
  filtered_y_list = low_pass_filter(y_cord_list, cutoff_frequency, sampling_frequency)
  #print("filtered_list:     ",filtered_y_list)
  return filtered_y_list

# 低通濾波器
def low_pass_filter(data, cutoff_freq, sampling_freq, order=5):
    nyquist_freq = 0.5 * sampling_freq
    normalized_cutoff_freq = cutoff_freq / nyquist_freq
    b, a = butter(order, normalized_cutoff_freq, btype='low')
    filtered_data = filtfilt(b, a, data)
    return filtered_data





