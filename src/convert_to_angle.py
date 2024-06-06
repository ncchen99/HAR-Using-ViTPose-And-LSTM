import pandas as pd
import numpy as np

def calculate_angle(point1, point2, point3):
    # 计算向量
    vector1 = point1 - point2
    vector2 = point3 - point2

    # 计算向量的长度
    vector1_norm = np.linalg.norm(vector1)
    vector2_norm = np.linalg.norm(vector2)

    # 计算单位向量
    vector1_unit = vector1 / vector1_norm
    vector2_unit = vector2 / vector2_norm

    # 计算夹角（使用向量的点积）
    dot_product = np.dot(vector1_unit, vector2_unit)
    angle = np.arccos(dot_product)

    # 将弧度转换为角度
    angle = np.degrees(angle)

    return angle

def calculate_angle_with_horizontal(point1, point2):
    # 水平线的向量
    horizontal_line = np.array([1, 0])

    # 计算关键点与水平线的向量
    vector = point2 - point1

    # 计算关键点与水平线的夹角（使用向量的点积）
    dot_product = np.dot(vector, horizontal_line)
    angle = np.arccos(dot_product / (np.linalg.norm(vector) * np.linalg.norm(horizontal_line)))

    # 将弧度转换为角度
    angle = np.degrees(angle)

    return angle

def make_angle(input_angle):

    angle_data = []

    # 逐帧计算角度
    for  row in input_angle:
        # 获取关键点坐标和置信度分数
        left_ankle = np.array([row[30], row[31]])
        left_knee = np.array([row[26], row[27]])
        left_hip = np.array([row[22], row[23]])
        left_shoulder = np.array([row[10], row[11]])
        left_eye = np.array([row[2], row[3]])
        left_elbow = np.array([row[14], row[15]])
        left_wrist = np.array([row[18], row[19]])

        right_ankle = np.array([row[32], row[33]])
        right_knee = np.array([row[28], row[29]])
        right_hip = np.array([row[24], row[25]])
        right_shoulder = np.array([row[12], row[13]])
        right_eye = np.array([row[4], row[5]])
        right_elbow = np.array([row[16], row[17]])
        right_wrist = np.array([row[20], row[21]])
        # 计算角度
        angle_left_ankle_knee_hip = calculate_angle(left_ankle, left_knee, left_hip)
        angle_left_knee_hip_shoulder = calculate_angle(left_knee, left_hip, left_shoulder)
        angle_left_hip_shoulder_eye = calculate_angle(left_hip, left_shoulder, left_eye)
        angle_left_shoulder_eye_elbow = calculate_angle(left_shoulder, left_eye, left_elbow)
        angle_left_shoulder_elbow_wrist = calculate_angle(left_shoulder, left_elbow, left_wrist)
        angle_left_hip_shoulder_elbow = calculate_angle(left_hip, left_shoulder, left_elbow)
        
        angle_left_ankle_knee = calculate_angle_with_horizontal(left_ankle, left_knee)
        angle_left_knee_hip = calculate_angle_with_horizontal(left_knee, left_hip)
        angle_left_hip_shoulder = calculate_angle_with_horizontal(left_hip, left_shoulder)
        angle_left_shoulder_eye = calculate_angle_with_horizontal(left_shoulder, left_eye)
        angle_left_shoulder_elbow = calculate_angle_with_horizontal(left_shoulder, left_elbow)
        angle_left_elbow_wrist = calculate_angle_with_horizontal(left_elbow, left_wrist)
        
        

        angle_right_ankle_knee_hip = calculate_angle(right_ankle, right_knee, right_hip)
        angle_right_knee_hip_shoulder = calculate_angle(right_knee, right_hip, right_shoulder)
        angle_right_hip_shoulder_eye = calculate_angle(right_hip, right_shoulder, right_eye)
        angle_right_shoulder_eye_elbow = calculate_angle(right_shoulder, right_eye, right_elbow)
        angle_right_shoulder_elbow_wrist = calculate_angle(right_shoulder, right_elbow, right_wrist)
        angle_right_hip_shoulder_elbow = calculate_angle(right_hip, right_shoulder, right_elbow)
        
        angle_right_ankle_knee = calculate_angle_with_horizontal(right_ankle, right_knee)
        angle_right_knee_hip = calculate_angle_with_horizontal(right_knee, right_hip)
        angle_right_hip_shoulder = calculate_angle_with_horizontal(right_hip, right_shoulder)
        angle_right_shoulder_eye = calculate_angle_with_horizontal(right_shoulder, right_eye)
        angle_right_shoulder_elbow = calculate_angle_with_horizontal(right_shoulder, right_elbow)
        angle_right_elbow_wrist = calculate_angle_with_horizontal(right_elbow, right_wrist)
        
        
        
        angle_data.append([
            angle_left_ankle_knee_hip, angle_left_knee_hip_shoulder, angle_left_hip_shoulder_eye, 
            angle_left_shoulder_eye_elbow, angle_left_shoulder_elbow_wrist, angle_left_hip_shoulder_elbow,                    
            angle_right_ankle_knee_hip, angle_right_knee_hip_shoulder, angle_right_hip_shoulder_eye, 
            angle_right_shoulder_eye_elbow, angle_right_shoulder_elbow_wrist, angle_right_hip_shoulder_elbow,
            angle_left_ankle_knee, angle_left_knee_hip, angle_left_hip_shoulder, 
            angle_left_shoulder_eye, angle_left_shoulder_elbow, angle_left_elbow_wrist, 
            angle_right_ankle_knee, angle_right_knee_hip, angle_right_hip_shoulder, 
            angle_right_shoulder_eye, angle_right_shoulder_elbow, angle_right_elbow_wrist
            ])
    return angle_data
        

    
    

