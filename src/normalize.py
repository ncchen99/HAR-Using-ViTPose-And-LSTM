import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .data import BodyPart

def get_center_point(landmarks, left_bodypart, right_bodypart):
    left = landmarks[:, left_bodypart.value, :]
    right = landmarks[:, right_bodypart.value, :]
    center = (left + right) * 0.5
    return center

def get_pose_size(landmarks, torso_size_multiplier=2.5):
    hips_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    shoulders_center = get_center_point(landmarks, BodyPart.LEFT_SHOULDER, BodyPart.RIGHT_SHOULDER)
    torso_size = np.linalg.norm(shoulders_center - hips_center, axis=1)

    pose_center_new = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center_new = np.expand_dims(pose_center_new, axis=1)
    pose_center_new = np.broadcast_to(pose_center_new, landmarks.shape)

    d = landmarks - pose_center_new
    d = d[0]  # Assuming taking the first index of the batch
    max_dist = np.max(np.linalg.norm(d, axis=1))

    pose_size = np.maximum(torso_size * torso_size_multiplier, max_dist)
    return pose_size


def normalize_pose_landmarks(landmarks):
    landmarks = np.reshape(np.nan_to_num(landmarks, copy=False, nan=0, posinf=None, neginf=None), (-1, 17, 2))
    pose_center = get_center_point(landmarks, BodyPart.LEFT_HIP, BodyPart.RIGHT_HIP)
    pose_center = np.expand_dims(pose_center, axis=1)
    pose_center = np.broadcast_to(pose_center, landmarks.shape)
    landmarks = landmarks - pose_center

    pose_size = get_pose_size(landmarks)
    landmarks /= pose_size[:, np.newaxis, np.newaxis]

    return landmarks.reshape(-1, 34)
