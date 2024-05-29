import os
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from lib.vitpose_preprocessor import VitPosePreprocessor

#@title Functions to run pose estimation with Vitpose

VIDEOS_ROOT = "data"
videos_in_train_folder = os.path.join(VIDEOS_ROOT, 'train')
videos_out_train_folder = 'poses_videos_out_train'
csvs_out_train_folder = 'poses_videos_out_train'

preprocessor = VitPosePreprocessor(
    videos_in_folder=videos_in_train_folder,
    videos_out_folder=videos_out_train_folder,
    csvs_out_path=csvs_out_train_folder,
)

preprocessor.process(per_pose_class_limit=None)


videos_in_test_folder = os.path.join(VIDEOS_ROOT, 'test')
videos_out_test_folder = 'poses_videos_out_test'
csvs_out_test_folder = 'poses_videos_out_test'

preprocessor = VitPosePreprocessor(
    videos_in_folder=videos_in_test_folder,
    videos_out_folder=videos_out_test_folder,
    csvs_out_path=csvs_out_test_folder,
)

preprocessor.process(per_pose_class_limit=None)