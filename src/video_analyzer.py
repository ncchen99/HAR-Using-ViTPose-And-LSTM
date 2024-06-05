import os
import time
import cv2
import ntpath


from lib.config import Config
from lib.tools import verify_video, convert_video, get_video_info
import src.colab_find_bounce_up as colab_find_bounce_up
import src.algo as algo

import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import warnings

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

from src.normalize import normalize_pose_landmarks

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

assert has_mmdet, 'Please install mmdet to run the demo.'

config = Config()
det_model = init_detector(
    config.det_config, config.det_checkpoint, device=config.device.lower())
# build the pose model from a config file and a checkpoint file
pose_model = init_pose_model(
    config.pose_config, config.pose_checkpoint, device=config.device.lower())

# how many frames to skip while inferencing
# configuring a higher value will result in better FPS (frames per rate), but accuracy might get impacted
SKIP_FRAME_COUNT = 0

# analyse the video
def analyse_video(lstm_classifiers, video_path, class_names):
    
    #print("Label detected ", label)
    file_name = ntpath.basename(video_path)
    config = Config(video_path, 'res_{}'.format(file_name))
    print(f"Starting video processing: {video_path} saved as \"res_{file_name}\"")


    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        warnings.warn(
            'Please set `dataset_info` in the config.'
            'Check https://github.com/open-mmlab/mmpose/pull/663 for details.',
            DeprecationWarning)
    else:
        dataset_info = DatasetInfo(dataset_info)

    ###############################
        
    cap = cv2.VideoCapture(config.video_path)
    # assert cap.isOpened(), f'Faild to load video file {config.video_path}'

    if config.out_video_path == '':
        save_out_video = False
    else:
        # os.makedirs(os.path.split(config.out_video_path)[0], exist_ok=True)
        save_out_video = True
    
    fps = None
    if save_out_video:
        fps = cap.get(cv2.CAP_PROP_FPS)
        size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        videoWriter = cv2.VideoWriter(config.out_video_path, fourcc, fps, size)

    # optional
    return_heatmap = False

    # e.g. use ('backbone', ) to return backbone feature
    output_layer_names = None

    frame_index = 0
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    result = []
    
    start = time.time()
    
    while (cap.isOpened()):
        flag, img = cap.read()
        if not flag:
            break
        # test a single image, the resulting box is (x1, y1, x2, y2)
        mmdet_results = inference_detector(det_model, img)

        # keep the person class bounding boxes.
        person_results = process_mmdet_results(mmdet_results, config.det_cat_id)

        # test a single image, with a list of bboxes.
        pose_results, returned_outputs = inference_top_down_pose_model(
            pose_model,
            img,
            person_results,
            bbox_thr=config.bbox_thr,
            format='xyxy',
            dataset=dataset,
            dataset_info=dataset_info,
            return_heatmap=return_heatmap,
            outputs=output_layer_names)

        # show the results
        vis_img = vis_pose_result(
            pose_model,
            img,
            pose_results[:1],
            dataset=dataset,
            dataset_info=dataset_info,
            kpt_score_thr=config.kpt_thr,
            radius=config.radius,
            thickness=config.thickness,
            show=False)
        
        frame_index += 1
        if len(pose_results) == 0:
            result.append([None]*34) # [playback_time]+ [None]*51
            continue

        # choose the first person in results -> pose_results[0]
        pose_landmarks = np.array(
            [[keypoint[0] , keypoint[1]]
              for keypoint in pose_results[0]["keypoints"]],
            dtype=np.float32)
        
        coordinates = pose_landmarks.flatten().astype(np.str_).tolist()
        
        result.append(coordinates)

        if save_out_video:
            videoWriter.write(vis_img)
            
        percentage = int(frame_index*100/tot_frames)
        yield f"data: {{ \"percentage\":\"{str(percentage)}\", \"result\": \"\" }} \n\n"

    cap.release()

    if save_out_video:
        videoWriter.release()

    analyze_done = time.time()
    print("Video processing finished in ", analyze_done - start)
    
    '''filled_result=colab_find_bounce_up.fill_ankle(result[33])
    jumpend=algo.find_inflection(filled_result)
    result=filled_result[jumpend-49:jumpend+1][:]'''

    result_text = ""

    for lstm_classifier, i in zip(lstm_classifiers, range(len(lstm_classifiers))):
        # 1. normalize the pose landmarks
        model_input = normalize_pose_landmarks(np.array(result, dtype=np.float32))
        # 2. convert to numpy float array
        model_input = model_input.astype(np.float32)
        # 3. convert input to tensor
        model_input = torch.Tensor(model_input)
        # # 4. add extra dimension
        model_input = torch.unsqueeze(model_input, dim=0)
        # 5. predict the action class using lstm
        y_pred = lstm_classifier(model_input)
        prob = F.softmax(y_pred, dim=1)
        print(f"prob: {prob.data}")
        # get the index of the max probability
        pred_index = prob.data.max(dim=1)[1]
        # pop the first value from buffer_window and add the new entry in FIFO fashion, to have a sliding window of size 32.
        print(f"class {i} pred_index : {pred_index.numpy()[0]}")
        if pred_index.numpy()[0] == 1:
            result_text += class_names[i] + " "
    
    yield f"data: {{ \"percentage\":\"100\", \"result\": \"{result_text if result_text != '' else '做的很好！'}\"}}\n\n"



def stream_video(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    print("fps ", fps)
    print("width height", width, height)
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print("tot_frames", tot_frames)
    while True:
        ret, frame = cap.read()
        if ret == False:
            break
        out_frame = cv2.imencode('.jpg', frame)[1].tobytes()
        result = (b'--frame\r\n'b'Content-Type: image/jpeg\r\n\r\n' +
                  out_frame + b'\r\n')
        yield result
    print("finished video streaming")
