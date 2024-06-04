import os
import time
import cv2
import ntpath

from lib.config import Config
from .lstm import WINDOW_SIZE
from lib.tools import verify_video, convert_video, get_video_info

import numpy as np
from pathlib import Path
import torch
import torch.nn.functional as F
import warnings

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         process_mmdet_results, vis_pose_result)
from mmpose.datasets import DatasetInfo

try:
    from mmdet.apis import inference_detector, init_detector
    has_mmdet = True
except (ImportError, ModuleNotFoundError):
    has_mmdet = False

assert has_mmdet, 'Please install mmdet to run the demo.'

LABELS = {
    0: "JUMPING",
    1: "JUMPING_JACKS",
    2: "BOXING",
    3: "WAVING_2HANDS",
    4: "WAVING_1HAND",
    5: "CLAPPING_HANDS"
}

# how many frames to skip while inferencing
# configuring a higher value will result in better FPS (frames per rate), but accuracy might get impacted
SKIP_FRAME_COUNT = 0

# analyse the video
def analyse_video(lstm_classifier, video_path):
    
    #print("Label detected ", label)
    file_name = ntpath.basename(video_path)
    config = Config(video_path, 'res_{}'.format(file_name))
    print(f"Starting video processing: {video_path} saved as \"res_{file_name}\"")
    det_model = init_detector(
        config.det_config, config.det_checkpoint, device=config.device.lower())
    # build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        config.pose_config, config.pose_checkpoint, device=config.device.lower())

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
        
        # May not need in video mode        
        ### Save landmarks if all landmarks were detected
        # min_landmark_score = min(
        #     [keypoint[2] for keypoint in person_results["keypoints"]])
        # should_keep_image = min_landmark_score >= detection_threshold
        # if not should_keep_image:
        #   self._messages.append('Skipped ' + video_path +
        #                         '. No pose was confidentlly detected.')
        #   continue
        
        ## Dont need to record the playback time bc have already fix the FPS of each video
        # playback_time = np.round((frame_index/fps)*1000, 1)
        # playback_time in milliseconds
        
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
    
    # convert input to tensor
    model_input = torch.Tensor(np.array(result, dtype=np.float32))
    # add extra dimension
    model_input = torch.unsqueeze(model_input, dim=0)
    # predict the action class using lstm
    y_pred = lstm_classifier(model_input)
    prob = F.softmax(y_pred, dim=1)
    # get the index of the max probability
    pred_index = prob.data.max(dim=1)[1]
    # pop the first value from buffer_window and add the new entry in FIFO fashion, to have a sliding window of size 32.
    label = LABELS[pred_index.numpy()[0]]
    print("Label detected ", label)
    yield f"data: {{ \"percentage\":\"100\", \"result\": \"{label}\"}}\n\n"
    #print("Label detected ", label)
    # yield label
    # save the result to a csv file
    # if csv_out_path:
    #     np.savetxt(csv_out_path, result, delimiter=",", fmt='%s')
    


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


"""
    # open the video
    cap = cv2.VideoCapture(video_path)
    # width of image frame
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    # height of image frame
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    # frames per second of the input video
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    # total number of frames in the video
    tot_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    # video output codec
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    # extract the file name from video path
    file_name = ntpath.basename(video_path)
    # video writer
    vid_writer = cv2.VideoWriter('res_{}'.format(
        file_name), fourcc, 30, (width, height))
    # counter
    counter = 0
    # buffer to keep the output of detectron2 pose estimation
    buffer_window = []
    # start time
    start = time.time()
    label = None
    # iterate through the video
    while True:
        # read the frame
        ret, frame = cap.read()
        # return if end of the video
        if ret == False:
            break
        # make a copy of the frame
        img = frame.copy()
        if(counter % (SKIP_FRAME_COUNT+1) == 0):
            # predict pose estimation on the frame
            outputs = pose_detector(frame)
            # filter the outputs with a good confidence score
            persons, pIndicies = filter_persons(outputs)
            if len(persons) >= 1:
                # pick only pose estimation results of the first person.
                # actually, we expect only one person to be present in the video.
                p = persons[0]
                # draw the body joints on the person body
                draw_keypoints(p, img)
                # input feature array for lstm
                features = []
                # add pose estimate results to the feature array
                for i, row in enumerate(p):
                    features.append(row[0])
                    features.append(row[1])

                # append the feature array into the buffer
                # not that max buffer size is 32 and buffer_window operates in a sliding window fashion
                if len(buffer_window) < WINDOW_SIZE:
                    buffer_window.append(features)
                else:
                    # convert input to tensor
                    model_input = torch.Tensor(np.array(buffer_window, dtype=np.float32))
                    # add extra dimension
                    model_input = torch.unsqueeze(model_input, dim=0)
                    # predict the action class using lstm
                    y_pred = lstm_classifier(model_input)
                    prob = F.softmax(y_pred, dim=1)
                    # get the index of the max probability
                    pred_index = prob.data.max(dim=1)[1]
                    # pop the first value from buffer_window and add the new entry in FIFO fashion, to have a sliding window of size 32.
                    buffer_window.pop(0)
                    buffer_window.append(features)
                    label = LABELS[pred_index.numpy()[0]]
                    #print("Label detected ", label)

        # add predicted label into the frame
        if label is not None:
            cv2.putText(img, 'Action: {}'.format(label),
                        (int(width-400), height-50), cv2.FONT_HERSHEY_COMPLEX, 0.9, (102, 255, 255), 2)
        # increment counter
        counter += 1
        # write the frame into the result video
        vid_writer.write(img)
        # compute the completion percentage
        percentage = int(counter*100/tot_frames)
        # return the completion percentage
        yield "data:" + str(percentage) + "\n\n"
    analyze_done = time.time()
    print("Video processing finished in ", analyze_done - start)
"""
