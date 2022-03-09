import ctypes
import cv2
import numpy as np
import json
import time
import os
import math
import argparse

from models.yolov5_trt import YoLov5TRT
import utils

WINDOW_SHAPE = (1280, 720)
FPS_THRESHOLD = 1  # each 1 frame
FREEZE_TIME = 1  # waiting time for each frame in milliseconds


def define_working_directory():
    cwd = os.getcwd()
    target_path = '/'.join(cwd.split('/')[-2:])
    if not target_path.endswith('tensorrtx/yolov5'):
        os.chdir(cwd + 'tensorrtx/yolov5')


def display_subjects(args):
    # choosing a video source
    if args.video_path is not None:
        video_file = 'VideoFile'
        video_source = args.video_path
    else:
        video_file = 'camera'
        # video_source = f'rtsp://{login}:{password}@{host}'
        video_source = args.host
    
    global WINDOW_SHAPE
    if args.mini:
        WINDOW_SHAPE = (720, 405)
    
    try:
        model = YoLov5TRT(args.engine_path)
    except AttributeError as e:
        print(e)
        print("Model load has failed!")
        return


    # starting person detection
    print('Connecting to the camera...')
    video_reader = cv2.VideoCapture(video_source)
    frame_counter = 0
    try:
        global FPS_THRESHOLD 
        print('Person detection has started...')
        while video_reader.isOpened():
            status, frame = video_reader.read()
            if not status:
                break

            frame_counter += 1

            boxes, scores, use_time = model.infer_subjects(frame)

            print('Input from {}, time->{:.2f}ms'.format(video_source, use_time * 1000))
            if args.visualize:
                frame = cv2.resize(frame, WINDOW_SHAPE)
                for i in range(len(boxes)):
                    utils.draw.plot_one_box(boxes[i], frame, color=(0, 255, 0), label="{:.2f}".format(scores[i]))

                cv2.imshow(f'Video_{video_file}', frame)
                ch = cv2.waitKey(FREEZE_TIME)
                if ch == 27 or ch == ord('q'):
                    break
                elif ch == ord('s'):
                    cv2.imwrite(f'/home/asylbek/Pictures/{frame_counter}.jpg', frame)

    finally:
        if args.visualize:
            cv2.destroyAllWindows()

        model.destroy()
        print('Person detection has finished!')



def transmit_live(args):
    video_file = 'camera'
    if args.host is not None:
        if args.camera == 'intel':
            video_source = args.host
        elif args.camera == 'hikvision':
            video_source = f'rtsp://{args.login}:{args.password}@{args.host}'
        else:
            raise ValueError("The type of the camera must be 'intek' or 'hikvision'")
    else:
        video_source = 4
    video_reader = cv2.VideoCapture(video_source)

    global WINDOW_SHAPE
    if args.mini:
        WINDOW_SHAPE = (720, 405)

    frame_counter = 0
    while video_reader.isOpened():
        status, frame = video_reader.read()
        if not status:
            break

        frame_counter += 1
        frame = cv2.resize(frame, WINDOW_SHAPE)
        cv2.imshow(f'Video_{video_file}', frame)
        ch = cv2.waitKey(FREEZE_TIME)
        if ch == 27 or ch == ord('q'):
            break
        elif ch == ord('s'):
            cv2.imwrite(f'/home/asylbek/Pictures/{frame_counter}.jpg', frame)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--login', default=None, help='The login of the camera')
    parser.add_argument('--password', default=None, help='The password of the camera')
    parser.add_argument('--host', default=None, help='The IP address of the camera')
    parser.add_argument('--mini', default=False, help='The window size is reduced if it is true')
    parser.add_argument('--engine-path', default="weights/yolov5m.engine", help='The path to a engine file for testing')
    parser.add_argument('--plugin-path', default="weights/libmyplugins.so", help='The path to a plugin file for testing')
    parser.add_argument('--video-path', default=None, help='The path to a video file for testing')
    parser.add_argument('--visualize', default=False, action='store_true', help='The window displays the videostream if it is true')
    parser.add_argument('--camera', default='intel', help="The type of the camera - 'intel' or 'hikvision'")
    args = parser.parse_args()
    login = args.login
    password = args.password
    host = args.host
    mini = args.mini
    engine_path = args.engine_path
    plugin_path = args.plugin_path
    video_path = args.video_path
    visualize = args.visualize

    # define_working_directory()
    ctypes.CDLL(plugin_path)

    categories = ["person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
            "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
            "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
            "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
            "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
            "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
            "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
            "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
            "hair drier", "toothbrush"]

    transmit_live(args)
    # display_subjects(args)
