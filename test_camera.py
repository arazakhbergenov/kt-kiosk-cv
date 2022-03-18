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

RAW_IMG_SHAPE = (640, 480)
DISPLAY_SHAPE = (640, 480)
FPS_THRESHOLD = 1  # each 1 frame
FREEZE_TIME = 1  # waiting time for each frame in milliseconds


def display_subjects(args):
    # choosing a video source
    if args.video_path is not None:
        video_file = 'VideoFile'
        video_source = args.video_path
    else:
        video_file = 'camera'
        # video_source = f'rtsp://{login}:{password}@{host}'
        video_source = args.host
        depth_source = args.depth
    
    global DISPLAY_SHAPE
    if args.mini:
        DISPLAY_SHAPE = (720, 405)
    
    try:
        model = YoLov5TRT(args.engine_path)
    except AttributeError as e:
        print(e)
        print("Model load has failed!")
        return


    # starting person detection
    print('Connecting to the camera...')
    video_reader = cv2.VideoCapture(video_source)
    depth_reader = cv2.VideoCapture(depth_source)
    frame_counter = 0
    try:
        global FPS_THRESHOLD 
        print('Person detection has started...')
        while video_reader.isOpened() and depth_reader.isOpened():
            status_rgb, frame = video_reader.read()
            status_depth, depth = depth_reader.read()
            if not status_rgb:
                print('Input from RGB Camera {} not available'.format(args.host))
                break
            if not status_depth:
                print('Input from Depth Camera {} not available'.format(args.depth))
                break

            frame_counter += 1

            frame = cv2.resize(frame, RAW_IMG_SHAPE)
            boxes, scores, use_time = model.infer_subjects(frame)
            distances, centers = model.infer_distances(depth, boxes)

            print('Input from {}, time->{:.2f}ms, frame_shape={}, depth_shape={}'.format(video_source, use_time * 1000, frame.shape, depth.shape))
            if args.visualize:
                frame = cv2.resize(frame, DISPLAY_SHAPE)
                for i in range(len(boxes)):
                    utils.draw.plot_one_box(boxes[i], frame, color=(0, 255, 0), label="{:.2f} {:.2f} {} {}".format(
                        scores[i], distances[i], centers[i, 0], centers[i, 1]))
                    cv2.circle(frame, (centers[i, 0], centers[i, 1]), 5, (0, 255, 0), thickness=-1, lineType=cv2.LINE_AA)

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

    global DISPLAY_SHAPE
    if args.mini:
        DISPLAY_SHAPE = (720, 405)

    frame_counter = 0
    while video_reader.isOpened():
        status, frame = video_reader.read()
        if not status:
            break

        frame_counter += 1
        frame = cv2.resize(frame, DISPLAY_SHAPE)
        # print(frame.shape, frame[360, 640, :])
        # print(frame[:, :, 0].shape)
        # frame = cv2.applyColorMap(cv2.convertScaleAbs(frame[:, :, 0], alpha=0.03), cv2.COLORMAP_JET)

        cv2.imshow(f'Video_{video_file}', frame)
        ch = cv2.waitKey(FREEZE_TIME)
        if ch == 27 or ch == ord('q'):
            break
        elif ch == ord('s'):
            cv2.imwrite(f'/home/asylbek/Pictures/{frame_counter}.jpg', frame)

    cv2.destroyAllWindows()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default=None, help='The IP address of the RGB camera')
    parser.add_argument('--depth', default=None, help='The IP address of the Depth camera')
    parser.add_argument('--login', default=None, help='The login of the Hikvision camera')
    parser.add_argument('--password', default=None, help='The password of the Hikvision camera')
    parser.add_argument('--camera', default='intel', help="The type of the camera - 'intel' or 'hikvision'")
    parser.add_argument('--video-path', default=None, help='The path to a video file for testing')

    parser.add_argument('--engine-path', default="weights/yolov5m.engine", help='The path to a engine file for testing')
    parser.add_argument('--plugin-path', default="weights/libmyplugins.so", help='The path to a plugin file for testing')    
    
    parser.add_argument('--mini', default=False, help='The window size is reduced if it is true')
    parser.add_argument('--visualize', default=False, action='store_true', help='The window displays the videostream if it is true')
    parser.add_argument('--live', default=False, action='store_true', help='Live-stream if it is true or run inference if it is false')
    args = parser.parse_args()

    # define_working_directory()
    ctypes.CDLL(args.plugin_path)

    if args.live:
        transmit_live(args)
    else:
        display_subjects(args)
