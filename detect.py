

import argparse
from email.policy import default

import cv2
from utils.model_ops import RepointCV

from utils.trt_ops import torch2onnx


def detect(opt):
    model = RepointCV(opt.weights)

    try:
        img = cv2.imread(opt.source)
        img = model.preprocess(img)
        pred = model.infer(img)
        model.destroy()
        print("Successful model load and inference")
    except Exception as e:
        model.destroy()
        print("Failed model load and inference")
    
    print(pred[2].shape)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='weights/crowdhuman_yolov5m_384x640.engine', help='model.engine path')
    parser.add_argument('--source', type=str, default='data/images/zidane.jpg', help='source')  # file, 0 for webcam
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')

    opt = parser.parse_args()
    detect(opt)

