import torch
from utils.model_ops import attempt_load
from utils.torch_utils import select_device

def torch2onnx(opt):
    weights, source, device = opt.weights, opt.source, opt.device
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA

    model = attempt_load(weights, map_location=device)  # load FP32 model
    stride = int(model.stride.max())  # model stride

    if half:
        model.half()
     
    a = 5


