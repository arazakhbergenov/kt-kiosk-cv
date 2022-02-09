import torch
import torch.nn as nn
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt
from utils.boxes import letterbox, letterbox2


from utils.web_ops import attempt_download
   
class Ensemble(nn.ModuleList):
    # Ensemble of models
    def __init__(self):
        super(Ensemble, self).__init__()

    def forward(self, x, augment=False):
        y = []
        for module in self:
            y.append(module(x, augment)[0])
        # y = torch.stack(y).max(0)[0]  # max ensemble
        # y = torch.stack(y).mean(0)  # mean ensemble
        y = torch.cat(y, 1)  # nms ensemble
        return y, None  # inference, train output


def attempt_load(weights, map_location=None):
    # Loads an ensemble of models weights=[a,b,c] or a single model weights=[a] or weights=a
    model = Ensemble()
    for w in weights if isinstance(weights, list) else [weights]:
        attempt_download(w)
        model.append(torch.load(w, map_location=map_location)['model'].float().fuse().eval())  # load FP32 model

    # Compatibility updates
    for m in model.modules():
        if type(m) in [nn.Hardswish, nn.LeakyReLU, nn.ReLU, nn.ReLU6, nn.SiLU]:
            m.inplace = True  # pytorch 1.7.0 compatibility
        elif type(m) is Conv:
            m._non_persistent_buffers_set = set()  # pytorch 1.6.0 compatibility

    if len(model) == 1:
        return model[-1]  # return model
    else:
        print('Ensemble created with %s\n' % weights)
        for k in ['names', 'stride']:
            setattr(model, k, getattr(model[-1], k))
        return model  # return ensemble


class HostDevice:
    """
    Простой вспомогательный класс данных, который немного удобнее использовать, чем tuple
    """
    def __init__(self, host, device):
        self.host = host
        self.device = device

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class RepointCV:
    """
    Класс является оболочкой для операций на TensorRT
    """
    def __init__(self, engine_file_path, image_size=(640, 384)):
        """
        Создание контекста модели на видеокарте и десериализация от engine файлов

        Args:
            engine_file_path: путь к engine файлу (модели на TensorRT)
            image_size: размер изображения
        """
        # Create a Context on this device,
        self.cfx = cuda.Device(0).make_context()
        stream = cuda.Stream()
        TRT_LOGGER = trt.Logger(trt.Logger.ERROR)
        runtime = trt.Runtime(TRT_LOGGER)

        trt.init_libnvinfer_plugins(None, '')

        # Deserialize the engine from file
        with open(engine_file_path, "rb") as f:
            engine = runtime.deserialize_cuda_engine(f.read())
        context = engine.create_execution_context()

        inputs = []
        outputs = []
        bindings = []

        for binding in engine:
            size = trt.volume(engine.get_binding_shape(
                binding)) * engine.max_batch_size
            dtype = trt.nptype(engine.get_binding_dtype(binding))
            # Allocate host and device buffers
            host_mem = cuda.pagelocked_empty(size, dtype)
            cuda_mem = cuda.mem_alloc(host_mem.nbytes)
            # Append the device buffer to device bindings.
            bindings.append(int(cuda_mem))
            # Append to the appropriate list.
            if engine.binding_is_input(binding):
                inputs.append(HostDevice(host_mem, cuda_mem))
            else:
                outputs.append(HostDevice(host_mem, cuda_mem))

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings

        self.image_size = image_size
    
    def preprocess(self, img, new_shape=(384, 640)):  # height, width
        # img = letterbox(img, new_shape=self.image_size)
        img = cv2.resize(img, (new_shape[1], new_shape[0]), interpolation=cv2.INTER_LINEAR)
        img = np.ascontiguousarray(img.transpose((2, 0, 1)))
        # img = torch.from_numpy(img).to(self.device)
        img = img.astype(np.float32) / 255.0
        # img = img[None]  # ???
        return img

    def infer(self, image):
        """
        Возвращает результаты последнего слоя модели (нейронной сети) на TensorRT

        Args:
            input_image: очищенное (нормализованное) изображение

        Returns:
            final_layer: результаты последнего слоя модели
        """
        # Make self the active context, pushing it on top of the context stack.
        self.cfx.push()
        # Restore
        stream = self.stream
        context = self.context
        inputs = self.inputs
        outputs = self.outputs
        bindings = self.bindings
        # Copy input image to host buffer
        np.copyto(inputs[0].host, image.ravel())
        # Transfer input data  to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
        # Synchronize the stream
        stream.synchronize()
        # Remove any context from the top of the context stack, deactivating it.
        self.cfx.pop()
        return [out.host for out in outputs]

    def detect_people(self, image):
        img = self.preprocess(image)
        pred = self.infer(image)
        # self.postprocess(pred, img, image)
        return pred
    
    def destroy(self):
        """
        Освобождает ресурсы модели формата TensorRT на видеокарте
        """
        self.cfx.pop()
    


