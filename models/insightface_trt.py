"""
Модуль собран для процедур по моделям в TensorRT формате
"""
import cv2
import numpy as np
import pycuda.autoinit
import pycuda.driver as cuda
import tensorrt as trt

import utils.face_detection as face_detection


class HostDeviceMem(object):
    """
    Простой вспомогательный класс данных, который немного удобнее использовать, чем tuple
    """
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class InsightFace:
    """
    Класс является оболочкой для операций на TensorRT
    """
    def __init__(self, engine_file_path, image_size=(112, 112)):
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
                inputs.append(HostDeviceMem(host_mem, cuda_mem))
            else:
                outputs.append(HostDeviceMem(host_mem, cuda_mem))

        # Store
        self.stream = stream
        self.context = context
        self.engine = engine
        self.inputs = inputs
        self.outputs = outputs
        self.bindings = bindings

        self.image_size = image_size

    def infer_faces(self, image_raw, threshold=0.5):
        """
        Функция обнаружения лиц

        Args:
            image_raw: (сырое) изображение на которым предположительно присутствиют лица
            threshold: порог для определения лица

        Returns:
            result_boxes: координаты регионов лиц
            result_landmarks: координаты ключевых точек лиц
        """
        det_img, det_scale = face_detection.preprocess_image(image_raw, self.image_size)
        input_size = tuple(det_img.shape[0:2][::-1])
        blob = cv2.dnn.blobFromImage(det_img, 1.0/128, input_size, (127.5, 127.5, 127.5), swapRB=True)

        net_outs = self.infer(blob)
        result_boxes, result_landmarks = face_detection.post_process(net_outs, blob, threshold, det_scale)

        return result_boxes, result_landmarks

    def get_cropped_face(self, image_raw, bbox, landmarks):
        """
        Экспортирует очищенное изображение

        Args:
            image_raw: (сырое) изображение на которым предположительно присутствиют лица
            bbox: координаты регионов лиц
            landmarks: координаты ключевых точек лиц

        Returns:
            input_image: очищенное (нормализованное) изображение лица
        """
        nimg = face_detection.crop_and_normalize(image_raw, bbox, landmarks, image_size='112,112')
        nimg = cv2.cvtColor(nimg, cv2.COLOR_BGR2RGB)
        input_image = np.transpose(nimg, (2, 0, 1))

        return input_image

    def get_age_gender(self, input_image):
        """
        Предсказывает пол и возраст по изображению лица

        Args:
            input_image: очищенное (нормализованное) изображение лица

        Returns:
            gender: пол
            age: возраст
        """
        host_outputs = self.infer(input_image)
        pred = host_outputs[0]
        g = pred[0:2]
        gender = np.argmax(g)
        a = pred[2:202].reshape((100, 2))
        a = np.argmax(a, axis=1)
        age = int(sum(a))
        return age, gender

    def infer(self, input_image):
        """
        Возвращает результаты последнего слоя модели (нейронной сети) на TensorRT

        Args:
            input_image: очищенное (нормализованное) изображение лица

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
        np.copyto(inputs[0].host, input_image.ravel())
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

    def destroy(self):
        """
        Освобождает ресурсы модели формата TensorRT на видеокарте
        """
        self.cfx.pop()
