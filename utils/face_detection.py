"""
Модуль операций по обнаружению лиц
"""
import cv2
import numpy as np
from skimage import transform as trans


def preprocess_image(image_raw, input_size):
    """
    Масштабирует изображение под выбранный/определенный размер
    Args:
        image_raw: входящее сырое изображение
        input_size: выбранный новый размер изображения

    Returns:
        det_img: масштабированное изображение
        det_scale: коэффициент масштабирования
    """
    im_ratio = float(image_raw.shape[0]) / image_raw.shape[1]
    model_ratio = float(input_size[1]) / input_size[0]
    if im_ratio > model_ratio:
        new_height = input_size[1]
        new_width = int(new_height / im_ratio)
    else:
        new_width = input_size[0]
        new_height = int(new_width * im_ratio)
    det_scale = float(new_height) / image_raw.shape[0]
    resized_img = cv2.resize(image_raw, (new_width, new_height))

    det_img = np.zeros((input_size[1], input_size[0], 3), dtype=np.uint8)
    det_img[:new_height, :new_width, :] = resized_img

    return det_img, det_scale


def post_process(net_outs, blob, threshold, det_scale):
    """
    Переводит результаты предсказаний на последнем слое модели на формат/масштаб изображений

    Args:
        net_outs: резульаты предсказания последнего слоя модели (нейронной сети)
        blob: входные параметры модели - размеры изображения
        threshold: порог для выбора предсказанных регионов лица
        det_scale: коэффициент масштабирования

    Returns:
        det (np.array(n, 5)): координаты регионов лиц
        kpss (np.array(n, 5, 2)): координаты ключевых точек лиц
    """
    scores_list = []
    bboxes_list = []
    kpss_list = []
    fmc = 3
    num_anchors = 2
    center_cache = {}

    input_height = blob.shape[2]
    input_width = blob.shape[3]

    for idx, stride in enumerate([8, 16, 32]):
        scores = net_outs[idx * fmc]
        scores = np.reshape(scores, (-1, 1))

        bbox_preds = net_outs[idx * fmc + 1]
        bbox_preds = bbox_preds * stride
        bbox_preds = np.reshape(bbox_preds, (-1, 4))

        kps_preds = net_outs[idx * fmc + 2] * stride
        kps_preds = np.reshape(kps_preds, (-1, 10))

        height = input_height // stride
        width = input_width // stride
        key = (height, width, stride)
        if key in center_cache:
            anchor_centers = center_cache[key]
        else:
            anchor_centers = np.stack(np.mgrid[:height, :width][::-1], axis=-1).astype(np.float32)
            anchor_centers = (anchor_centers * stride).reshape((-1, 2))
            anchor_centers = np.stack([anchor_centers] * num_anchors, axis=1).reshape((-1, 2))
            if len(center_cache) < 100:
                center_cache[key] = anchor_centers

        pos_inds = np.where(scores >= threshold)[0]
        bboxes = distance2bbox(anchor_centers, bbox_preds)
        pos_scores = scores[pos_inds]
        pos_bboxes = bboxes[pos_inds]
        scores_list.append(pos_scores)
        bboxes_list.append(pos_bboxes)

        kpss = distance2kps(anchor_centers, kps_preds)
        kpss = kpss.reshape((kpss.shape[0], -1, 2))
        pos_kpss = kpss[pos_inds]
        kpss_list.append(pos_kpss)

    scores = np.vstack(scores_list)
    scores_ravel = scores.ravel()
    order = scores_ravel.argsort()[::-1]
    bboxes = np.vstack(bboxes_list) / det_scale
    kpss = np.vstack(kpss_list) / det_scale

    pre_det = np.hstack((bboxes, scores)).astype(np.float32, copy=False)
    pre_det = pre_det[order, :]
    keep = nms(pre_det)
    det = pre_det[keep, :]

    kpss = kpss[order,:,:]
    kpss = kpss[keep,:,:]

    return det, kpss


def distance2bbox(points, distance, max_shape=None):
    """
    Смещает предсказанные координаты регионов лиц на масштаб сырого изображения

    Args:
        points: предсказанные координаты точек
    distance: длины на который должно быть смещение
    max_shape: размер масштаба на сыром изображений

    Returns:
        np.array: смещенные координаты регионов
    """
    x1 = points[:, 0] - distance[:, 0]
    y1 = points[:, 1] - distance[:, 1]
    x2 = points[:, 0] + distance[:, 2]
    y2 = points[:, 1] + distance[:, 3]
    if max_shape is not None:
        x1 = x1.clamp(min=0, max=max_shape[1])
        y1 = y1.clamp(min=0, max=max_shape[0])
        x2 = x2.clamp(min=0, max=max_shape[1])
        y2 = y2.clamp(min=0, max=max_shape[0])
    return np.stack([x1, y1, x2, y2], axis=-1)


def distance2kps(points, distance, max_shape=None):
    """
    Смещает предсказанные координаты ключевых точек лиц на масштаб сырого изображения

    Args:
        points: предсказанные координаты точек
    distance: длины на который должно быть смещение
    max_shape: размер масштаба на сыром изображений

    Returns:
        np.array: смещенные координаты ключевых точек лиц
    """
    preds = []
    for i in range(0, distance.shape[1], 2):
        px = points[:, i % 2] + distance[:, i]
        py = points[:, i % 2 + 1] + distance[:, i + 1]
        if max_shape is not None:
            px = px.clamp(min=0, max=max_shape[1])
            py = py.clamp(min=0, max=max_shape[0])
        preds.append(px)
        preds.append(py)
    return np.stack(preds, axis=-1)


def nms(dets):
    """
    Очищает повторно предсказанные регионы лиц

    Args:
        dets (np.array(n, 5)): начальный сырой список регионов

    Returns:
        keep (np.array(n, 5)): очщенный список регионов
    """
    thresh = 0.4
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep


def crop_and_normalize(img, bbox=None, landmark=None, **kwargs):
    """
    Возвращает нормализованное изображение лица

    Args:
        img: первичное изображение
        bbox: координаты региона лица на изображении
        landmark: координаты ключевых точек лица

    Returns:
        np.array(112, 112, 3): вырезанное и нормализованное (повернутое) изображение лица
    """
    M = None
    image_size = []
    str_image_size = kwargs.get('image_size', '')
    if len(str_image_size) > 0:
        image_size = [int(x) for x in str_image_size.split(',')]
        if len(image_size) == 1:
            image_size = [image_size[0], image_size[0]]
        assert len(image_size) == 2
        assert image_size[0] == 112
        assert image_size[0] == 112 or image_size[1] == 96
    if landmark is not None:
        assert len(image_size) == 2
        src = np.array([
            [30.2946, 51.6963],
            [65.5318, 51.5014],
            [48.0252, 71.7366],
            [33.5493, 92.3655],
            [62.7299, 92.2041]], dtype=np.float32)
        if image_size[1] == 112:
            src[:, 0] += 8.0
        dst = landmark.astype(np.float32)

        tform = trans.SimilarityTransform()
        tform.estimate(dst, src)
        M = tform.params[0:2, :]
        # M = cv2.estimateRigidTransform( dst.reshape(1,5,2), src.reshape(1,5,2), False)

    if M is None:
        if bbox is None:  # use center crop
            det = np.zeros(4, dtype=np.int32)
            det[0] = int(img.shape[1] * 0.0625)
            det[1] = int(img.shape[0] * 0.0625)
            det[2] = img.shape[1] - det[0]
            det[3] = img.shape[0] - det[1]
        else:
            det = bbox
        margin = kwargs.get('margin', 44)
        bb = np.zeros(4, dtype=np.int32)
        bb[0] = np.maximum(det[0] - margin / 2, 0)
        bb[1] = np.maximum(det[1] - margin / 2, 0)
        bb[2] = np.minimum(det[2] + margin / 2, img.shape[1])
        bb[3] = np.minimum(det[3] + margin / 2, img.shape[0])
        ret = img[bb[1]:bb[3], bb[0]:bb[2], :]
        if len(image_size) > 0:
            ret = cv2.resize(ret, (image_size[1], image_size[0]))
        return ret
    else:  # do align using landmark
        assert len(image_size) == 2
        warped = cv2.warpAffine(img, M, (image_size[1], image_size[0]), borderValue=0.0)
        return warped