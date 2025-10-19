import numpy as np
import cv2
import random


# 标注目标
def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    """
    description: Plots one bounding box on image img,
                 this function comes from YoLov5 project.
    param:
        x:      a box likes [x1,y1,x2,y2]
        img:    a opencv image object
        color:  color to draw rectangle, such as (0,255,0)
        label:  str
        line_thickness: int
    return:
        no return
    """
    tl = (
            line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1
    )  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    u = int((x[0] + x[2]) / 2)
    v = int((x[1] + x[3]) / 2)
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    cv2.circle(img, (u, v), radius=3, color=color, thickness=-1)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(
            img,
            label,
            (c1[0], c1[1] - 2),
            0,
            tl / 3,
            [225, 255, 255],
            thickness=tf,
            lineType=cv2.LINE_AA,
        )
    return u, v

def plot_one_box_new(img0, img, shape, pred_boxes, pred_class, pred_conf):
    box = pred_boxes
    left, top, width, height = box[0], box[1], box[2], box[3]
    box = (left, top, left + width, top + height)
    box = np.squeeze(
        scale_coords(shape, np.expand_dims(box, axis=0).astype("float"), img.shape[:2]).round(), axis=0).astype(
        "int")  # 进行坐标还原
    x0, y0, x1, y1 = box[0], box[1], box[2], box[3]
    # 执行画图函数
    cv2.rectangle(img0, (x0, y0), (x1, y1), (0, 0, 255), thickness=2)
    cv2.putText(img0, '{0}--{1:.2f}'.format(pred_class, pred_conf), (x0, y0 - 10),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), thickness=1)

    ball_x, ball_y = (x0 + (x1 - x0) // 2, y0 + (y1 - y0) // 2)
    cv2.circle(img0, (ball_x, ball_y), 1, (203, 192, 255), 2)
    return ball_x, ball_y


# 极大值抑制
def post_process_opencv(outputs, model_h, model_w, img_h, img_w, thred_nms, thred_cond):
    conf = outputs[:, 4].tolist()
    c_x = outputs[:, 0] / model_w * img_w
    c_y = outputs[:, 1] / model_h * img_h
    w = outputs[:, 2] / model_w * img_w
    h = outputs[:, 3] / model_h * img_h
    p_cls = outputs[:, 5:]
    if len(p_cls.shape) == 1:
        p_cls = np.expand_dims(p_cls, 1)
    cls_id = np.argmax(p_cls, axis=1)

    p_x1 = np.expand_dims(c_x - w / 2, -1)
    p_y1 = np.expand_dims(c_y - h / 2, -1)
    p_x2 = np.expand_dims(c_x + w / 2, -1)
    p_y2 = np.expand_dims(c_y + h / 2, -1)
    areas = np.concatenate((p_x1, p_y1, p_x2, p_y2), axis=-1)

    areas = areas.tolist()
    ids = cv2.dnn.NMSBoxes(areas, conf, thred_cond, thred_nms)
    if len(ids) > 0:
        return np.array(areas)[ids], np.array(conf)[ids], cls_id[ids]
    else:
        return [], [], []


def letterbox(img, new_shape=(640, 640), auto=False, scaleFill=False, scaleUp=True):
    """
    python的信封图片缩放
    :param img: 原图
    :param new_shape: 缩放后的图片
    :param color: 填充的颜色
    :param auto: 是否为自动
    :param scaleFill: 填充
    :param scaleUp: 向上填充
    :return:
    """
    shape = img.shape[:2]  # current shape[height,width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleUp:
        r = min(r, 1.0)  # 确保不超过1
    ration = r, r  # width,height 缩放比例
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]
    if auto:
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)
    elif scaleFill:
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ration = new_shape[1] / shape[1], new_shape[0] / shape[0]
    # 均分处理
    dw /= 2
    dh /= 2
    if shape[::-1] != new_unpad:
        img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(114, 114, 114))  # 添加边界
    return img, ration, (dw, dh)

def preprocess(img, img_size):
    """
    图片预处理过程
    :param img:
    :return:
    """
    img0 = img.copy()
    img = letterbox(img, new_shape=img_size)[0]  # 图片预处理
    img = img[:, :, ::-1].transpose(2, 0, 1)
    img = np.ascontiguousarray(img).astype(np.float32)
    img /= 255.0
    img = np.expand_dims(img, axis=0)
    assert len(img.shape) == 4
    return img0, img

# 对图像进行推理(核心)
def infer_img(img0, net, model_h, model_w, thred_nms, thred_cond):
    # 图像预处理
    # img = cv2.resize(img0, [model_w, model_h], interpolation=cv2.INTER_AREA)  # 缩放
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # 格式转换
    # img = img.astype(np.float32) / 255.0  # 归一化
    # blob = np.expand_dims(np.transpose(img, (2, 0, 1)), axis=0)  # 维度转换
    img0, blob = preprocess(img0, (model_h, model_w))

    # 模型推理
    outs = net.run(None, {net.get_inputs()[0].name: blob})[0].squeeze(axis=0)

    # 输出坐标矫正
    # outs = cal_outputs(outs, nl, na, model_w, model_h, anchor_grid, stride)

    # 检测框计算
    img_h, img_w, _ = np.shape(img0)
    boxes, confs, ids = post_process_opencv(outs, model_h, model_w, img_h, img_w, thred_nms, thred_cond)

    return boxes, confs, ids


def clip_coords(boxes, img_shape):
    """
    图片的边界处理
    :param boxes: 检测框
    :param img_shape: 图片的尺寸
    :return:
    """
    boxes[:, 0].clip(0, img_shape[1])  # x1
    boxes[:, 1].clip(0, img_shape[0])  # y1
    boxes[:, 2].clip(0, img_shape[1])  # x2
    boxes[:, 3].clip(0, img_shape[0])  # x2

def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    """
    坐标还原
    :param img1_shape: 旧图像的尺寸
    :param coords: 坐标
    :param img0_shape:新图像的尺寸
    :param ratio_pad: 填充率
    :return:
    """
    if ratio_pad is None:  # 从img0_shape中计算
        gain = min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])  # gain=old/new
        pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2
    else:
        gain = ratio_pad[0][0]
        pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, :4] /= gain
    clip_coords(coords, img0_shape)
    return coords

def detect(im, net, model_h, model_w, iou_thres, conf_thres):
    """

    :param img:
    :return:
    """
    img0, img = preprocess(im, (model_h, model_w))
    pred = net.run(None, {net.get_inputs()[0].name: img})[0]  # 执行推理
    pred = pred.astype(np.float32)
    pred = np.squeeze(pred, axis=0)
    boxes = []
    classIds = []
    confidences = []
    for detection in pred:
        scores = detection[5:]
        classID = np.argmax(scores)
        confidence = scores[classID] * detection[4]  # 置信度为类别的概率和目标框概率值得乘积

        if confidence > conf_thres:
            box = detection[0:4]
            (centerX, centerY, width, height) = box.astype("int")
            x = int(centerX - (width / 2))
            y = int(centerY - (height / 2))
            boxes.append([x, y, int(width), int(height)])
            classIds.append(classID)
            confidences.append(float(confidence))
    idxs = cv2.dnn.NMSBoxes(boxes, confidences, conf_thres, iou_thres)  # 执行nms算法
    pred_boxes = []
    pred_confes = []
    pred_classes = []
    if len(idxs) > 0:
        for i in idxs.flatten():
            confidence = confidences[i]
            if confidence >= conf_thres:
                pred_boxes.append(boxes[i])
                pred_confes.append(confidence)
                pred_classes.append(classIds[i])
    return im, pred_boxes, pred_confes, pred_classes
    # return pred_boxes, pred_confes, pred_classes
