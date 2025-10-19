# Author: Ethan Lee
# 2024/6/26 上午1:50

import cv2
import onnxruntime as ort
from general import plot_one_box, infer_img
import numpy as np
from static_ball_filter_eg import filter_static_objects
import time

# model_name = r'D:\96537\Documents\Lab\2024_ROBOCON\vision\yolov5\mch_arm_add_pure\weights\mch_arm.onnx'
model_name = 'mch_arm.onnx'
net = ort.InferenceSession(model_name)
dict_labels = {0: 'red_ball', 1: 'purple_ball', 2: 'blue_ball'}  # 标签
MODEL_H, MODEL_W = 640, 640  # 模型参数

cap = cv2.VideoCapture(4)
cap.set(3, 640)
cap.set(4, 360)

filter_input = []
frame_cnt = 0

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    time.sleep(0.2)  # 模拟大模型处理时延
    static_balls = []

    det_boxes, scores, ids = infer_img(image, net, MODEL_H, MODEL_W, thred_nms=0.8, thred_cond=0.5)
    for box, score, label_id in zip(det_boxes, scores, ids):  # 在处理过的图像上面进行操作
        label = '%s' % (dict_labels[label_id])
        det_ball_x, det_ball_y = plot_one_box(box.astype(np.int16), image,
                                              color=(0, 0, 255) if label == 'red_ball' else (255, 0, 0),
                                              label=label, line_thickness=None)

        static_balls.append((det_ball_x, det_ball_y))
    frame_cnt += 1
    filter_input.append(static_balls)

    cv2.imshow('mch_arm', image)
    if frame_cnt < 3:
        continue

    # print(f'filter_input: {filter_input}')
    # static_objects = filter_static_objects(filter_input, 30, 10000)
    static_objects = filter_static_objects(filter_input, 30, 3)
    print(f'filter_input: {filter_input}')

    filter_input = []
    frame_cnt = 0

    print(f"Number of static objects detected: {len(static_objects)}")
    print("Static object coordinates:")
    for obj in static_objects:
        print(f"  X: {obj[0]:.2f}, Y: {obj[1]:.2f}")

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
