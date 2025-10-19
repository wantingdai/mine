# Author: Ethan Lee
# 2024/6/26 上午1:34
import cv2
import onnxruntime as ort
from general import plot_one_box, infer_img, detect, scale_coords, plot_one_box_new
import numpy as np

model_name = r'D:\96537\Documents\Lab\2024_ROBOCON\vision\yolov5\mch_arm_add_pure\weights\mch_arm.onnx'
net = ort.InferenceSession(model_name)
dict_labels = {0: 'red_ball', 1: 'purple_ball', 2: 'blue_ball'}  # 标签
MODEL_H, MODEL_W = 640, 640  # 模型参数
shape = (MODEL_H, MODEL_W)

cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 360)

while cap.isOpened():
    ret, image = cap.read()
    if not ret:
        break

    # det_boxes, scores, ids = infer_img(image, net, MODEL_H, MODEL_W, thred_nms=0.8, thred_cond=0.3)
    # for box, score, label_id in zip(det_boxes, scores, ids):  # 在处理过的图像上面进行操作
    #     label = '%s' % (dict_labels[label_id])
    #     det_ball_x, det_ball_y = plot_one_box(box.astype(np.int16), image,
    #                                           color=(0, 0, 255) if label == 'red_ball' else (255, 0, 0),
    #                                           label=label, line_thickness=None)
    img, pred_boxes, pred_confes, pred_classes = detect(image, net, MODEL_H, MODEL_W, 0.45, 0.5)
    if len(pred_boxes) > 0:
        for i, _ in enumerate(pred_boxes):
            ball_x, ball_y = plot_one_box_new(image, img, shape, pred_boxes[i], pred_classes[i], pred_confes[i])
            print(f'中心点：{ball_x, ball_y}')

    cv2.imshow('mch_arm', image)

    if cv2.waitKey(1) == 27:
        break

cv2.destroyAllWindows()
cap.release()
