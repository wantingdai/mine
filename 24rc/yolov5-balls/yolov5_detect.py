import cv2
import numpy as np
import _thread
import onnxruntime as rt

result = []

def nms(pred, conf_thres, iou_thres):
    # 置信度抑制，小于置信度阈值则删除
    conf = pred[..., 4] > conf_thres
    box = pred[conf == True]
    # 类别获取
    cls_conf = box[..., 5:]
    cls = []
    for i in range(len(cls_conf)):
        cls.append(int(np.argmax(cls_conf[i])))
    # 获取类别
    total_cls = list(set(cls))  #删除重复项，获取出现的类别标签列表,example=[0, 17]
    output_box = []   #最终输出的预测框
    # 不同分类候选框置信度
    for i in range(len(total_cls)):
        clss = total_cls[i]   #当前类别标签
        # 从所有候选框中取出当前类别对应的所有候选框
        cls_box = []
        for j in range(len(cls)):
            if cls[j] == clss:
                box[j][5] = clss
                cls_box.append(box[j][:6])
        cls_box = np.array(cls_box)
        box_conf = cls_box[..., 4]   #取出候选框置信度
        box_conf_sort = np.argsort(box_conf)   #获取排序后索引
        max_conf_box = cls_box[box_conf_sort[len(box_conf) - 1]]
        output_box.append(max_conf_box)   #将置信度最高的候选框输出为第一个预测框
        cls_box = np.delete(cls_box, 0, 0)  #删除置信度最高的候选框
        while len(cls_box) > 0:
            max_conf_box = output_box[len(output_box) - 1]     #将输出预测框列表最后一个作为当前最大置信度候选框
            del_index = []
            for j in range(len(cls_box)):
                current_box = cls_box[j]      #当前预测框
                interArea = getInter(max_conf_box, current_box)    #当前预测框与最大预测框交集
                iou = getIou(max_conf_box, current_box, interArea)  # 计算交并比
                if iou > iou_thres:
                    del_index.append(j)   #根据交并比确定需要移出的索引
            cls_box = np.delete(cls_box, del_index, 0)   #删除此轮需要移出的候选框
            if len(cls_box) > 0:
                output_box.append(cls_box[0])
                cls_box = np.delete(cls_box, 0, 0)
    return output_box


#计算并集
def getIou(box1, box2, inter_area):
    box1_area = box1[2] * box1[3]
    box2_area = box2[2] * box2[3]
    union = box1_area + box2_area - inter_area
    iou = inter_area / union
    return iou


#计算交集
def getInter(box1, box2):
    box1_x1, box1_y1, box1_x2, box1_y2 = box1[0], box1[1], box1[0] + box1[2], box1[1] + box1[3]
    box2_x1, box2_y1, box2_x2, box2_y2 = box2[0], box2[1], box2[0] + box2[2], box2[1] + box2[3]
    if box1_x1 > box2_x2 or box1_x2 < box2_x1:
        return 0
    if box1_y1 > box2_y2 or box1_y2 < box2_y1:
        return 0
    x_list = [box1_x1, box1_x2, box2_x1, box2_x2]
    x_list = np.sort(x_list)
    x_inter = x_list[2] - x_list[1]
    y_list = [box1_y1, box1_y2, box2_y1, box2_y2]
    y_list = np.sort(y_list)
    y_inter = y_list[2] - y_list[1]
    inter = x_inter * y_inter
    return inter


def infer(sess, input, label, img):
    height, width = 640, 640
    x_scale = img.shape[1] / width
    y_scale = img.shape[0] / height
    img = cv2.resize(img, (height, width))  # 尺寸变换
    img = img / 255.
    img = img[:, :, ::-1].transpose((2, 0, 1))  # HWC转CHW
    data = np.expand_dims(img, axis=0)  # 扩展维度至[1,3,640,640]
    pred = sess.run([label], {input: data.astype(np.float32)})[0]
    pred = np.squeeze(pred)
    out = nms(pred, 0.8, 0.45)
    if len(out):
        for detect in out:
            detect = [int((detect[0] - detect[2] / 2) * x_scale), int((detect[1] - detect[3] / 2) * y_scale),
                      int((detect[0] + detect[2] / 2) * x_scale), int((detect[1] + detect[3] / 2) * y_scale)]
            result.append(detect)


if __name__ == "__main__":
    sess = rt.InferenceSession(r'D:\96537\Documents\Lab\2024_ROBOCON\vision\yolov5\mch_arm_add_pure\weights\mch_arm.onnx')
    input_name = sess.get_inputs()[0].name
    label_name = sess.get_outputs()[0].name
    # 开启本地摄像头
    capture = cv2.VideoCapture(0)
    # 开启ip摄像头
    # video = "rtsp://admin:admin@192.168.1.158:8554/live"  # 此处@后的ipv4 地址需要改为app提供的地址
    # capture = cv2.VideoCapture(video)
    success, img0 = capture.read()
    i = 0
    while success:
        i += 1
        if i == 5:       #每隔5帧进行一次检测
            result = []
            i = 0
            try:
                _thread.start_new_thread(infer, (sess, input_name, label_name, img0))
            except:
                print('thread error')
        print(result)
        for j in range(len(result)):
            img0 = cv2.rectangle(img0, (result[j][0], result[j][1]), (result[j][2], result[j][3]), (0, 255, 0), 1)
        img0 = cv2.flip(img0,1,dst=None)
        cv2.imshow("camera", img0)
        # 按键处理，注意，焦点应当在摄像头窗口，不是在终端命令行窗口
        key = cv2.waitKey(1)
        if key == 27:
            break
        success, img0 = capture.read()

    capture.release()
    cv2.destroyWindow("camera")




