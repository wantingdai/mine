# import pyrealsense2 as rs
import time

import onnxruntime as ort
import numpy as np
import cv2
# from apply_on_R2 import getAvailableActions, decide_action, isEndGame, loadPolicy
# from GetBall import send_data_def
from general import plot_one_box, infer_img, detect, plot_one_box_new

# ser = send_data_def()  # 打开串口

'''
以下变量没用到：
silo_x, silo_y, silo_list, RedBall_x, RedBall_y, BlueBall_x, BlueBall_y,
RedBall_x_values, RedBall_y_values, BlueBall_x_values, BlueBall_y_values = [], [], [], []
'''

# -------------------------------- #
# 1表示我方颜色球，-1表示敌方颜色球
# 以下是分别就持红球和持蓝球进行讨论

"""
@ Red_Silo各个参数注解：
    1、第一行和第二行的global的各个参数都是用于处理视觉看到的球，为了将视觉看到的球变成二维数组的形式
    2、第三行的global的各个参数是用于进行决策。其中value是作为权重进行处理
       举例：第一个球框中有两个红球，第二个球框中有两个蓝球，那么最终决策会优先选择将红球放到第二个球框中，避免敌方大胜
       temp记录第i个球框的球的个数(不管红球还是蓝球)，maxx用于记录上文最大的value的值，用于比对找到第i个球框
       deceision表示决策的结果，有且仅有0, 1, 2, 3, 4五个数字，我们绝不坐以待毙！我们把命运掌握在自己手里！
       layers用于进行决策的层数的处理，分为三个决策层数，分别对照三个情况，默认layers的值为1
       情况1: 存在有已经放好了两层的球的球框，此时优先级最高，因此除非已经决策完成，否则都会优先执行该步骤，当成功对有两层球的球框进行封顶之后，layers会变为-1，决策完成
       情况2( if layers == 1 ): 没有放了两层球的球框，寻找空框，当找到空框后，layers变为-1，决策完成，否则，layers会变为2，进入第三种情况
       情况3( if layers == 2 ): 找第一层是我方球的球框放球，放完球，决策完成，layers变为-1
"""


class Silo:
    def __init__(self, SIDE: str):
        # 模型
        # self.model_name = 'v5l-med_scratch-silo.onnx'  # 模型名称
        self.model_name = 'silo_m_com.onnx'  # 模型名称
        self.net = ort.InferenceSession(self.model_name)  # 读取模型
        # self.label_dict = {0: 'red_ball', 1: 'blue_ball', 2: 'silo'}  # 标签
        self.label_dict = {0: 'red_ball', 1: 'purple_ball', 2: 'blue_ball'}  # 标签
        self.MODEL_H, self.MODEL_W = 640, 640  # 模型参数
        self.shape = (self.MODEL_W, self.MODEL_H)

        # 识别
        self.ball_list = {'red_ball': [], 'blue_ball': []}  # 不同颜色球的记录列表
        self.silo_state = [[0, 0, 0] for _ in range(5)]  # 框内球的情况，1表示我方颜色球，-1表示敌方颜色球
        self.SILO_INTERVAL = [0, 125, 250, 375, 500, 625]  # 框位置x轴区间划分 (pixel)

        # 决策
        self.max_val = 1
        self.decision = -1
        self.layers = 1
        self.dec_buf = {  # decision buffer
            'red_ball': {k: [0, 0, 0, 0, 0] for k in ['num', 'silo', 'delta']},
            'blue_ball': {k: [0, 0, 0, 0, 0] for k in ['num', 'silo', 'delta']},
            'value': [0, 0, 0, 0, 0],
            'tmp': [0, 0, 0, 0, 0]
        }
        # Note: 下面这样写也是可以初始化的
        # for k in self.dec_buf.keys():  # 初始化 decision buffer
        #     if type(self.dec_buf[k]) is list:  # Test
        #         self.dec_buf[k] = [0 for _ in range(5)]
        #     elif type(self.dec_buf[k]) is dict:
        #         for k_ in self.dec_buf[k].keys():  # Test
        #             self.dec_buf[k][k_] = [0 for _ in range(5)]

        assert SIDE == 'r' or SIDE == 'b' or SIDE == 'p'
        self.SIDE = 'red_ball' if SIDE == 'r' else 'blue_ball'
        self.OPP_SIDE = 'blue_ball' if SIDE == 'r' else 'red_ball'

        '''使用雷达'''
        self.silo_our_side_cnt = np.array([0 for _ in range(5)])  # 用于记忆框内有多少我方过的球
        self.silo_state = np.array(self.silo_state)
        self.YXS_cnt = 0

    def detect(self, image):  # Note: 仅统计框内球数，与阵营无关
        # det_boxes, scores, ids = infer_img(image, self.net, self.MODEL_H,
        #                                    self.MODEL_W, 0.8, 0.2)  # 0.8表示极大值抑制的参数，0.2表示置信度的参数
        # for box, score, label_id in zip(det_boxes, scores, ids):  # 在处理过的图像上面进行操作
        #     label = '%s' % (self.label_dict[label_id])
        #     det_ball_x, det_ball_y = plot_one_box(box.astype(np.int16), image,
        #                                           color=(0, 0, 255) if label == 'red_ball' else (255, 0, 0),
        #                                           label=label, line_thickness=None)
        #     self.ball_list[label].append([det_ball_x, det_ball_y])
        img, pred_boxes, pred_confes, pred_classes = detect(image, self.net, self.MODEL_H, self.MODEL_W, 0.3,
                                                            0.3)
        if len(pred_boxes) > 0:
            for i, _ in enumerate(pred_boxes):
                label = '%s' % (self.label_dict[pred_classes[i]])  # 标签，利用标签得   `1到对红球的判断
                x, y = plot_one_box_new(image, img, self.shape, pred_boxes[i], pred_classes[i], pred_confes[i])
                self.ball_list[label].append([x, y])

        cv2.imshow('Silo', image)  # 图像展示
        if cv2.waitKey(1) == 27:
            pass

        for k in self.ball_list.keys():  # 选择字典键值(颜色控制)
            # 统计识别出来的在第0个框的红球有几个，如果不做统计，则下次再看时会因为有红球在第0个框的范围而将红球在第0个框的数量增多
            for i in range(len(self.ball_list[k])):  # 选择键值对应的列表
                for j in range(5):  # 区间控制
                    if self.SILO_INTERVAL[j] < self.ball_list[k][i][0] <= self.SILO_INTERVAL[j + 1]:
                        self.dec_buf[k]['num'][j] += 1

    def retry_record_our_side(self):
        self.silo_our_side_cnt = self.dec_buf[self.SIDE]['num']
        print(f'our side: {self.silo_our_side_cnt}')

    def sort(self):  # Note: 与阵营相关。整理球筐，打上阵营信息，为决策做准备
        # 由于红球和蓝球的个数并不相同，因此红球数组的长度和蓝球的数组长度也不同，因此需要分开记录
        # 统计球筐内球数
        for i in range(5):
            for j in range(3):
                # 统计各球筐中的球数
                if self.silo_state[i][j] == 1:
                    self.dec_buf[self.SIDE]['silo'][i] += 1
                elif self.silo_state[i][j] == -1:
                    self.dec_buf[self.OPP_SIDE]['silo'][i] += 1

        for i in range(5):
            self.dec_buf[self.SIDE]['delta'][i] = (self.dec_buf[self.SIDE]['num'][i] -
                                                   self.dec_buf[self.SIDE]['silo'][i])
            self.dec_buf[self.OPP_SIDE]['delta'][i] = (self.dec_buf[self.OPP_SIDE]['num'][i] -
                                                       self.dec_buf[self.OPP_SIDE]['silo'][i])

        # TODO: 这。。。七层缩进，是否应该优化一下 Note: 可以用array来简化，但是懒得弄了
        for k in self.ball_list.keys():  # 颜色键值控制
            for i in range(len(self.ball_list[k])):  # 颜色内球下标控制
                for j in range(5):  # 区间控制、框下标控制
                    if self.SILO_INTERVAL[j] < self.ball_list[k][i][0] <= self.SILO_INTERVAL[j + 1]:
                        for k_ in range(3):  # 层数控制
                            if self.dec_buf[k]['delta'][j] == 0:
                                break  # 如果摄像头看到的球数量和上次看到的球数量没有区别，说明敌方在这个过程中没有放球
                            elif self.dec_buf[k]['delta'][j] > 0:  # 如果摄像头看到的球数量大于上次看到的球数量，则表明敌方在这个过程中放了球
                                if self.silo_state[j][k_] == 0:
                                    self.silo_state[j][k_] = 1 if k == self.SIDE else -1  # 将这个位置标对应颜色球
                                    self.dec_buf[k]['delta'][j] -= 1  # 将摄像头看到的和实际记录的差距-1，这样处理即使敌方在这个过程中放了两个球也能都被记录成功
                            else:  # (delta<0) 如果摄像头看到的数量少于上次记录的球数量，则说明球长脚从框里跑出去了，建议报警
                                print("ERROR! 存在球从框中移出！")
        print(f'silo_state: {self.silo_state}')

    def detect_sort_lidar(self, ros_msg_handler, ros_msg_sender):  # 雷达一步到位
        rm = ros_msg_handler
        while True:
            silo_inside_cnt = np.array(rm.read_recv('s'))
            if len(silo_inside_cnt) != 0:
                break
        ros_msg_sender.send('0,0,0,0')
        # 遍历 silo_my，将 silo_state 中的对应列的前n行赋值为1
        for i in range(len(self.silo_our_side_cnt)):
            self.silo_state[i, :self.silo_our_side_cnt[i]] = 1  # 使用与silo_my[i]长度相同的切片进行赋值

        # 遍历 silo_cnt，如果 silo_cnt 中的值大于 silo_my 中的对应值，
        # 那么在 silo_state 中的对应列的下一行赋值为-1
        for i in range(len(silo_inside_cnt)):
            if silo_inside_cnt[i] > self.silo_our_side_cnt[i]:
                self.silo_state[i, self.silo_our_side_cnt[i]:silo_inside_cnt[i]] = -1

        print(f'detect: {silo_inside_cnt}, our_side: {self.silo_our_side_cnt}, silo_state: {self.silo_state}')

    def decide(self):  # Note: 决策，用已经标注好阵营的state进行判断（与颜色无关）
        # 处理框内有两球的情况
        for i in range(5):
            for j in range(3):
                if self.silo_state[i][j] != 0:  # 第i个球框中有球(无论颜色，因为红球在上蓝球在下和蓝球在上红球在下本质上没有区别)
                    self.dec_buf['tmp'][i] += 1  # 第i个球框的球的层数+1
                    if self.silo_state[i][j] == -1:  # 是敌方的球
                        self.dec_buf['value'][i] += 2  # 第i个球框的决策权重增加1.5
                        # TODO: 注意修改一我一敌的情况
                    elif self.silo_state[i][j] == 1:  # 是我方的球
                        self.dec_buf['value'][i] += 1.5  # 第i个球框的决策权重增加2

        for i in range(5):
            if self.dec_buf['tmp'][i] == 2:  # 存在已经放了两个球的球框
                if self.dec_buf['value'][i] >= self.max_val:  # 选出多个满足条件的球框中最大决策权重的球框(即选出我方球数量多的那个球框)
                    self.max_val = self.dec_buf['value'][i]
        for i in range(5):
            if self.max_val == self.dec_buf['value'][i]:
                self.decision = i  # 决策处理，放到第i个球框中
                self.layers = -1  # 表示已经决策完成

        # TODO: 可以用match()函数
        if self.layers == 1:  # 不存在已经放了两个球的球框
            self.layers = 2  # 先将决策置为2，避免后续如果无法进入决策还要再做一次判断
            for i in range(5):
                if self.dec_buf['tmp'][i] == 0:  # 存在空框
                    self.decision = i  # 决策处理，放到第i个空的球框中
                    self.layers = -1  # 决策已经完成
            # 如果不存在空框的话，由于在进入layers==1的开头就已经将layers=2，所以可以不用再处理
        if self.layers == 2:  # 不存在已经放了两个球的球框，也不存在空框
            for i in range(5):
                if self.dec_buf['tmp'][i] == 1:  # 存在放了一个球的球框
                    for j in range(3):
                        if self.silo_state[i][j] == 1:  # 找第一层是我方球的球框
                            self.decision = i  # 决策处理，将球放到第i个第一层是我方球的球框中
                            self.layers = -1  # 决策已经完成
        if self.layers == -1:  # 决策已经完成
            # cv2.destroyAllWindows()
            pass
        #     self.reset()  # 进行清空重置

    def reset(self):
        # 清空ball_list各元素
        self.ball_list = {'red_ball': [], 'blue_ball': []}
        self.silo_state = np.array([[0, 0, 0] for _ in range(5)])

        self.max_val = 1
        self.decision = -1
        self.layers = 1
        self.dec_buf = {  # decision buffer
            'red_ball': {k: [0, 0, 0, 0, 0] for k in ['num', 'silo', 'delta']},
            'blue_ball': {k: [0, 0, 0, 0, 0] for k in ['num', 'silo', 'delta']},
            'value': [0, 0, 0, 0, 0],
            'tmp': [0, 0, 0, 0, 0]
        }

    def make_decision(self, image, ros_msg_recver, ros_msg_sender):
        self.reset()  # 进行清空重置
        self.detect(image)  # 内部有图像展示语句
        self.sort()
        # odom = ros_msg_recver.read_recv('o')
        # ros_msg_sender.send(f'1,{odom[0]},{odom[1]},{odom[-1]}')  # Send Command to Ros
        # time.sleep(0.5)  # Wait for ros，an important parameter
        # self.detect_sort_lidar(ros_msg_recver, ros_msg_sender)
        self.decide()  # 内部有关闭图像语句

        if self.decision != -1:
            self.silo_our_side_cnt[self.decision] += 1  # 记录我方放置的球
        return self.decision

    def make_retry_decision(self, image):
        self.reset()  # 进行清空重置
        self.detect(image)  # 内部有图像展示语句
        self.sort()
        self.decide()  # 内部有关闭图像语句
        self.retry_record_our_side()

        if self.decision != -1:
            self.silo_our_side_cnt[self.decision] += 1  # 记录我方放置的球

        return self.decision

    def YXS(self):
        SILO_YXS = [2, 1, 3, 0, 4]

        self.decision = SILO_YXS[self.YXS_cnt//3]

        self.YXS_cnt += 1
        return self.decision

    @staticmethod
    def arrive_dec_pos(odom_val):  # 检查是否到达范围
        arrived = False
        # if -0.85 < odom_val[1] < -0.55:  # Y
        if -1.1 < odom_val[1] < -0.9:  # Y
            if -0.2 < odom_val[0] < 0.2:  # X
                # if -0.05 < odom_val[2]:  # Z
                if abs(odom_val[-1]) <= 5:  # YAW
                    arrived = True
        return arrived

