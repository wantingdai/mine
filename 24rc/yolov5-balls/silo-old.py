import pyrealsense2 as rs
import onnxruntime as ort
import numpy as np
import cv2
from apply_on_R2 import getAvailableActions, decide_action, isEndGame, loadPolicy
# from GetBall import send_data_def
from general import plot_one_box, infer_img
import rc_utils

name = 'v5l-med_scratch-silo.onnx'  # 模型名称
net = ort.InferenceSession(name)  # 读取模型
dic_labels = {0: 'red_ball', 1: 'blue_ball', 2: 'silo'}  # 标签
model_h, model_w = 640, 640  # 模型参数
RedBall_x, RedBall_y, BlueBall_x, BlueBall_y, max_x, decision, layers = 0, 0, 0, 0, 1, 0, 0
RedBall_list, BlueBall_list = [], []
RedBall_x_values, RedBall_y_values, BlueBall_x_values, BlueBall_y_values = [], [], [], []
start_numpy = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
num_redball, num_blueball, silo_blueball, silo_redball, red_d, blue_d, value, temp = [], [], [], [], [], [], [], []

'''
以下是无用变量：
silo_x, silo_y, silo_list, 
'''

# -------------------------------- #
# 1表示我方颜色球，-1表示敌方颜色球
# 以下是分别就持红球和持蓝球进行讨论

"""
@ Red_Silo各个参数注解：
    1、第一行和第二行的global的各个参数都是用于处理视觉看到的球，为了将视觉看到的球变成二维数组的形式
    2、第三行的global的各个参数是用于进行决策。其中value是作为权重进行处理
       举例：第一个球框中有两个红球，第二个球框中有两个蓝球，那么最终决策会优先选择将红球放到第二个球框中，避免敌方大胜
       temp记录第i个球框的球的个数(不管红球还是蓝球)，max_x用于记录上文最大的value的值，用于比对找到第i个球框
       decision表示决策的结果，有且仅有0, 1, 2, 3, 4五个数字，我们绝不坐以待毙！我们把命运掌握在自己手里！
       layers用于进行决策的层数的处理，分为三个决策层数，分别对照三个情况，默认layers的值为1
       情况1: 存在有已经放好了两层的球的球框，此时优先级最高，因此除非已经决策完成，否则都会优先执行该步骤，当成功对有两层球的球框进行封顶之后，layers会变为-1，决策完成
       情况2( if layers == 1 ): 没有放了两层球的球框，寻找空框，当找到空框后，layers变为-1，决策完成，否则，layers会变为2，进入第三种情况
       情况3( if layers == 2 ): 找第一层是我方球的球框放球，放完球，决策完成，layers变为-1
"""


def Red_Silo(image, silo_model):
    global start_numpy, RedBall_x_values, RedBall_y_values, BlueBall_x_values, BlueBall_y_values, RedBall_list, BlueBall_list, RedBall_x, RedBall_y, BlueBall_x, BlueBall_y
    global num_redball, num_blueball, silo_redball, silo_blueball, red_d, blue_d
    global value, temp, max_x, decision, layers
    max_x, decision, layers = 0, 0, 1
    num_redball, num_blueball, silo_blueball, \
        silo_redball, red_d, blue_d, value, \
        temp = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], \
        [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]

    det_boxes, scores, ids = infer_img(image, net, model_h, model_w, 0.8, 0.2)  # 0.8表示极大值抑制的参数，0.2表示置信度的参数
    for box, score, id in zip(det_boxes, scores, ids):  # 在处理过的图像上面进行操作
        label = '%s' % (dic_labels[id])
        if label == "red_ball":
            RedBall_x, RedBall_y = plot_one_box(box.astype(np.int16), image, color=(255, 0, 0), label=label,
                                                line_thickness=None)
            RedBall_x_values.append(RedBall_x)
            RedBall_y_values.append(RedBall_y)
            RedBall_list = np.array(list(zip(RedBall_x_values, RedBall_y_values)))  # 红球的数组
        if label == "blue_ball":
            BlueBall_x, BlueBall_y = plot_one_box(box.astype(np.int16), image, color=(0, 0, 255), label=label,
                                                  line_thickness=None)
            BlueBall_x_values.append(BlueBall_x)
            BlueBall_y_values.append(BlueBall_y)
            BlueBall_list = np.array(list(zip(BlueBall_x_values, BlueBall_y_values)))  # 蓝球的数组

    # ------------------------------------- #
    # 由于红球和蓝球的个数并不相同，因此红球数组的长度和蓝球的数组长度也不同，因此需要分开记录 #
    for i in range(len(RedBall_list)):
        if 0 < RedBall_list[i][0] <= 125:
            num_redball[0] += 1  # 统计识别出来的在第0个框的红球有几个，如果不做统计，则下次再看时会因为有红球在第0个框的范围而将红球在第0个框的数量增多
        if 125 < RedBall_list[i][0] <= 250:
            num_redball[1] += 1
        if 250 < RedBall_list[i][0] <= 375:
            num_redball[2] += 1
        if 375 < RedBall_list[i][0] <= 500:
            num_redball[3] += 1
        if 500 < RedBall_list[i][0] <= 625:
            num_redball[4] += 1

    for i in range(len(BlueBall_list)):
        if 0 < BlueBall_list[i][0] <= 125:
            num_blueball[0] += 1  # 统计识别出来的在第0个框的蓝球有几个，如果不做统计，则下次再看时会因为有蓝球在第0个框的范围而将蓝球在第0个框的数量增多
        if 125 < BlueBall_list[i][0] <= 250:
            num_blueball[1] += 1
        if 250 < BlueBall_list[i][0] <= 375:
            num_blueball[2] += 1
        if 375 < BlueBall_list[i][0] <= 500:
            num_blueball[3] += 1
        if 500 < BlueBall_list[i][0] <= 625:
            num_blueball[4] += 1

    for i in range(0, 5):
        for j in range(0, 3):
            if start_numpy[i][j] == 1:
                silo_redball[i] += 1  # 统计五个球框中的各个球框的红球数量
            if start_numpy[i][j] == -1:
                silo_blueball[i] += 1  # 统计五个球框中的各个球框的蓝球数量

    for i in range(0, 5):
        red_d[i] = num_redball[i] - silo_redball[i]  # 判断每个球框中，摄像头看到的红球数量和上一次记录的红球数量进行求差
        blue_d[i] = num_blueball[i] - silo_blueball[i]  # 判断每个球框中，摄像头看到的蓝球数量和上一次记录的蓝球数量进行求差

    """
    @ 以下是对球框中的球进行数据处理，使其变成二维数组的形式进行决策，注释以第一个球框为例
    """
    for i in range(len(BlueBall_list)):
        if 0 < BlueBall_list[i][0] <= 125:
            for k in range(0, 3):
                if blue_d[0] == 0:  # 如果摄像头看到的蓝球数量和上次看到的蓝球数量没有区别，说明敌方在这个过程中没有放蓝球
                    break
                elif blue_d[0] > 0:  # 如果摄像头看到的蓝球数量大于上次看到的蓝球数量，则表明敌方在这个过程中放了蓝球
                    if start_numpy[0][k] == 0:
                        start_numpy[0][k] = -1  # 将这个位置标为蓝球
                        blue_d[0] -= 1  # 将摄像头看到的和实际记录的差距-1，这样处理即使敌方在这个过程中放了两个球也能都被记录成功
                elif blue_d[0] < 0:  # 如果摄像头看到的数量少于上次记录的蓝球数量，则说明蓝球长脚从框里跑出去了，建议报警
                    print("ERROR!")

        if 125 < BlueBall_list[i][0] <= 250:
            for k in range(0, 3):
                if blue_d[1] == 0:
                    break
                elif blue_d[1] > 0:
                    if start_numpy[1][k] == 0:
                        start_numpy[1][k] = -1
                        blue_d[1] -= 1
                elif blue_d[1] < 0:
                    print("ERROR!")

        if 250 < BlueBall_list[i][0] <= 375:
            for k in range(0, 3):
                if blue_d[2] == 0:
                    break
                elif blue_d[2] > 0:
                    if start_numpy[2][k] == 0:
                        start_numpy[2][k] = -1
                        blue_d[2] -= 1
                elif blue_d[2] < 0:
                    print("ERROR!")

        if 375 < BlueBall_list[i][0] <= 500:
            for k in range(0, 3):
                if blue_d[3] == 0:
                    break
                elif blue_d[3] > 0:
                    if start_numpy[3][k] == 0:
                        start_numpy[3][k] = -1
                        blue_d[3] -= 1
                elif blue_d[3] < 0:
                    print("ERROR!")

        if 500 < BlueBall_list[i][0] <= 625:
            for k in range(0, 3):
                if blue_d[4] == 0:
                    break
                elif blue_d[4] > 0:
                    if start_numpy[4][k] == 0:
                        start_numpy[4][k] = -1
                        blue_d[4] -= 1
                elif blue_d[4] < 0:
                    print("ERROR!")

    # --------------------------------------- #
    # 此处为红球部分，避免R2放完球之后需要对start_numpy进行处理，并把start_numpy再从main.py中传进来 #
    for i in range(len(RedBall_list)):
        if 0 < RedBall_list[i][0] <= 125:
            for k in range(0, 3):
                if red_d[0] == 0:
                    break  # 退出
                if red_d[0] > 0:
                    if start_numpy[0][k] == 0:
                        start_numpy[0][k] = 1
                        red_d[0] -= 1
                if red_d[0] < 0:
                    print("ERROR!")

        if 125 < RedBall_list[i][0] <= 250:
            for k in range(0, 3):
                if red_d[1] == 0:
                    break
                if red_d[1] > 0:
                    if start_numpy[1][k] == 0:
                        start_numpy[1][k] = 1
                        red_d[1] -= 1
                if red_d[1] < 0:
                    print("ERROR!")

        if 250 < RedBall_list[i][0] <= 375:
            for k in range(0, 3):
                if red_d[2] == 0:
                    break
                elif red_d[2] > 0:
                    if start_numpy[2][k] == 0:
                        start_numpy[2][k] = 1
                        red_d[2] -= 1
                elif red_d[2] < 0:
                    print("ERROR!")

        if 375 < RedBall_list[i][0] <= 500:
            for k in range(0, 3):
                if red_d[3] == 0:
                    break
                elif red_d[3] > 0:
                    if start_numpy[3][k] == 0:
                        start_numpy[3][k] = 1
                        red_d[3] -= 1
                elif red_d[3] < 0:
                    print("ERROR!")

        if 500 < RedBall_list[i][0] <= 625:
            for k in range(0, 3):
                if red_d[4] == 0:
                    break
                elif red_d[4] > 0:
                    if start_numpy[4][k] == 0:
                        start_numpy[4][k] = 1
                        red_d[4] -= 1
                elif red_d[4] < 0:
                    print("ERROR!")

    print(start_numpy)
    # print(RedBall_list)  # 打印红球的识别
    # print(BlueBall_list)  # 打印蓝球的识别

    # ------------------------------ #
    # 此处为决策部分 #
    for i in range(0, 5):
        for j in range(0, 3):
            if start_numpy[i][j] != 0:  # 第i个球框中有球(无论颜色，因为红球在上蓝球在下和蓝球在上红球在下本质上没有区别)
                temp[i] += 1  # 第i个球框的球的层数+1
                if start_numpy[i][j] == -1:  # 是敌方的球
                    value[i] += 2  # 第i个球框的决策权重增加2
                elif start_numpy[i][j] == 1:  # 是我方的球
                    value[i] += 1.5  # 第i个球框的决策权重增加1.5

    for i in range(0, 5):
        if temp[i] == 2:  # 存在已经放了两个球的球框
            if value[i] >= max_x:  # 选出多个满足条件的球框中最大决策权重的球框(即选出蓝球数量多的那个球框)
                max_x = value[i]
    for i in range(0, 5):
        if max_x == value[i]:
            decision = i  # 决策处理，放到第i个球框中
            layers = -1  # 表示已经决策完成

    if layers == 1:  # 不存在已经放了两个球的球框
        layers = 2  # 先将决策置为2，避免后续如果无法进入决策还要再做一次判断
        for i in range(0, 5):
            if temp[i] == 0:  # 存在空框
                decision = i  # 决策处理，放到第i个空的球框中
                layers = -1  # 决策已经完成
            # 如果不存在空框的话，由于在进入layers==1的开头就已经将layers=2，所以可以不用再处理

    if layers == 2:  # 不存在已经放了两个球的球框，也不存在空框
        for i in range(0, 5):
             if temp[i] == 1:  # 存在放了一个球的球框
                for j in range(0, 3):
                    if start_numpy[i][j] == 1:  # 找第一层是我方球的球框
                        decision = i  # 决策处理，将球放到第i个第一层是我方球的球框中
                        layers = -1  # 决策已经完成

    if layers == -1:  # 决策已经完成
        print(decision)  # 打印出决策
        # return decision  # 反馈给main.py

        # 进行清空重置
        value, temp = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
        layers, decision = 1, -1

        RedBall_x_values.clear()
        RedBall_y_values.clear()
        RedBall_list = []

        BlueBall_x_values.clear()
        BlueBall_y_values.clear()
        BlueBall_list = []

        start_numpy = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
        silo_redball, silo_blueball, num_blueball, num_redball = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]

    cv2.imshow('img', image)  # 图像展示
    if cv2.waitKey(1) & 0xff == (ord('q')):
        return


def Blue_Silo(image, silo_model):
    global start_numpy, RedBall_x_values, RedBall_y_values, BlueBall_x_values, BlueBall_y_values, silo_list, RedBall_list, BlueBall_list, silo_x, silo_y, RedBall_x, RedBall_y, BlueBall_x, BlueBall_y
    global num_redball, num_blueball, silo_redball, silo_blueball, red_d, blue_d
    num_redball, num_blueball, silo_blueball, silo_redball, red_d, blue_d = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0,
                                                                                                               0, 0], [
        0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]
    det_boxes, scores, ids = infer_img(image, net, model_h, model_w, 0.8, 0.2)  # 0.8表示极大值抑制的参数，0.2表示置信度的参数
    for box, score, id in zip(det_boxes, scores, ids):  # 在处理过的图像上面进行操作
        label = '%s' % (dic_labels[id])
        if label == "red_ball":
            RedBall_x, RedBall_y = plot_one_box(box.astype(np.int16), image, color=(0, 255, 0), label=label,
                                                line_thickness=None)
            RedBall_x_values.append(RedBall_x)
            RedBall_y_values.append(RedBall_y)
            RedBall_list = np.array(list(zip(RedBall_x_values, RedBall_y_values)))
        if label == "blue_ball":
            BlueBall_x, BlueBall_y = plot_one_box(box.astype(np.int16), image, color=(0, 255, 0), label=label,
                                                  line_thickness=None)
            BlueBall_x_values.append(BlueBall_x)
            BlueBall_y_values.append(BlueBall_y)
            BlueBall_list = np.array(list(zip(BlueBall_x_values, BlueBall_y_values)))

    for i in range(len(RedBall_list)):
        if 0 < RedBall_list[i][0] <= 125:
            num_redball[0] += 1  # 统计识别出来的在第0个框的红球有几个，如果不做统计，则下次再看时会因为有红球在第0个框的范围而将红球在第0个框的数量增多
        if 125 < RedBall_list[i][0] <= 250:
            num_redball[1] += 1
        if 250 < RedBall_list[i][0] <= 375:
            num_redball[2] += 1
        if 375 < RedBall_list[i][0] <= 500:
            num_redball[3] += 1
        if 500 < RedBall_list[i][0] <= 625:
            num_redball[4] += 1

    for i in range(len(BlueBall_list)):
        if 0 < BlueBall_list[i][0] <= 125:
            num_blueball[0] += 1  # 统计识别出来的在第0个框的红球有几个，如果不做统计，则下次再看时会因为有红球在第0个框的范围而将红球在第0个框的数量增多
        if 125 < BlueBall_list[i][0] <= 250:
            num_blueball[1] += 1
        if 250 < BlueBall_list[i][0] <= 375:
            num_blueball[2] += 1
        if 375 < BlueBall_list[i][0] <= 500:
            num_blueball[3] += 1
        if 500 < BlueBall_list[i][0] <= 625:
            num_blueball[4] += 1

    for i in range(0, 5):
        for j in range(0, 3):
            if start_numpy[i][j] == -1:
                silo_redball[i] += 1  # 统计五个球框中的各个球框的红球数量
            if start_numpy[i][j] == 1:
                silo_blueball[i] += 1

    for i in range(0, 5):
        red_d[i] = num_redball[i] - silo_redball[i]
        blue_d[i] = num_blueball[i] - silo_blueball[i]

    for i in range(len(RedBall_list)):
        if 0 < RedBall_list[i][0] <= 125:
            for k in range(0, 3):
                if red_d[0] == 0:
                    break
                if red_d[0] > 0:
                    if start_numpy[0][k] == 0:
                        start_numpy[0][k] = -1
                        red_d[0] -= 1
                if red_d[0] < 0:
                    print("ERROR!")

        if 125 < RedBall_list[i][0] <= 250:
            for k in range(0, 3):
                if red_d[1] == 0:
                    break
                if red_d[1] > 0:
                    if start_numpy[1][k] == 0:
                        start_numpy[1][k] = -1
                        red_d[1] -= 1
                if red_d[1] < 0:
                    print("ERROR!")

        if 250 < RedBall_list[i][0] <= 375:
            for k in range(0, 3):
                if red_d[2] == 0:
                    break
                elif red_d[2] > 0:
                    if start_numpy[2][k] == 0:
                        start_numpy[2][k] = -1
                        red_d[2] -= 1
                elif red_d[2] < 0:
                    print("ERROR!")

        if 375 < RedBall_list[i][0] <= 500:
            for k in range(0, 3):
                if red_d[3] == 0:
                    break
                elif red_d[3] > 0:
                    if start_numpy[3][k] == 0:
                        start_numpy[3][k] = -1
                        red_d[3] -= 1
                elif red_d[3] < 0:
                    print("ERROR!")

        if 500 < RedBall_list[i][0] <= 625:
            for k in range(0, 3):
                if red_d[4] == 0:
                    break
                elif red_d[4] > 0:
                    if start_numpy[4][k] == 0:
                        start_numpy[4][k] = -1
                        red_d[4] -= 1
                elif red_d[4] < 0:
                    print("ERROR!")

    for i in range(len(BlueBall_list)):
        if 0 < BlueBall_list[i][0] <= 125:
            for k in range(0, 3):
                if blue_d[0] == 0:
                    break
                elif blue_d[0] > 0:
                    if start_numpy[0][k] == 0:
                        start_numpy[0][k] = 1
                        blue_d[0] -= 1
                elif blue_d[0] < 0:
                    print("ERROR!")

        if 125 < BlueBall_list[i][0] <= 250:
            for k in range(0, 3):
                if blue_d[1] == 0:
                    break
                elif blue_d[1] > 0:
                    if start_numpy[1][k] == 0:
                        start_numpy[1][k] = 1
                        blue_d[1] -= 1
                elif blue_d[1] < 0:
                    print("ERROR!")

        if 250 < BlueBall_list[i][0] <= 375:
            for k in range(0, 3):
                if blue_d[2] == 0:
                    break
                elif blue_d[2] > 0:
                    if start_numpy[2][k] == 0:
                        start_numpy[2][k] = 1
                        blue_d[2] -= 1
                elif blue_d[2] < 0:
                    print("ERROR!")

        if 375 < BlueBall_list[i][0] <= 500:
            for k in range(0, 3):
                if blue_d[3] == 0:
                    break
                elif blue_d[3] > 0:
                    if start_numpy[3][k] == 0:
                        start_numpy[3][k] = 1
                        blue_d[3] -= 1
                elif blue_d[3] < 0:
                    print("ERROR!")

        if 500 < BlueBall_list[i][0] <= 625:
            for k in range(0, 3):
                if blue_d[4] == 0:
                    break
                elif blue_d[4] > 0:
                    if start_numpy[4][k] == 0:
                        start_numpy[4][k] = 1
                        blue_d[4] -= 1
                elif blue_d[4] < 0:
                    print("ERROR!")

    print(start_numpy)
    # print(RedBall_list)
    # print(BlueBall_list)
    obs = start_numpy
    # print(obs)
    isEnd = isEndGame(obs)
    if isEnd is None:
        ava_acts = getAvailableActions(obs)
        decision = decide_action(obs, silo_model, ava_acts)
        # print(decision)
        return decision
        # send_data(decision)
    elif isEnd == 'f':
        print('谷仓已满，未大胜')
    elif isEnd == 'm':
        print('大胜！')
        print('播放结算动画！')
    elif isEnd == 'e':
        print('DAMN!!!敌方大胜')

    # 进行清空重置
    RedBall_x_values.clear()
    RedBall_y_values.clear()
    RedBall_list = []

    BlueBall_x_values.clear()
    BlueBall_y_values.clear()
    BlueBall_list = []

    start_numpy = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
    silo_redball, silo_blueball, num_blueball, num_redball = [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]

    cv2.imshow('img', image)  # 图像展示
    if cv2.waitKey(1) & 0xff == (ord('q')):
        return


# if __name__ == '__main__':
#     start_numpy = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
#     pipeline, align = rc_utils.init_realsense()
#
#     while True:
#         frames = pipeline.wait_for_frames()
#         aligned_frames = align.process(frames)
#
#         # Get aligned frames
#         aligned_depth_frame = aligned_frames.get_depth_frame()
#         color_frame = aligned_frames.get_color_frame()
#
#         # Validate that both frames are valid
#         if not aligned_depth_frame or not color_frame:
#             continue
#
#         # Convert images to numpy arrays
#         depth_image = np.asanyarray(aligned_depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())
#
#         # 滤除不满足深度要求的点（以黑色填充）
#         # 设置深度阈值 (单位为米)
#         depth_min = 2.0
#         depth_max = 2.8
#
#         # 将深度图像转换为有效范围内的掩码
#         mask = np.logical_and(depth_image > depth_min * 1000, depth_image < depth_max * 1000)
#         # 创建一个结构元素,用于开运算
#         kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
#
#         # 对mask进行开运算
#         mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
#
#         # 将不满足深度要求的点在彩色图像上变为黑色
#         color_image[np.logical_not(mask)] = [0, 0, 0]
#         # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#
#         # Stack both images horizontally
#         images = color_image
#         model = loadPolicy()
#         Red_Silo(images, model)
#         # cv2.imshow('img', img)
#
#     pipeline.stop()
#     cv2.destroyAllWindows()
