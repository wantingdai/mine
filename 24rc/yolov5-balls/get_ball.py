import pyrealsense2 as rs
import math
import onnxruntime as ort
from general import plot_one_box, infer_img, detect, scale_coords, plot_one_box_new
import numpy as np
import cv2
import ros_comm
# import serial
# import serial.tools.list_ports
from sklearn.cluster import KMeans
# import rc_utils
# from collections import deque
import time
from static_ball_filter_eg import filter_static_objects



class GetBall:
    def __init__(self, SIDE: str):
        self.model_name = 'start_m.onnx'  # 模型名称  # red
        # self.model_name = 'start_blue_add.onnx'  # blue
        self.model_name1 = 'mch_arm_n.onnx'  # 模型名称
        self.net = ort.InferenceSession(self.model_name)  # 读取模型
        self.net_catch = ort.InferenceSession(self.model_name1)
        self.dict_labels = {0: 'red_ball', 1: 'purple_ball', 2: 'blue_ball'}  # 标签
        self.MODEL_H, self.MODEL_W = 640, 640  # 模型参数
        self.shape = (self.MODEL_W, self.MODEL_H)

        self.position_list = []
        self.camera_angle = 0
        self.sent = False  # True: 已发送过一次
        self.adjust_x, self.adjust_y = 0, 0
        self.ball_loc_list = []  # 用于滤波
        # self.last_mode = None  # 用于检测模式切换动作，控制队列清空
        # self.FILTER_LEN = 3
        # self.filter_queue = deque(maxlen=self.FILTER_LEN)  # 均值滤波（单个球能用，多个不行）
        # self.ZONE3_X_LIM = 200
        # self.ZONE3_Y_LIM = 320
        self.ZONE3_X_LIM = 200
        self.ZONE3_Y_LIM = 450
        self.output_x = 0
        self.output_y = 0
        # self.CATCH_ROI = [80, 560, 30, 350]  # x_min, x_max, y_min, y_max  单位:pixel
        self.CATCH_ROI = [78, 558, 30, 330]  # 6.21 gai dong
        self.in_roi = False
        self.send_pos = None
        self.distor_comp = 1

        self.distance_x = [150, 75, 0, -75, -150]  # 五个球框前的点和START点位的x的差距
        # self.distance_y = [82, 82, 70, 82, 82]  # 五个球框前的点和START点位的y的差距 blue
        self.distance_y = [82, 82, 82, 82, 82]  # 五个球框前的点和START点位的y的差距 red
        # self.distance_y = [70, 70, 70, 70, 70]
        self.send_pos_x = None
        self.send_pos_y = None
        self.real_send_pos_x = None
        self.real_send_pos_y = None
        self.flag = None
        self.use = False
        assert SIDE == 'r' or SIDE == 'b'  # 确保输入正确
        self.SIDE = 'red_ball' if SIDE == 'r' else 'blue_ball'

        # 滤波变量
        self.filter_input = []
        self.frame_cnt = 0

    def catch(self, image, depth_frame, ser, mode: str):
        pass

    def find_and_get(self, image, depth_frame, ser):  # mode = 'f'(find) or 'c'(catch) 用于控制是机械臂还是底盘
        self.in_roi = False
        # self.adjust_x, self.adjust_y = -2.7, 22.3
        # 球在机械臂左边
        # self.adjust_x, self.adjust_y = -2.0, 20.5  # TODO
        # 实际中，机械臂和摄像头之间的误差为+3, +24
        self.adjust_x, self.adjust_y = -1.0, 22
        self.camera_angle = 0

        # det_boxes, scores, ids = infer_img(image, self.net_catch, self.MODEL_H, self.MODEL_W,
        #                                    thred_nms=0.85, thred_cond=0.3)
        # for box, score, label_id in zip(det_boxes, scores, ids):  # 在处理过的图像上面进行操作
        #     label = '%s' % (self.dict_labels[label_id])  # 标签，利用标签得到对红球的判断
        #     if label == self.SIDE:
        #         # self.in_roi = True
        #         x, y = plot_one_box(box.astype(np.int16), image,
        #                             color=(0, 0, 255) if self.SIDE == 'red_ball' else (255, 0, 0),
        #                             label=label, line_thickness=None)
        #         # if x != 0 and y != 0:3
        #         # 将每个识别出来的识别框的中心点放入坐标列表中，用于给深度摄像头判断距离
        #         if self.CATCH_ROI[0] <= x <= self.CATCH_ROI[1] and self.CATCH_ROI[2] <= y <= self.CATCH_ROI[3]:
        #             self.position_list.append((x, y))

        img, pred_boxes, pred_confes, pred_classes = detect(image, self.net_catch, self.MODEL_H, self.MODEL_W, 0.45, 0.5)
        if len(pred_boxes) > 0:
            for i, _ in enumerate(pred_boxes):
                label = '%s' % (self.dict_labels[pred_classes[i]])  # 标签，利用标签得到对红球的判断
                if label == self.SIDE:
                    # self.in_roi = True
                    x, y = plot_one_box_new(image, img, self.shape, pred_boxes[i], pred_classes[i], pred_confes[i])
                    # if x != 0 and y != 0:3
                    # 将每个识别出来的识别框的中心点放入坐标列表中，用于给深度摄像头判断距离
                    if self.CATCH_ROI[0] <= x <= self.CATCH_ROI[1] and self.CATCH_ROI[2] <= y <= self.CATCH_ROI[3]:
                        self.position_list.append((x, y))

        if len(self.position_list) != 0:
            # 得到距离摄像头最近的球的中心点坐标
            min_x, min_y = Get_min_distance(self.position_list)
            # print(min_x, min_y)
            self.position_list.clear()
            # # 限制ROI
            # if self.CATCH_ROI[0] > min_x or self.CATCH_ROI[1] < min_x:
            #     print(f"X值超域, x={min_x}")
            #     self.in_roi = False
            #     return False
            # elif self.CATCH_ROI[2] > min_y or self.CATCH_ROI[3] < min_y:
            #     print(f"Y值超域, y={min_y}")
            #     self.in_roi = False
            #     return False
            # elif self.CATCH_ROI[0] < min_x < self.CATCH_ROI[1] and self.CATCH_ROI[2] < min_y < self.CATCH_ROI[3]:
            if True:  # Temp
                cv2.rectangle(image, (self.CATCH_ROI[0], self.CATCH_ROI[2]), (self.CATCH_ROI[1], self.CATCH_ROI[3]),
                              (203, 192, 255), 1)
                cv2.imshow('img', image)
                # self.in_roi = True

                # 利用最近的球的中心点坐标，得到距离，这样可以有效避免球在深度摄像头边缘被检测，导致深度检测为0的结果
                distance = depth_frame.get_distance(min_x, min_y) * 100 + 9.5  # 得到的距离单位是m，需要进行单位转换，加上球半径得到距离球心的距离

                # points是相机坐标系中的坐标，其中第一个参数表示的是深度摄像头的内参矩阵
                points = rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics,
                                                         [min_x, min_y], distance)
                # 相机坐标系中的数据转换成为世界坐标系,camera_angle=0，垂直朝下看去
                relative_x = points[0] + self.adjust_x  # 此处数据根据机械组给出
                relative_y = points[2] * math.cos(Angles2Radians(90 - self.camera_angle)) - points[1] * math.sin(
                    Angles2Radians(90 - self.camera_angle)) + self.adjust_y  # 此处数据根据机械组给出

                self.ball_loc_list.append((relative_x, relative_y))

                if self.ball_loc_list:  # TODO: 就先这么写着
                    # self.output_x, self.output_y = min(self.ball_loc_list, key=lambda point: point[0]**2 + point[1] ** 2)
                    self.output_x, self.output_y = min(self.ball_loc_list, key=lambda point: abs(point[0] - 320))
                    if abs(self.output_x) >= 25:
                        self.output_y -= 2
                    elif 25 >= abs(self.output_x) >= 19.5:
                        self.output_y -= 2.5
                    # if self.output_x * self.distor_comp <= -16:
                    #     self.output_x = self.output_x - 5
                    #     self.output_y = self.output_y - 4
                    # elif self.output_x * self.distor_comp >= 16:
                    #     self.output_x = self.output_x + 5
                    #     self.output_y = self.output_y - 2
                    # print(self.output_x * self.distor_comp, self.output_y)
                    # if self.output_y >= 25:
                    #     self.output_x -= 0.2
                    # if 25 >= self.output_x >= 20:
                    #     self.output_x -= 1
                    #     self.output_y -= 0.5
                    # if -23 <= self.output_x <= -20:
                    #     self.output_x += 2.6
                    #     self.output_y -= 1.2
                    # elif self.output_x < -23:
                    #     self.output_x -= 1.6
                    #     self.output_y -= 3
                    ser.send_catch(self.output_x, self.output_y)
                    # time.sleep(0.5)  # Temp: 降低发送频率  # TODO： 补打数据集后即可删除
                    self.ball_loc_list.clear()
                    self.sent = False
                    self.ball_loc_list.clear()
                    return True

                # elif self.in_roi is False:  # ROI内没有可取球
                #     print('ROI内无可取球!')
                #     self.ball_loc_list.clear()
                #     return False

        else:
            print('ROI内无可取球!')
            self.ball_loc_list.clear()
            self.position_list.clear()
            return False

    def reset_find_and_get(self):
        self.in_roi = False

    def reset_yzy_idea(self):
        self.sent = False
        self.flag = None
        self.use = False
        self.send_pos_x = 0
        self.send_pos_y = 0
        self.real_send_pos_x = 0
        self.real_send_pos_y = 0

    def reset_dtd_idea(self):
        self.sent = False
        self.send_pos = None

    def dtd_idea_check(self, image, x_arrived:bool, WIDTH, ROI=(0, 360)):
        if not x_arrived:  # 未开始向下运动
            return True  # 保证在没有到达x轴的情况下，不会send interrupt
        # 注意摄像头是竖过来装的，用y值判断
        ball_in_area = False
        det_boxes, scores, ids = infer_img(image, self.net, self.MODEL_H, self.MODEL_W,
                                           thred_nms=0.5, thred_cond=0.2)
        for box, score, label_id in zip(det_boxes, scores, ids):  # 在处理过的图像上面进行操作
            label = '%s' % (self.dict_labels[label_id])  # 标签，利用标签得到对红球的判断

            # 找到了红球
            if label == self.SIDE:
                """
                @ 只有读取到红球并且此时没有其他球正在进行任务，才进行后续操作
                """
                x, y = plot_one_box(box.astype(np.int16), image,
                                    color=(0, 0, 255) if self.SIDE == 'red_ball' else (255, 0, 0),
                                    label=label, line_thickness=None)

                if ROI[0] <= y <= ROI[1]:
                    ball_in_area = True


        cv2.line(image, (0, ROI[0]), (WIDTH, ROI[0]), (203, 192, 255), 2)
        cv2.line(image, (0, ROI[1]), (WIDTH, ROI[1]), (203, 192, 255), 2)
        cv2.imshow('img', image)

        return ball_in_area

    def yzy_idea_static_filter(self, image, depth_frame, ser, decision, odom_val):
        if self.SIDE == 'blue_ball':
            # 创建变量
            self.adjust_x = 5.3  # 车往球右边，就让adjust_x往+方向调整，车往球左边，就让adjust_x往-方向调整
            self.camera_angle = 85
            self.adjust_y = 0

            static_balls = []
            self.use = False

            # det_boxes, scores, ids = infer_img(image, self.net, self.MODEL_H, self.MODEL_W,
            #                                    thred_nms=0.85, thred_cond=0.1)  # 此处进行修改，防止一直识别到其他东西认为红球
            # for box, score, label_id in zip(det_boxes, scores, ids):  # 在处理过的图像上面进行操作
            #     label = '%s' % (self.dict_labels[label_id])  # 标签，利用标签得到对红球的判断
            #
            #     # 找到了红球
            #     if label == self.SIDE:
            #         """
            #         @ 只有读取到红球并且此时没有其他球正在进行任务，才进行后续操作
            #         """
            #         x, y = plot_one_box(box.astype(np.int16), image,
            #                             color=(0, 0, 255) if self.SIDE == 'red_ball' else (255, 0, 0),
            #                             label=label, line_thickness=None)
            #         if x != 0 and y != 0:
            #             static_balls.append((x, y))
            if self.SIDE == 'red_ball':
                conf_thres = 0.5
            elif self.SIDE == 'blue_ball':
                conf_thres = 0.3
            img, pred_boxes, pred_confes, pred_classes = detect(image, self.net, self.MODEL_H, self.MODEL_W, 0.45,
                                                                conf_thres)
            if len(pred_boxes) > 0:
                for i, _ in enumerate(pred_boxes):
                    label = '%s' % (self.dict_labels[pred_classes[i]])  # 标签，利用标签得到对红球的判断
                    if label == self.SIDE:
                        # self.in_roi = True
                        x, y = plot_one_box_new(image, img, self.shape, pred_boxes[i], pred_classes[i], pred_confes[i])

                        if x != 0 and y != 0:
                            static_balls.append((x, y))

            cv2.imshow('START POINT', image)
            cv2.waitKey(1)

            self.frame_cnt += 1
            self.filter_input.append(static_balls)

            if self.frame_cnt < 2:
                return False

            static_objects = filter_static_objects(self.filter_input, 1000000000000000, 10000000000000)
            static_objects = [tuple(obj) for obj in static_objects]
            # print(f'filter_input: {self.filter_input}')
            # print(f'static_obj: {static_objects}')
            print(f"Number of static objects detected: {len(static_objects)}")
            # 滤波变量清零
            self.filter_input.clear()
            static_balls.clear()
            self.frame_cnt = 0

            if len(static_objects) != 0:
                # min_x, min_y = Get_min_depth(static_objects, depth_frame)
                min_x, min_y = Get_min_distance(static_objects)
                static_objects.clear()
                distance = depth_frame.get_distance(min_x, min_y) * 100 + 9.5  # 得到的距离单位是m，需要进行单位转换，加上球半径得到距离球心的距离

                # points是相机坐标系中的坐标，其中第一个参数表示的是深度摄像头的内参矩阵
                points = rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics,
                                                         [min_x, min_y], distance)
                # 相机坐标系中的数据转换成为世界坐标系,camera_angle=0，垂直朝下看去
                relative_x = (points[0] + self.adjust_x)  # 此处数据根据机械组给出
                relative_y = (points[2] * math.cos(Angles2Radians(90 - self.camera_angle)) - points[1] * math.sin(
                    Angles2Radians(90 - self.camera_angle)) + self.adjust_y)  # 此处数据根据机械组给出

                angle = odom_val[-1]
                rad = (math.pi / 180) * angle

                mid_relative_x = (relative_x - relative_y * math.tan(rad)) * math.cos(rad)
                mid_relative_y = relative_y / math.cos(rad) + (
                        relative_x - relative_y * math.tan(rad)) * math.sin(rad)

                relative_x = mid_relative_x  # relative_x对应的是START点位的坐标，real_relative_x对应的是球框前的真实坐标
                relative_y = mid_relative_y
                real_relative_x = mid_relative_x
                real_relative_y = mid_relative_y
                print("real_points(DIRECT):", real_relative_x, real_relative_y)

                if 0 <= decision <= 4:
                    self.use = True
                    relative_x = real_relative_x + self.distance_x[decision]  # 要将球框前的坐标转为对应的START点位的坐标
                    relative_y = real_relative_y - self.distance_y[decision]
                    print("relative_points(START):", relative_x, relative_y)
                    # if relative_y <= -self.ZONE3_Y_LIM - self.distance_y[decision]:
                    #     relative_y = -self.ZONE3_Y_LIM - self.distance_y[decision]
                    # if relative_x <= -self.ZONE3_X_LIM - self.distance_x[decision]:
                    #     relative_x = -self.ZONE3_X_LIM - self.distance_x[decision]
                    # elif relative_x >= self.ZONE3_X_LIM - self.distance_x[decision]:
                    #     relative_x = self.ZONE3_X_LIM - self.distance_x[decision]
                # else:  # No silo
                #     if relative_y <= -self.ZONE3_Y_LIM:
                #         relative_y = -self.ZONE3_Y_LIM
                #     if relative_x <= -self.ZONE3_X_LIM:
                #         relative_x = -self.ZONE3_X_LIM
                #     elif relative_x >= self.ZONE3_X_LIM:
                #         relative_x = self.ZONE3_X_LIM

                if relative_y <= -self.ZONE3_Y_LIM:
                    relative_y = -self.ZONE3_Y_LIM
                if relative_x <= -self.ZONE3_X_LIM:
                    relative_x = -self.ZONE3_X_LIM
                elif relative_x >= self.ZONE3_X_LIM:
                    relative_x = self.ZONE3_X_LIM

                # if relative_x <= -300 or relative_x >= 300:
                #     relative_x = 0
                #     print("Find Wrong Ball!")

                if not self.sent:
                    self.send_pos_x = -relative_x  # START点位的坐标用于判断什么时候进死角，什么时候转弯推墙
                    self.send_pos_y = -relative_y
                    self.real_send_pos_x = -real_relative_x  # 对应球框前的真实坐标
                    self.real_send_pos_y = -real_relative_y
                    # blue
                    self.send_pos_x += 13  # Temp

                    # red 左边不对
                    # if self.send_pos_x >= 120:  # Temp
                    #     self.send_pos_x += 5
                    if self.send_pos_y <= -410:
                        self.send_pos_y = -410
                        if self.send_pos_x <= -160:  # 球在右边死角
                            self.send_pos_x = -165
                            self.send_pos_y = -405
                            self.flag = 1  # 向右转45°
                        elif self.send_pos_x >= 155:  # 球在左边死角
                            self.send_pos_x = 160
                            self.send_pos_y = -400
                            self.flag = 2  # 向左转45°

                        elif -160 < self.send_pos_x < 155:
                            self.flag = None

                    elif self.send_pos_y > -410:
                        if self.send_pos_x <= -150:  # 球在右边墙
                            self.send_pos_y += 0
                            self.send_pos_x = -135
                            self.flag = 3  # 向右转90°

                        elif self.send_pos_x >= 160:  # 球在左边墙
                            # red
                            # self.send_pos_y = self.send_pos_y  # 不变
                            # blue
                            self.send_pos_y = self.send_pos_y - 8  # Temp
                            self.send_pos_x = 140
                            self.flag = 4  # 向左转90°

                        elif -150 < self.send_pos_x < 160:
                            self.flag = None

                    # 转化为球框前看球的坐标系
                    if 0 <= decision <= 4:
                        self.real_send_pos_x = self.send_pos_x + self.distance_x[decision]
                        self.real_send_pos_y = self.send_pos_y - self.distance_y[decision]

                    if self.use:  # 表示是球框前看球的点位
                        if self.real_send_pos_y <= -520:
                            self.real_send_pos_y = -520  # 最远给定的是-430，算上82的y值差，最远到512
                        if decision == 4:
                            if self.flag == 2:  # 球在左边死角的情况
                                self.real_send_pos_x = -21
                            elif self.flag == 4:  # 球在左边墙壁的情况
                                self.real_send_pos_x = -30
                                self.real_send_pos_y = self.real_send_pos_y + 6
                            # elif self.flag is None:
                            #     self.real_send_pos_x = self.real_send_pos_x + 20
                        elif decision == 3:
                            if self.flag == 2:  # 球在左边死角的情况
                                self.real_send_pos_x = 83
                            elif self.flag == 4:  # 球在左边墙壁的情况
                                self.real_send_pos_x = 75
                                self.real_send_pos_y = self.real_send_pos_y
                            elif self.flag == 1:  # 球在右边死角的情况
                                self.real_send_pos_x = -225
                        elif decision == 2:
                            if self.flag == 2:  # 左边死角
                                self.real_relative_x = 140
                                self.real_send_pos_y = -515
                            elif self.flag == 1:
                                self.real_send_pos_y -= 6
                            elif self.flag == 4:  # 左边墙
                                self.real_send_pos_x = 130
                                self.real_send_pos_y = self.real_send_pos_y - 3
                        elif decision == 1:
                            if self.flag == 1:  # 球在右边死角的情况
                                self.real_send_pos_x = -70
                            elif self.flag == 2:  # 球在左边死角的情况
                                self.real_send_pos_y -= 3
                                self.real_send_pos_x -= 3
                            elif self.flag == 3:  # 球在右边墙壁的情况
                                self.real_send_pos_x = -70
                                self.real_send_pos_y = self.real_send_pos_y - 8
                        elif decision == 0:
                            if self.flag == 1:  # 球在右边死角的情况
                                self.real_send_pos_x = 40
                            elif self.flag == 3:  # 球在右边墙壁的情况
                                self.real_send_pos_x = 30
                                self.real_send_pos_y = self.real_send_pos_y + 2
                    # 整体调整  此处的左右是相对于机器人的逆向的左右，是从silo向下看的
                    if self.flag == 4:  # 左墙
                        self.send_pos_x = self.send_pos_x - 20
                        self.real_send_pos_x = self.real_send_pos_x - 20
                    elif self.flag == 3:  # 右墙
                        self.send_pos_x = self.send_pos_x + 20
                        self.real_send_pos_x = self.real_send_pos_x + 20
                    elif self.flag == 2:  # 左角
                        self.send_pos_x = self.send_pos_x - 20
                        self.real_send_pos_x = self.real_send_pos_x - 20
                    elif self.flag == 1:  # 右角
                        self.send_pos_x = self.send_pos_x + 20
                        self.real_send_pos_x = self.real_send_pos_x + 20
                    else:
                        self.send_pos_x = self.send_pos_x - 10
                        self.real_send_pos_x = self.real_send_pos_x - 10
                        self.send_pos_y += 33
                        self.real_send_pos_y += 33
                    self.send_pos_x -= 10
                    self.real_send_pos_x -= 10
                    # self.send_pos_y -= 30
                    # self.real_send_pos_y -= 30
                    self.send_pos_y -= 20
                    self.real_send_pos_y -= 20

                if self.send_pos_x != 0 or self.send_pos_y != 0:
                    self.sent = True
                    if self.use:
                        print("SILO")
                        # for i in range(0, 5):
                        # print(self.real_send_pos_x, self.real_send_pos_y, self.flag)
                        print('START:', self.send_pos_x, self.send_pos_y, self.flag)
                        print('SILO: ', self.real_send_pos_x, self.real_send_pos_y, self.flag)
                        ser.send_yzy_idea(self.real_send_pos_x, self.real_send_pos_y, flag=self.flag)  # 组装发送格式
                    else:
                        print("NO SILO")
                        # for i in range(0, 5):
                        print('START:', self.send_pos_x, self.send_pos_y, self.flag)
                        ser.send_yzy_idea(self.send_pos_x, self.send_pos_y, flag=self.flag)
                elif self.send_pos_x and self.send_pos_y == 0:
                    print("Nothing")
            self.use = False
            self.flag = None
            self.real_send_pos_x = 0
            self.real_send_pos_y = 0
            self.send_pos_x = 0
            self.send_pos_y = 0

            return True

        elif self.SIDE == 'red_ball':
            # 创建变量
            self.adjust_x = 5.3  # 车往球右边，就让adjust_x往+方向调整，车往球左边，就让adjust_x往-方向调整
            self.camera_angle = 85
            self.adjust_y = 0

            static_balls = []
            self.use = False

            # det_boxes, scores, ids = infer_img(image, self.net, self.MODEL_H, self.MODEL_W,
            #                                    thred_nms=0.85, thred_cond=0.1)  # 此处进行修改，防止一直识别到其他东西认为红球
            # for box, score, label_id in zip(det_boxes, scores, ids):  # 在处理过的图像上面进行操作
            #     label = '%s' % (self.dict_labels[label_id])  # 标签，利用标签得到对红球的判断
            #
            #     # 找到了红球
            #     if label == self.SIDE:
            #         """
            #         @ 只有读取到红球并且此时没有其他球正在进行任务，才进行后续操作
            #         """
            #         x, y = plot_one_box(box.astype(np.int16), image,
            #                             color=(0, 0, 255) if self.SIDE == 'red_ball' else (255, 0, 0),
            #                             label=label, line_thickness=None)
            #         if x != 0 and y != 0:
            #             static_balls.append((x, y))
            img, pred_boxes, pred_confes, pred_classes = detect(image, self.net, self.MODEL_H, self.MODEL_W, 0.45,
                                                                0.5)
            if len(pred_boxes) > 0:
                for i, _ in enumerate(pred_boxes):
                    label = '%s' % (self.dict_labels[pred_classes[i]])  # 标签，利用标签得到对红球的判断
                    if label == self.SIDE:
                        # self.in_roi = True
                        x, y = plot_one_box_new(image, img, self.shape, pred_boxes[i], pred_classes[i], pred_confes[i])

                        if x != 0 and y != 0:
                            static_balls.append((x, y))

            cv2.imshow('START POINT', image)
            cv2.waitKey(1)

            self.frame_cnt += 1
            self.filter_input.append(static_balls)

            if self.frame_cnt < 2:
                return False

            static_objects = filter_static_objects(self.filter_input, 1000000000000000, 10000000000000)
            static_objects = [tuple(obj) for obj in static_objects]
            # print(f'filter_input: {self.filter_input}')
            # print(f'static_obj: {static_objects}')
            print(f"Number of static objects detected: {len(static_objects)}")
            # 滤波变量清零
            self.filter_input.clear()
            static_balls.clear()
            self.frame_cnt = 0

            if len(static_objects) != 0:
                # min_x, min_y = Get_min_depth(static_objects, depth_frame)
                min_x, min_y = Get_min_distance(static_objects)
                static_objects.clear()
                distance = depth_frame.get_distance(min_x, min_y) * 100 + 9.5  # 得到的距离单位是m，需要进行单位转换，加上球半径得到距离球心的距离

                # points是相机坐标系中的坐标，其中第一个参数表示的是深度摄像头的内参矩阵
                points = rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics,
                                                         [min_x, min_y], distance)
                # 相机坐标系中的数据转换成为世界坐标系,camera_angle=0，垂直朝下看去
                relative_x = (points[0] + self.adjust_x)  # 此处数据根据机械组给出
                relative_y = (points[2] * math.cos(Angles2Radians(90 - self.camera_angle)) - points[1] * math.sin(
                    Angles2Radians(90 - self.camera_angle)) + self.adjust_y)  # 此处数据根据机械组给出

                angle = odom_val[-1]
                rad = (math.pi / 180) * angle

                mid_relative_x = (relative_x - relative_y * math.tan(rad)) * math.cos(rad)
                mid_relative_y = relative_y / math.cos(rad) + (
                        relative_x - relative_y * math.tan(rad)) * math.sin(rad)

                relative_x = mid_relative_x  # relative_x对应的是START点位的坐标，real_relative_x对应的是球框前的真实坐标
                relative_y = mid_relative_y
                real_relative_x = mid_relative_x
                real_relative_y = mid_relative_y
                print("real_points(DIRECT):", real_relative_x, real_relative_y)

                if 0 <= decision <= 4:
                    self.use = True
                    relative_x = real_relative_x + self.distance_x[decision]  # 要将球框前的坐标转为对应的START点位的坐标
                    relative_y = real_relative_y - self.distance_y[decision]
                    print("relative_points(START):", relative_x, relative_y)
                    # if relative_y <= -self.ZONE3_Y_LIM - self.distance_y[decision]:
                    #     relative_y = -self.ZONE3_Y_LIM - self.distance_y[decision]
                    # if relative_x <= -self.ZONE3_X_LIM - self.distance_x[decision]:
                    #     relative_x = -self.ZONE3_X_LIM - self.distance_x[decision]
                    # elif relative_x >= self.ZONE3_X_LIM - self.distance_x[decision]:
                    #     relative_x = self.ZONE3_X_LIM - self.distance_x[decision]
                # else:  # No silo
                #     if relative_y <= -self.ZONE3_Y_LIM:
                #         relative_y = -self.ZONE3_Y_LIM
                #     if relative_x <= -self.ZONE3_X_LIM:
                #         relative_x = -self.ZONE3_X_LIM
                #     elif relative_x >= self.ZONE3_X_LIM:
                #         relative_x = self.ZONE3_X_LIM

                if relative_y <= -self.ZONE3_Y_LIM:
                    relative_y = -self.ZONE3_Y_LIM
                if relative_x <= -self.ZONE3_X_LIM:
                    relative_x = -self.ZONE3_X_LIM
                elif relative_x >= self.ZONE3_X_LIM:
                    relative_x = self.ZONE3_X_LIM

                # if relative_x <= -300 or relative_x >= 300:
                #     relative_x = 0
                #     print("Find Wrong Ball!")

                if not self.sent:
                    self.send_pos_x = -relative_x  # START点位的坐标用于判断什么时候进死角，什么时候转弯推墙
                    self.send_pos_y = -relative_y
                    self.real_send_pos_x = -real_relative_x  # 对应球框前的真实坐标
                    self.real_send_pos_y = -real_relative_y

                    # red 左边不对
                    if self.send_pos_x >= 120:  # Temp
                        self.send_pos_x += 10
                    if self.send_pos_y <= -410:
                        self.send_pos_y = -410
                        if self.send_pos_x <= -160:  # 球在右边死角
                            self.send_pos_x = -165
                            self.send_pos_y = -405
                            self.flag = 1  # 向右转45°
                        elif self.send_pos_x >= 155:  # 球在左边死角
                            self.send_pos_x = 160
                            self.send_pos_y = -400
                            self.flag = 2  # 向左转45°

                        elif -160 < self.send_pos_x <= 0:
                            self.flag = None
                        elif 0 < self.send_pos_x < 155:
                            self.flag = None
                            self.send_pos_x += 10
                    elif self.send_pos_y > -410:
                        if self.send_pos_x <= -150:  # 球在右边墙
                            self.send_pos_y += 2
                            self.send_pos_x = -135
                            self.flag = 3  # 向右转90°

                        elif self.send_pos_x >= 160:  # 球在左边墙
                            # red
                            self.send_pos_y = self.send_pos_y  # 不变
                            self.send_pos_x = 140
                            self.flag = 4  # 向左转90°

                        elif -150 < self.send_pos_x <=  0:
                            self.flag = None
                        elif 0 < self.send_pos_x < 160:
                            self.flag = None
                            self.send_pos_x += 10
                    # 转化为球框前看球的坐标系
                    if 0 <= decision <= 4:
                        self.real_send_pos_x = self.send_pos_x + self.distance_x[decision]
                        self.real_send_pos_y = self.send_pos_y - self.distance_y[decision]

                    if self.use:  # 表示是球框前看球的点位
                        if self.real_send_pos_y <= -520:
                            self.real_send_pos_y = -520  # 最远给定的是-430，算上82的y值差，最远到512
                        if decision == 4:
                            if self.flag == 2:  # 球在左边死角的情况
                                self.real_send_pos_x = -21
                            elif self.flag == 4:  # 球在左边墙壁的情况
                                self.real_send_pos_x = -30
                                self.real_send_pos_y = self.real_send_pos_y + 6
                            # elif self.flag is None:
                            #     self.real_send_pos_x = self.real_send_pos_x + 20
                        elif decision == 3:
                            if self.flag == 2:  # 球在左边死角的情况
                                self.real_send_pos_x = 83
                            elif self.flag == 4:  # 球在左边墙壁的情况
                                self.real_send_pos_x = 75
                                self.real_send_pos_y = self.real_send_pos_y
                            elif self.flag == 1:  # 球在右边死角的情况
                                self.real_send_pos_x = -225
                        elif decision == 2:
                            if self.flag == 2:  # 左边死角
                                self.real_relative_x = 140
                                self.real_send_pos_y = -515
                            elif self.flag == 1:
                                self.real_send_pos_y -= 6
                            elif self.flag == 4:  # 左边墙
                                self.real_send_pos_x = 130
                                self.real_send_pos_y = self.real_send_pos_y - 3
                        elif decision == 1:
                            if self.flag == 1:  # 球在右边死角的情况
                                self.real_send_pos_x = -70
                            elif self.flag == 2:  # 球在左边死角的情况
                                self.real_send_pos_y -= 3
                                self.real_send_pos_x -= 3
                            elif self.flag == 3:  # 球在右边墙壁的情况
                                self.real_send_pos_x = -70
                                self.real_send_pos_y = self.real_send_pos_y - 8
                        elif decision == 0:
                            if self.flag == 1:  # 球在右边死角的情况
                                self.real_send_pos_x = 40
                            elif self.flag == 3:  # 球在右边墙壁的情况
                                self.real_send_pos_x = 30
                                self.real_send_pos_y = self.real_send_pos_y + 2
                    # 整体调整  此处的左右是相对于机器人的逆向的左右，是从silo向下看的
                    if self.flag == 4:  # 左墙
                        self.send_pos_x = self.send_pos_x - 20
                        self.real_send_pos_x = self.real_send_pos_x - 20
                    elif self.flag == 3:  # 右墙
                        self.send_pos_x = self.send_pos_x + 20
                        self.real_send_pos_x = self.real_send_pos_x + 20
                    elif self.flag == 2:  # 左角
                        self.send_pos_x = self.send_pos_x - 20
                        self.real_send_pos_x = self.real_send_pos_x - 20
                    elif self.flag == 1:  # 右角
                        self.send_pos_x = self.send_pos_x + 20
                        self.real_send_pos_x = self.real_send_pos_x + 20
                    else:
                        self.send_pos_x = self.send_pos_x - 10
                        self.real_send_pos_x = self.real_send_pos_x - 10
                        self.send_pos_y += 38
                        self.real_send_pos_y += 38
                    # self.send_pos_x -= 0
                    # self.real_send_pos_x -= 0
                    self.send_pos_y -= 30
                    self.real_send_pos_y -= 30

                if self.send_pos_x != 0 or self.send_pos_y != 0:
                    self.sent = True
                    if self.use:
                        print("SILO")
                        # for i in range(0, 5):
                        # print(self.real_send_pos_x, self.real_send_pos_y, self.flag)
                        print('START:', self.send_pos_x, self.send_pos_y, self.flag)
                        print('SILO: ', self.real_send_pos_x, self.real_send_pos_y, self.flag)
                        ser.send_yzy_idea(self.real_send_pos_x, self.real_send_pos_y, flag=self.flag)  # 组装发送格式
                    else:
                        print("NO SILO")
                        # for i in range(0, 5):
                        print('START:', self.send_pos_x, self.send_pos_y, self.flag)
                        ser.send_yzy_idea(self.send_pos_x, self.send_pos_y, flag=self.flag)
                elif self.send_pos_x and self.send_pos_y == 0:
                    print("Nothing")
            self.use = False
            self.flag = None
            self.real_send_pos_x = 0
            self.real_send_pos_y = 0
            self.send_pos_x = 0
            self.send_pos_y = 0

            return True


    def yzy_idea(self, image, depth_frame, ser, decision, odom_val, WING_LEN=67):
        # 异常处理
        if self.ZONE3_X_LIM / WING_LEN >= 3 or self.ZONE3_X_LIM / WING_LEN < 2:
            raise ValueError('所设置的光翼长度太短，已经不符合五段区间分配！')

        # 创建变量
        self.adjust_x = -20
        self.camera_angle = 85
        self.adjust_y = 0
        det_boxes, scores, ids = infer_img(image, self.net, self.MODEL_H, self.MODEL_W,
                                           thred_nms=0.85, thred_cond=0.1)  # 此处进行修改，防止一直识别到其他东西认为红球
        for box, score, label_id in zip(det_boxes, scores, ids):  # 在处理过的图像上面进行操作
            label = '%s' % (self.dict_labels[label_id])  # 标签，利用标签得到对红球的判断

            # 找到了红球
            if label == self.SIDE:
                """
                @ 只有读取到红球并且此时没有其他球正在进行任务，才进行后续操作
                """
                x, y = plot_one_box(box.astype(np.int16), image,
                                    color=(0, 0, 255) if self.SIDE == 'red_ball' else (255, 0, 0),
                                    label=label, line_thickness=None)
                if x != 0 and y != 0:
                    self.position_list.append((x, y))

        if len(self.position_list) != 0:
            min_x, min_y = Get_min_depth(self.position_list, depth_frame)
            self.position_list.clear()
            distance = depth_frame.get_distance(min_x, min_y) * 100 + 9.5  # 得到的距离单位是m，需要进行单位转换，加上球半径得到距离球心的距离

            # points是相机坐标系中的坐标，其中第一个参数表示的是深度摄像头的内参矩阵
            points = rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics,
                                                     [min_x, min_y], distance)
            # 相机坐标系中的数据转换成为世界坐标系,camera_angle=0，垂直朝下看去
            relative_x = (points[0] + self.adjust_x)  # 此处数据根据机械组给出
            relative_y = (points[2] * math.cos(Angles2Radians(90 - self.camera_angle)) - points[1] * math.sin(
                Angles2Radians(90 - self.camera_angle)) + self.adjust_y)  # 此处数据根据机械组给出

            angle = odom_val[-1]
            rad = (math.pi / 180) * angle

            mid_relative_x = (relative_x - relative_y * math.tan(rad)) * math.cos(rad)
            mid_relative_y = relative_y / math.cos(rad) + (
                    relative_x - relative_y * math.tan(rad)) * math.sin(rad)

            relative_x = mid_relative_x  # relative_x对应的是START点位的坐标，real_relative_x对应的是球框前的真实坐标
            relative_y = mid_relative_y
            real_relative_x = mid_relative_x
            real_relative_y = mid_relative_y
            print("real_points:", real_relative_x, real_relative_y)

            if 0 <= decision <= 4:
                self.use = True
                relative_x = real_relative_x + self.distance_x[decision]  # 要将球框前的坐标转为对应的START点位的坐标
                relative_y = real_relative_y - self.distance_y[decision]
                print("relative_points:", relative_x, relative_y)

            if relative_x <= -300 or relative_x >= 300:
                relative_x = 0
                print("Find Wrong Ball!")

            if not self.sent:
                self.send_pos_x = -relative_x  # START点位的坐标用于判断什么时候进死角，什么时候转弯推墙
                self.send_pos_y = -relative_y
                self.real_send_pos_x = -real_relative_x  # 对应球框前的真实坐标
                self.real_send_pos_y = -real_relative_y

                if self.send_pos_y <= -420:
                    self.send_pos_y = -420
                    if self.send_pos_x <= -150:  # 球在右边死角
                        self.send_pos_x = -135
                        self.send_pos_y = -420
                        self.flag = 1  # 向右转45°

                    elif self.send_pos_x >= 140:  # 球在左边死角
                        self.send_pos_x = 140
                        self.send_pos_y = -410
                        self.flag = 2  # 向左转45°

                    elif -140 < self.send_pos_x < 140:
                        self.flag = None

                elif self.send_pos_y > -400:
                    if self.send_pos_x <= -150:  # 球在右边墙
                        self.send_pos_y = self.send_pos_y - 10
                        self.send_pos_x = -136
                        self.flag = 3  # 向右转90°

                    elif self.send_pos_x >= 135:  # 球在左边墙
                        self.send_pos_y = self.send_pos_y - 3
                        self.send_pos_x = 130
                        self.flag = 4  # 向左转90°

                    elif -150 < self.send_pos_x < 135:
                        self.flag = None

                # 转化为球框前看球的坐标系
                if 0 <= decision <= 4:
                    self.real_send_pos_x = self.send_pos_x + self.distance_x[decision]
                    self.real_send_pos_y = self.send_pos_y - self.distance_y[decision]

                if self.use:  # 表示是球框前看球的点位
                    if self.real_send_pos_y <= -512:
                        self.real_send_pos_y = -490  # 最远给定的是-430，算上82的y值差，最远到512
                    if decision == 4:
                        self.real_send_pos_x = self.real_send_pos_x - 5
                        if self.flag == 2:  # 球在左边死角的情况
                            self.real_send_pos_x = -30
                        elif self.flag == 4:  # 球在左边墙壁的情况
                            self.real_send_pos_x = -30
                            self.real_send_pos_y = self.real_send_pos_y + 3
                    if decision == 3:
                        if self.flag == 2:  # 球在左边死角的情况
                            self.real_send_pos_x = 75
                        elif self.flag == 4:  # 球在左边墙壁的情况
                            self.real_send_pos_x = 75
                            self.real_send_pos_y = self.real_send_pos_y - 3
                        elif self.flag == 1:  # 球在右边死角的情况
                            self.real_send_pos_x = -220
                    if decision == 1:
                        if self.flag == 1:  # 球在右边死角的情况
                            self.real_send_pos_x = -70
                        elif self.flag == 3:  # 球在右边墙壁的情况
                            self.real_send_pos_x = -70
                            self.real_send_pos_y = self.real_send_pos_y - 5
                    if decision == 0:
                        if self.flag == 1:  # 球在右边死角的情况
                            self.real_send_pos_x = 30
                        elif self.flag == 3:  # 球在右边墙壁的情况
                            self.real_send_pos_x = 25
                            self.real_send_pos_y = self.real_send_pos_y - 5

            if self.send_pos_x != 0 and self.send_pos_y != 0:
                self.sent = True
                print("points:", self.send_pos_x, self.send_pos_y)
                if self.use:
                    print("SILO")
                    # for i in range(0, 5):
                    # print(self.real_send_pos_x, self.real_send_pos_y, self.flag)
                    ser.send_yzy_idea(self.real_send_pos_x, self.real_send_pos_y, flag=self.flag)  # 组装发送格式
                else:
                    print("NO SILO")
                    # for i in range(0, 5):
                    # print(self.send_pos_x, self.send_pos_y, self.flag)
                    ser.send_yzy_idea(self.send_pos_x, self.send_pos_y, flag=self.flag)
            elif self.send_pos_x != 0 and self.send_pos_y >= -210:
                # pass  # TODO: 摄像头深度可能失灵，要写摄像头重启
                print(min_x, min_y, distance)
                print(self.send_pos_x, self.send_pos_y)
            elif self.send_pos_x and self.send_pos_y == 0:
                print("Nothing")
        self.position_list.clear()
        self.use = False
        self.real_send_pos_x = 0
        self.real_send_pos_y = 0
        self.send_pos_x = 0
        self.send_pos_y = 0
        cv2.imshow('START POINT', image)

    @staticmethod
    def get_downhill(odom_val):  # check if got down the hill, if yes, stop sending ball pos to insure not to waste one chance of the machine arm
        arrived = False
        if odom_val[2] < -0.05:  # Z
            arrived = True
        return arrived


    def yzy_track(self, image, depth_frame, ser, x_arriced: bool, WING_LEN=67):
        # 异常处理
        if self.ZONE3_X_LIM / WING_LEN >= 3 or self.ZONE3_X_LIM / WING_LEN < 2:
            raise ValueError('所设置的光翼长度太短，已经不符合五段区间分配！')

        # 创建变量
        self.adjust_x = -15
        self.camera_angle = 30
        self.adjust_y = 0
        relative_x, relative_y = 0, 0  # i表示摄像头的向下转动次数
        update = False

        det_boxes, scores, ids = infer_img(image, self.net, self.MODEL_H, self.MODEL_W,
                                           thred_nms=0.9, thred_cond=0.7)
        for box, score, label_id in zip(det_boxes, scores, ids):  # 在处理过的图像上面进行操作
            label = '%s' % (self.dict_labels[label_id])  # 标签，利用标签得到对红球的判断

            # 找到了红球
            if label == self.SIDE:
                update = True  # 找到球，标记为可以更新
                """
                # @ 只有读取到红球并且此时没有其他球正在进行任务，才进行后续操作
                """
                x, y = plot_one_box(box.astype(np.int16), image,
                                    color=(0, 0, 255) if self.SIDE == 'red_ball' else (255, 0, 0),
                                    label=label, line_thickness=None)
                self.position_list.append((x, y))

                min_x, min_y = Get_min_depth(self.position_list, depth_frame)
                distance = depth_frame.get_distance(min_x, min_y) * 100 + 9.5  # 得到的距离单位是m，需要进行单位转换，加上球半径得到距离球心的距离

                # points是相机坐标系中的坐标，其中第一个参数表示的是深度摄像头的内参矩阵
                points = rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics,
                                                         [min_x, min_y], distance)
                # 相机坐标系中的数据转换成为世界坐标系,camera_angle=0，垂直朝下看去
                relative_x = (points[0] + self.adjust_x)  # 此处数据根据机械组给出
                relative_y = (points[2] * math.cos(Angles2Radians(90 - self.camera_angle)) - points[1] * math.sin(
                    Angles2Radians(90 - self.camera_angle)) + self.adjust_y)  # 此处数据根据机械组给出

        # 以下是数据的输出部分
        # if not self.sent:
        self.send_pos_x = -relative_x
        self.send_pos_y = -relative_y

        if self.send_pos_y <= -420:
            self.send_pos_y = -420
        if self.send_pos_x <= -220 or self.send_pos_x >= 220:
            self.send_pos_x = 0
            print("Find Wrong Ball!")

        if relative_y and relative_x == 0:
            ser.send_stay()
        else:
            ser.send_track(self.send_pos_x, update)  # 组装发送格式，只有x是有效数据，y值不需要

        cv2.imshow('img', image)


    def dtd_idea(self, image, depth_frame, ser, x_arrived:bool, WING_LEN=67):
        # 异常处理
        if self.ZONE3_X_LIM / WING_LEN >= 3 or self.ZONE3_X_LIM / WING_LEN < 2:
            raise ValueError('所设置的光翼长度太短，已经不符合五段区间分配！')

        # 创建变量
        self.adjust_x = -15
        ball_x_list = []  # 目标球x轴分布列表
        area_cnt = [0, 0, 0, 0, 0]  # 目标球区间分布情况
        OFFSET = 5  # 单位：cm, 离墙距离
        AREA_POS = [self.ZONE3_X_LIM - WING_LEN / 2 - OFFSET,
                    WING_LEN,
                    0,
                    -WING_LEN,
                    -(self.ZONE3_X_LIM - WING_LEN / 2 - OFFSET)]  # 这样给符号是为了x轴翻转
        ball_in_area = False  # 是否有球在范围内

        det_boxes, scores, ids = infer_img(image, self.net, self.MODEL_H, self.MODEL_W,
                                           thred_nms=0.9, thred_cond=0.5)
        for box, score, label_id in zip(det_boxes, scores, ids):  # 在处理过的图像上面进行操作
            label = '%s' % (self.dict_labels[label_id])  # 标签，利用标签得到对红球的判断

            # 找到了红球
            if label == self.SIDE:
                """
                @ 只有读取到红球并且此时没有其他球正在进行任务，才进行后续操作
                """
                x, y = plot_one_box(box.astype(np.int16), image,
                                    color=(0, 0, 255) if self.SIDE == 'red_ball' else (255, 0, 0),
                                    label=label, line_thickness=None)

                distance = depth_frame.get_distance(x, y) * 100 + 9.5  # 得到的距离单位是m，需要进行单位转换，加上球半径得到距离球心的距离

                # points是相机坐标系中的坐标，其中第一个参数表示的是深度摄像头的内参矩阵
                points = rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics,
                                                         [x, y], distance)
                # 相机坐标系中的数据转换成为世界坐标系,camera_angle=0，垂直朝下看去
                relative_x = points[0] + self.adjust_x  # 此处数据根据机械组给出

                if relative_x == 0:  # 只有不等于0的时候才能记录有效数据
                    continue
                # print(relative_x)  # Temp
                ball_x_list.append(relative_x)

        # 统计各区域球数
        for x in ball_x_list:
            if abs(x) < WING_LEN / 2:
                area_cnt[2] += 1
            elif abs(x) < 3 * WING_LEN / 2:
                if x < 0:
                    area_cnt[1] += 1
                else:
                    area_cnt[3] += 1
            else:
                if x < 0:
                    area_cnt[0] += 1
                else:
                    area_cnt[4] += 1

        print(area_cnt)
        if not self.sent:  # 保持值不改变
            if any(area_cnt):
                max_index = area_cnt.index(max(area_cnt))
                self.send_pos = AREA_POS[max_index]
            else:
                ser.send_stay()  # 无球就等着

        if not x_arrived:
            if self.send_pos is not None:
                self.sent = True
                ser.send_dtd_idea(self.send_pos, -500)  # 组装发送格式，只有x是有效数据，y值不需要
        else:
            ball_in_area = False if area_cnt[2] == 0 else True  # 如果中间一列空了（就是要抓的一列），则行进的一列无球

        cv2.imshow('img', image)

        return ball_in_area

    def dtd_idea_Kmeans(self, image, depth_frame, ser, x_arrived:bool, WING_LEN=70.5,
                        OFFSET=5  # 单位：cm, 离墙距离
                        ):
        # 创建变量
        self.adjust_x = -15
        ball_x_list = []  # 目标球x轴分布列表
        ball_in_area = False  # 是否有球在范围内


        det_boxes, scores, ids = infer_img(image, self.net, self.MODEL_H, self.MODEL_W,
                                           thred_nms=0.9, thred_cond=0.5)
        for box, score, label_id in zip(det_boxes, scores, ids):  # 在处理过的图像上面进行操作
            label = '%s' % (self.dict_labels[label_id])  # 标签，利用标签得到对红球的判断

            # 找到了红球
            if label == self.SIDE:
                """
                @ 只有读取到红球并且此时没有其他球正在进行任务，才进行后续操作
                """
                x, y = plot_one_box(box.astype(np.int16), image,
                                    color=(0, 0, 255) if self.SIDE == 'red_ball' else (255, 0, 0),
                                    label=label, line_thickness=None)

                distance = depth_frame.get_distance(x, y) * 100 + 9.5  # 得到的距离单位是m，需要进行单位转换，加上球半径得到距离球心的距离

                # points是相机坐标系中的坐标，其中第一个参数表示的是深度摄像头的内参矩阵
                points = rs.rs2_deproject_pixel_to_point(depth_frame.profile.as_video_stream_profile().intrinsics,
                                                         [x, y], distance)
                # 相机坐标系中的数据转换成为世界坐标系,camera_angle=0，垂直朝下看去
                relative_x = points[0] + self.adjust_x  # 此处数据根据机械组给出

                if relative_x == 0:  # 只有不等于0的时候才能记录有效数据
                    continue
                # print(relative_x)  # Temp
                ball_x_list.append(relative_x)

        if not self.sent:  # 保持值不改变
            if ball_x_list:
                self.send_pos = K_means(ball_x_list)
                if abs(self.send_pos) > self.ZONE3_X_LIM:  # 超域限制
                    X_LIM = self.ZONE3_X_LIM - OFFSET
                    self.send_pos  = X_LIM if self.send_pos > 0 else -X_LIM
            else:
                ser.send_stay()  # 无球就等着

        if not x_arrived:
            if self.send_pos is not None:
                self.sent = True
                ser.send_dtd_idea(self.send_pos, -500)  # 组装发送格式，只有x是有效数据，y值不需要
        else:  # 仅在x到达情况下改变 ball_in_area 的值，默认情况下为False
            for item in ball_x_list:  # 如果抓取范围空了，则行进的一列无球
                ball_in_area = True if abs(item) < WING_LEN / 2 else False

        cv2.imshow('img', image)

        return ball_in_area


def Radians2Angles(radians):
    """
    @ 弧度制转化为角度制
    Args:
        radians: 弧度
    Returns: 角度
    """

    return radians / math.pi * 180


def Angles2Radians(angle):
    """
    @ 角度制转化为弧度制
    Args:
        angle: 角度
    Returns: 弧度
    """
    return angle / 180 * math.pi


def Get_coordinates(pixel_x, pixel_y, aligned_depth_frame):
    """
    @ 通过读取像素点的中心x和y，得到深度摄像头距离其的距离
    """
    return aligned_depth_frame.get_distance(pixel_x, pixel_y)


def Get_min_depth(list, depth_frame):
    min_dist = float('inf')
    near_x, near_y = 0, 0
    for i in range(len(list)):
        depth = depth_frame.get_distance(list[i][0], list[i][1])
        if depth < min_dist:
            min_dist = depth
            near_x, near_y = list[i][0], list[i][1]
    return near_x, near_y  # 返回距离最小的球坐标


def Get_max_depth(list, depth_frame):
    max_dist = float('0')
    near_x, near_y = 0, 0
    for i in range(len(list)):
        depth = depth_frame.get_distance(list[i][0], list[i][1])
        if depth > max_dist:
            max_dist = depth
            near_x, near_y = list[i][0], list[i][1]
    return near_x, near_y  # 返回距离最小的球坐标


def Get_min_distance(list):
    min_dist = float('inf')
    near_x, near_y = 0, 0
    for i in range(len(list)):
        list_x = list[i][0] - 320
        list_y = list[i][1] - 360
        depth = list_x ** 2 + list_y ** 2
        if depth < min_dist:
            min_dist = depth
            near_x, near_y = list[i][0], list[i][1]
    return near_x, near_y


def K_means(target_list, n_clusters=5, n_init=3):
    # 使用K-Means算法将数据聚类为n_clusters个类别，进行n_init次计算取最优解
    kmeans = KMeans(n_clusters=n_clusters, n_init=n_init).fit(np.array(target_list).reshape(-1, 1))

    # 找出每个类别的中心点
    centroids = kmeans.cluster_centers_

    # 确定每个数据点属于哪个类别
    labels = kmeans.labels_

    # 找到出现次数最多的类别
    most_common_label = np.bincount(labels).argmax()

    # 输出该类别的中心点作为最终目标的坐标
    target_coordinates = centroids[most_common_label]

    return target_coordinates

# if __name__ == '__main__':
#
#     pipeline, align = rc_utils.init_realsense()
#
#     '''
#     @ 联调用代码
#     @ 深度摄像头读取图像得到球的中心点坐标，传参数给主板，主板进行抓取
#     '''
#     # ………………………………………………………………………………………………………………………… #
#     while True:
#         """
#         @ 以下是对深度摄像头进行设置和读取操作
#         """
#         frames = pipeline.wait_for_frames()
#         aligned_frames = align.process(frames)
#         # Get aligned frames
#         aligned_depth_frame = aligned_frames.get_depth_frame()
#         color_frame = aligned_frames.get_color_frame()
#         # Validate that both frames are valid
#         if not aligned_depth_frame or not color_frame:
#             continue
#
#         # Convert images to numpy arrays
#         depth_image = np.asanyarray(aligned_depth_frame.get_data())
#         color_image = np.asanyarray(color_frame.get_data())
#         depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#         img = color_image
#
#     pipeline.stop()
#     cv2.destroyAllWindows()
