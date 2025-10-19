# Author: Ethan Lee
# Date: 2024.5.16

# 此代码为杂项方法整合，主要包括串口和深度相机
import pyrealsense2 as rs
import serial
import serial.tools.list_ports
import numpy as np
import cv2
import time
import threading


def init_realsense(WIDTH=640, HEIGHT=360):
    # Configure depth and color streams
    # noinspection PyArgumentList
    pipeline = rs.pipeline()
    config = rs.config()

    config.enable_stream(rs.stream.depth, WIDTH, HEIGHT, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, WIDTH, HEIGHT, rs.format.bgr8, 30)
    # Align the depth frame to color frame
    align = rs.align(rs.stream.color)
    # Start streaming
    pipeline.start(config)

    return pipeline, align


def depth_filter(depth_min, depth_max, depth_image, color_image):
    # 将深度图像转换为有效范围内的掩码
    mask = np.logical_and(depth_image >= depth_min * 1000, depth_image <= depth_max * 1000)
    # 创建一个结构元素,用于开运算
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (10, 10))
    # 对mask进行开运算
    mask = cv2.morphologyEx(mask.astype(np.uint8), cv2.MORPH_OPEN, kernel)
    # 将不满足深度要求的点在彩色图像上变为黑色
    color_image[np.logical_not(mask)] = [0, 0, 0]
    # Apply colormap on depth image (image must be converted to 8-bit per pixel first)
    # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
    # Stack both images horizontally
    img = color_image

    return img


# if __name__ == '__main__':
#     # start_numpy = [[0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0], [0, 0, 0]]
#     """
#     @ 以下是深度摄像头初始化设置部分
#     """
#     # Configure depth and color streams
#     # noinspection PyArgumentList
#     pipeline = rs.pipeline()
#     config = rs.config()
#
#     config.enable_stream(rs.stream.depth, 640, 360, rs.format.z16, 30)
#     config.enable_stream(rs.stream.color, 640, 360, rs.format.bgr8, 30)
#     # Align the depth frame to color frame
#     align = rs.align(rs.stream.color)
#     # Start streaming
#     pipeline.start(config)
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
#         # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
#
#         # Stack both images horizontally
#         images = color_image
#         model = loadPolicy()
#
#         cv2.imshow('img', images)
#         dec = Silo('r')
#         # dec.make_decision(images)
#
#         # Red_Silo(images, model)
#         # cv2.imshow('img', img)
#
#     pipeline.stop()
#     cv2.destroyAllWindows()


def setup_self_check(ser, pipeline):
    while True:  # 是否要发送一些东西以确保发送正常？
        frames = pipeline.wait_for_frames()
        if frames:
            break

    return True


class SerialCtrl:

    def __init__(self, serial_port, ros_msg_handler):
        self.timeout = 0.5
        self.data_to_send = None
        self.serial_port = serial_port
        self.start_time = 0
        self.curr_log = None  # 当前的日志信息, 变化时才记录
        self.rm = ros_msg_handler
        self.thread = threading.Thread(target=self.send_data_continuously)
        self.thread.start()

    @staticmethod
    def find_serial():
        ports_list = list(serial.tools.list_ports.comports())
        if len(ports_list) <= 0:
            print("No Find")
        else:
            print("Find")
            for comport in ports_list:
                print(list(comport)[0], list(comport)[1])  # 输出串口设备及其名称

    def secure_send_data(self, data, mode:str, timeout=0.2):  # 发送的数据a必须包括有'\r\n'
        start_time = time.time()

        while True:
            curt_time = time.time()
            hold_time = curt_time - start_time

            if hold_time > timeout:
                if mode == 'd':
                    self.send_decision(data)
                elif mode == 'f':
                    self.send_dtd_idea(data[0], data[1])
                start_time = time.time()

            recv_data = self.receive_data()
            if recv_data == 'ACK':
                break

        self.send_data("(+NNN.N,-NNN.N)+Fe\r\n")  # 表示已经收到ACK，三次握手结束（e的前一位为R）

    def send_data_continuously(self):
        while True:
            if self.data_to_send is None:
                self.send_lidar_update()
            self.send_data(self.data_to_send, self.timeout)

    def send_data(self, data, TIMEOUT=0.5):  # 发送的数据a必须包括有'\r\n'
        curt_time = time.time()
        if curt_time - self.start_time > TIMEOUT:
            # 执行后复位
            self.timeout = 0.5  # 复位timeout为默认值，逻辑有点怪但是不想改了
            self.data_to_send = None

            data = data[:-3] + f'{self.rm.read_recv("o")[5]:+07.2f}' + data[-3:]
            print(f'串口发送：{data}')
            if self.serial_port.isOpen():
                self.serial_port.write(data.encode('utf-8'))
                self.write_log(data, 'SEND')  # 写入日志
            else:
                print("Serial No Open!")

            self.start_time = time.time()


    def receive_data(self):
        try:
            while self.serial_port.isOpen():
                if self.serial_port.in_waiting > 0:
                    data = self.serial_port.readline().decode('utf-8').strip()  # 读取一行数据并解码为文本
                    self.write_log(data, 'RECV')  # 写入日志
                    return data

                elif self.serial_port.in_waiting == 0:
                    break
        except TimeoutError:  # 如果报错则退出
            pass
        except UnicodeDecodeError:
            pass

    def send_yzy_idea(self, x, y, flag):
        if flag == 1:
            # self.send_data(f"({x:+06.1f},{y:+06.1f})+Re\r\n")
            print(flag)
            self.data_to_send = f"({x:+06.1f},{y:+06.1f})+Re\r\n"
        elif flag == 2:
            # self.send_data(f"({x:+06.1f},{y:+06.1f})+Le\r\n")
            print(flag)
            self.data_to_send = f"({x:+06.1f},{y:+06.1f})+Le\r\n"
        elif flag == 3:
            # self.send_data(f"({x:+06.1f},{y:+06.1f})++e\r\n")
            print(flag)
            self.data_to_send = f"({x:+06.1f},{y:+06.1f})++e\r\n"
        elif flag == 4:
            # self.send_data(f"({x:+06.1f},{y:+06.1f})+-e\r\n")
            print(flag)
            self.data_to_send = f"({x:+06.1f},{y:+06.1f})+-e\r\n"
        else:
            # self.send_data(f"({x:+06.1f},{y:+06.1f})+Fe\r\n")
            print(flag)
            self.data_to_send = f"({x:+06.1f},{y:+06.1f})+Fe\r\n"

    def send_track(self, x, update):
        if update is False:
            self.send_data(f"(+WWW.W,-WWW.W)+Fe\r\n")  # 表示没找到球
        elif update is True:
            self.send_data(f"(+WWW.W,{x:+06.1f})+Fe\r\n")  # 找到球后，给的是x的相对坐标

    def send_catch(self, x, y):
        # self.send_data(f"({x:+06.1f},{y:+06.1f})+Fe\r\n", TIMEOUT=0)  # 不要降频
        self.data_to_send = f"({x:+06.1f},{y:+06.1f})+Fe\r\n"
        # self.timeout = 0.1  # 略微降频
        self.timeout = 0  # no slow down

    def send_dtd_idea(self, x, y):
        # self.send_data(f"({x:+06.1f},{y:+06.1f})+Fe\r\n")
        self.data_to_send = f"({x:+06.1f},{y:+06.1f})+Fe\r\n"

    def send_decision(self, decision):
        # self.send_data(f"(+FFF.F,-FFF.F)+{decision}e\r\n")  # Final
        self.data_to_send = f"(+FFF.F,-FFF.F)+{decision}e\r\n"  # Final

    def send_stay(self):
        # self.send_data('(+LLL.L,-LLL.L)+Le\r\n')  # Stay for command
        self.data_to_send = '(+LLL.L,-LLL.L)+Le\r\n'  # Stay for command

    def send_retry(self):  # 到达暂存区开始取球的重开
        # self.send_data('(+EEE.E,-EEE.E)+Ee\r\n')  # Error
        self.data_to_send = '(+EEE.E,-EEE.E)+Ee\r\n'  # Error

    def send_interrupt(self):  # 区域内无球了，直接打断，进行重试
        # self.send_data('(+SSS.S,-SSS.S)+Se\r\n')  # Stop
        self.data_to_send = '(+SSS.S,-SSS.S)+Se\r\n'  # Stop

    def send_transition(self):  # 取球后的过渡状态
        # self.send_data('(+NNN.N,-NNN.N)+Fe\r\n')  # None
        self.data_to_send = '(+NNN.N,-NNN.N)+Fe\r\n'  # None

    def send_obs_avoid(self):
        self.data_to_send = '(+TTT.T,-TTT.T)+Fe\r\n'  # 避障

    def send_lidar_update(self):  # 默认发送的值，用于更新雷达数据
        self.data_to_send = '(+AAA.A,-AAA.A)+Ae\r\n'

    # TODO: 记录一下数据包的各位解释

    def write_log(self, data, mode:str):
        if data == self.curr_log:
            return
        # 获取当前时间
        current_time = time.localtime()

        # 格式化时间
        formatted_time = time.strftime("%Y-%m-%d %H:%M:%S", current_time)

        log = f'[{formatted_time}] [{mode}]: {data}\r\n'
        with open('/home/ethan/yolov5-balls/serial_log.txt', 'a') as f:  # 追加模式
            f.write(log)
        self.curr_log = data

    def clear_log(self):
        self.curr_log = None
        with open('/home/ethan/yolov5-balls/serial_log.txt', 'w') as f:  # 写入模式
            f.write('')
            print('已清空串口日志')