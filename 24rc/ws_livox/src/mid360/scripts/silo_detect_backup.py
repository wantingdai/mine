#!/usr/bin/python3
# coding=utf-8
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import numpy as np
import socket
import threading


class SocketSend:
    def __init__(self):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 socket 对象
        self.host = socket.gethostname()  # 获取本地主机名
        self.port = 23456  # 设置端口号

    def send(self, ros_msg):
        self.s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 socket 对象
        try:
            self.s.connect((self.host, self.port))  # 连接服务，指定主机和端口
            self.s.send(ros_msg.encode('utf-8'))  # 发送数据
        except ConnectionRefusedError:
            rospy.loginfo('接收端未启动！')
        finally:
            self.s.close()


class SocketRecv:
    def __init__(self):
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 socket 对象
        self.host = socket.gethostname()  # 获取本地主机名
        self.port = 34567  # 设置端口号
        self.serversocket.bind((self.host, self.port))  # 绑定端口
        self.serversocket.listen(5)  # 设置最大连接数，超过后排队
        self.serversocket.setblocking(False)  # 设置为非阻塞模式
        self.recver_thread = threading.Thread(target=self.recver)
        self.recver_thread.start()
        self.command = None

    def recver(self):  # 建立客户端连接
        while threading.main_thread().is_alive():
            try:
                clientsocket, addr = self.serversocket.accept()
                msg = clientsocket.recv(1024)  # 1024是一个参数，用于指定recv方法一次性最多接收的数据量，单位是字节
                ros_msg = msg.decode('utf-8')
                clientsocket.close()
                self.command = eval(ros_msg)
            except BlockingIOError:
                continue  # 如果没有可接受的连接，继续循环

    def read_recv(self):
        while threading.main_thread().is_alive():
            if self.command is not None:
                return self.command
            else:
                # print('发送端未启动！')
                continue


def PointCloudCallback(data):
    global filter_cnt, layer_point_cnt, layer_point_cnt_single

    msg = ros_recv.read_recv()
    # print(type(msg), msg)
    if type(msg) == type(0):
        cmd = msg
        # print(f'Command: {cmd}, Silo: None')
    # elif type(msg) == type((0, 0)):
    else:
        cmd, dec_silo = ros_recv.read_recv()
        # print(f'Command: {cmd}, Silo: {dec_silo}')

    if cmd == 1:
        silo_cnt = [0 for _ in range(5)]
        layer_activated = [[False for _ in range(LAYER_NUM)] for _ in range(5)]  # 每一个框都有一个自己的列表
        gen = point_cloud2.read_points(data, field_names=['x', 'y', 'z', 'intensity'], skip_nans=True)
        p_cnt = 0

        for p in gen:
            if not DEV:  # 修正为顺时针旋转90°的坐标系，修正后的坐标系用u、v、w表示
                p = list(p)
                for i in range(3):  # 进行预设偏移
                    p[i] += OFFSET_POSE[i]

                tmp = p[0]
                p[0] = -p[1]  # u = -y
                p[1] = tmp  # v = x

            for silo in range(5):
                if Y_MIN < p[1] < Y_MAX and (HEIGHT_MIN < p[2] < HEIGHT_MAX):  # y,z limit
                    if CENTER_POS[silo] + X_MIN < p[0] < CENTER_POS[silo] + X_MAX:  # x limit
                        if 0 < p[-1] < 20:  # intensity limit
                            for i in range(LAYER_NUM):  # 遍历框范围中的点，判断其所在的层
                                if LAYERS[i] <= p[2] < LAYERS[i + 1]:
                                    layer_point_cnt[silo][i] += 1
            filter_cnt += 1
            p_cnt += 1

        if filter_cnt > FILTER_FRAME:  # 取n帧
            # TODO: 收到清零信号后清零
            for silo in range(5):
                for i in range(LAYER_NUM):
                    if layer_point_cnt[silo][i] >= ACT_THRESH[silo]:
                        layer_activated[silo][i] = True
                # 处理框中情况
                num_true = sum(layer_activated[silo])
                if num_true == 1:
                    silo_cnt[silo] = 1
                elif num_true == 2:
                    silo_cnt[silo] = 2
                elif num_true >= 3:
                    silo_cnt[silo] = 3

                # 输出
                print(p_cnt, end='\t')
                print(layer_point_cnt[silo], end='\t')
                print(layer_activated[silo], end='\t')
                print(silo_cnt)

            layer_point_cnt = [[0 for _ in range(LAYER_NUM)] for _ in range(5)]  # 清零
            filter_cnt = 0

            print('END DETECT')
            ros_send.send(str(silo_cnt))

    elif cmd == 2:
        silo_cnt_single = 0
        layer_activated_single = [False for _ in range(LAYER_NUM)]  # 每一个框都有一个自己的列表
        gen = point_cloud2.read_points(data, field_names=['x', 'y', 'z', 'intensity'], skip_nans=True)
        p_cnt = 0

        for p in gen:
            if Y_MIN < p[1] < Y_MAX and (HEIGHT_MIN < p[2] < HEIGHT_MAX):  # y,z limit
                if CENTER_POS[dec_silo] + X_MIN < p[0] < CENTER_POS[dec_silo] + X_MAX:  # x limit
                    if p[-1] < 50:  # intensity limit
                        for i in range(LAYER_NUM):  # 遍历框范围中的点，判断其所在的层
                            if LAYERS[i] <= p[2] < LAYERS[i + 1]:
                                layer_point_cnt_single[i] += 1
            filter_cnt += 1
            p_cnt += 1

        if filter_cnt > FILTER_FRAME:  # 取n帧
            # TODO: 收到清零信号后清零
            for i in range(LAYER_NUM):
                if layer_point_cnt_single[i] >= NEAR_THRESH:
                    layer_point_cnt_single[i] = True
            # 处理框中情况
            num_true = sum(layer_activated_single)
            if num_true == 1:
                silo_cnt_single = 1
            elif num_true == 2:
                silo_cnt_single = 2
            elif num_true >= 3:
                silo_cnt_single = 3

            # 输出
            print(p_cnt, end='\t')
            print(layer_point_cnt_single, end='\t')
            print(layer_activated_single, end='\t')
            print(silo_cnt_single)

            layer_point_cnt = [0 for _ in range(LAYER_NUM)]  # 清零
            filter_cnt = 0

            print('END CHECK')
            ros_send.send(str(silo_cnt_single))


def pc_listener():
    rospy.init_node('silo_detector', anonymous=True)
    rospy.Subscriber("/cloud_registered", PointCloud2, PointCloudCallback, queue_size=10)
    rospy.spin()


if __name__ == '__main__':
    # 实例化发送端
    ros_send = SocketSend()
    ros_recv = SocketRecv()

    # 单位为米
    OFFSET = 0.05  # 不可以大于Gap
    Y_MIN, Y_MAX = 0.95 - OFFSET, 1.2
    HEIGHT_MIN, HEIGHT_MAX = 0.0, 0.45
    DIAMETER = 0.25
    X_MIN, X_MAX = -(DIAMETER / 2 + OFFSET), (DIAMETER / 2 + OFFSET)
    GAP = 0.5
    CENTER_GAP = DIAMETER + GAP
    CENTER_POS = [-2 * CENTER_GAP, -CENTER_GAP, 0, CENTER_GAP, 2 * CENTER_GAP]
    LAYER_NUM = 3
    LAYERS = np.linspace(HEIGHT_MIN, HEIGHT_MAX, LAYER_NUM + 1)  # 分为layer num层

    layer_point_cnt = [[0 for _ in range(LAYER_NUM)] for _ in range(5)]  # 每一个框的每一层都要一个自己的列表
    filter_cnt = 0
    layer_point_cnt_single = [0 for _ in range(LAYER_NUM)]
    # ACT_THRESH = [10, 10, 10, 10, 10]
    ACT_THRESH = [10, 40, 40, 40, 10]
    FILTER_FRAME = 3
    FILTER_FRAME *= 4000

    NEAR_THRESH = 50

    # 修正为顺时针旋转90°的坐标系，修正后的坐标系用u、v、w表示
    DEV = True  # 调试模式
    # OFFSET 是相对于最终点的位置状态，未处理的世界坐标

    # 蓝场
    # OFFSET_POSE = [4.37, -9.40, -0.20]
    # OFFSET_EULER = [0, 0, -90]

    # 红场
    OFFSET_POSE = [-4.37, -9.40, -0.20]
    OFFSET_EULER = [0, 0, 90]

    print('Silo正在监听！！！')
    pc_listener()

'''
数据格式解读(加括号的为猜测)：

如果源发的就是PointCloud2格式：
示例：(-0.020999999716877937, 0.2919999957084656, 0.28299999237060547, 1.0, 0, 0, 1.7184756354382356e+18)
解读：(x, y, z, intensity[密度，反射律], (line), (反射次数/回波次数), 时间戳)

如果使用livox_repub转换CustomMsg为PointCloud2后的格式：
示例：(-0.125, 0.004000000189989805, 0.039000000804662704, 0.0, 0.0, 0.0, 3.0, 0.08767209202051163)
解读：(x, y, z, R, G, B, (回波次数), (line))

使用cloud_registered格式：
示例：(0.03984922543168068, 0.24823284149169922, 0.009252305142581463, 0.0, 0.0, 0.0, 23.0, 0.0)
解读：(x, y, z, R, G, B, Intensity[密度，反射律], (line))
'''
