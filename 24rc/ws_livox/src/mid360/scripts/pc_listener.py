#!/usr/bin/python3
# coding=utf-8
# 专门把point cloud listener单拎出来，写个清晰的Demo。传下去
# Author: Ethan Lee
# Date: 2024.6.25
import rospy
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import threading
import socket


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


def PointCloudCallback(data):
    gen = point_cloud2.read_points(data, field_names=['x', 'y', 'z', 'intensity'], skip_nans=True)
    point_cnt = 0
    for p in gen:
        if 0 <= p[1] < 0.6 and -0.10 < p[2] < 0.05:  # y,z limit
            if -0.25 < p[0] < 0.25:  # x limit
                if 0 <= p[-1] < 255:  # intensity limit
                    # print("x : %f  y: %f  z: %f  intensity: %f" % (p[0], p[1], p[2], p[3]))
                    point_cnt += 1
                # print("x : %f  y: %f  z: %f  intensity: %f" % (p[0], p[1], p[2], p[3]))
    if point_cnt:
        ros_send.send('1')
        print('AVOID!')
    else:
        ros_send.send('0')


def pc_listener():
    rospy.init_node('pc_listener', anonymous=True)
    rospy.Subscriber("/cloud_registered_body", PointCloud2, PointCloudCallback, queue_size=10)  # 局部坐标点云
    rospy.spin()


if __name__ == '__main__':
    ros_send = SocketSend()
    pc_listener()
