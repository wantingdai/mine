#!/usr/bin/python3
# coding=utf-8
import os

import rospy
from nav_msgs.msg import Odometry
import tf.transformations as tf_trans
from math import pi
import serial
import socket


# ser = serial.Serial(port='/dev/ttyUSB0', baudrate=115200, bytesize=8, parity='N', stopbits=1, timeout=2,rtscts=True,dsrdtr=True)
# ser.isOpen()

def OdomCallback(data):
    pos = [
        data.pose.pose.position.x,
        data.pose.pose.position.y,
        data.pose.pose.position.z
    ]

    quaternion = [
        data.pose.pose.orientation.x,
        data.pose.pose.orientation.y,
        data.pose.pose.orientation.z,
        data.pose.pose.orientation.w
    ]
    (roll, pitch, yaw) = tf_trans.euler_from_quaternion(quaternion)
    roll, pitch, yaw = rad_degree_tf(roll, pitch, yaw)

    if MODE == 'r':  # 修正为顺时针旋转90°的坐标系，修正后的坐标系用u、v、w表示
        for i in range(3):  # 进行预设偏移
            pos[i] += OFFSET_POSE[i]

        roll += OFFSET_EULER[0]
        pitch += OFFSET_EULER[1]
        yaw += OFFSET_EULER[2]

        tmp = pos[0]
        pos[0] = -pos[1]  # u = -y
        pos[1] = tmp  # v = x

        tmp = pitch
        pitch = roll
        roll = tmp

    elif MODE == 'b':  # 修正为ni时针旋转90°的坐标系，修正后的坐标系用u、v、w表示
        for i in range(3):  # 进行预设偏移
            pos[i] += OFFSET_POSE[i]

        roll += OFFSET_EULER[0]
        pitch += OFFSET_EULER[1]
        yaw += OFFSET_EULER[2]

        tmp = pos[0]
        pos[0] = pos[1]  # u = y
        pos[1] = -tmp  # v = -x

        tmp = pitch
        pitch = roll
        roll = -tmp

    loginfo = "\n里程计数据: Position: x: %s, y: %s, z: %s \n 角度: ROLL: %s, PITCH: %s, YAW: %s" % (
    pos[0], pos[1], pos[2], roll, pitch, yaw)
    # os.system('clear')  # Attention: 加了电脑会卡
    # rospy.loginfo(loginfo)
    print(loginfo)

    ros_msg = f'{pos[0]},{pos[1]},{pos[2]},{roll},{pitch},{yaw}'
    # 把发送消息出去
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 socket 对象
    host = socket.gethostname()  # 获取本地主机名
    port = 12345  # 设置端口号
    try:
        s.connect((host, port))  # 连接服务，指定主机和端口
        s.send(ros_msg.encode('utf-8'))  # 发送数据
        s.close()
    except ConnectionRefusedError:
        rospy.loginfo('接收端未启动！')

    # serialinfo = "\r\nx=%s  y=%s   angle=%s" % (pos[0], pos[1], yaw)
    # ser.write(serialinfo.encode('gbk'))


def listen_to_odometry():
    rospy.init_node('odometry_listener')
    rospy.Subscriber("/Odometry", Odometry, OdomCallback)
    rospy.spin()


def rad_degree_tf(roll, pitch, yaw):
    roll = roll * 180 / pi
    pitch = pitch * 180 / pi
    yaw = yaw * 180 / pi
    return roll, pitch, yaw


if __name__ == '__main__':
    MODE = 'r'  # 模式 r b DEV
    # OFFSET 是相对于最终点的位置状态，未处理的世界坐标

    if MODE == 'b':
        # 蓝场
        OFFSET_POSE = [4.37, -9.40, -0.20]
        OFFSET_EULER = [0, 0, -90]
    elif MODE == 'r':
        # 红场
        OFFSET_POSE = [-4.37, -9.40, -0.20]
        OFFSET_EULER = [0, 0, 90]

    listen_to_odometry()
