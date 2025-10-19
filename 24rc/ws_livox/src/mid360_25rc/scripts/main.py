#!/usr/bin/python3
# coding=utf-8
import rospy
import math
from nav_msgs.msg import Odometry
import tf.transformations as tf_trans
from geometry_msgs.msg import Pose, Point, Quaternion
from math import pi
import tf  

Init_Pose = [0.0, 0.0, 0.0]  # 雷达起始位置
Init_Quaternion = [0.0, 0.0, 0.0]  # 雷达起始位姿

class LivoxOdometryReader:
    def __init__(self, topic_name="/Odometry"):
        rospy.init_node('livox_odometry_reader', anonymous=True)
        
        # 订阅 Odometry 话题
        self.odom_sub = rospy.Subscriber(topic_name, Odometry, self.odom_callback)
        
        # 存储当前最新的位姿数据
        self.current_pose = Pose()
        self.current_position = Point()
        self.current_orientation = Quaternion()

        # 初始化 TF 转换器（用于四元数转欧拉角）
        self.tf_listener = tf.TransformListener()

    def odom_callback(self, msg):
        # 提取位置 (x, y, z)
        self.current_position = msg.pose.pose.position
        
        # 提取姿态（四元数形式）
        self.current_orientation = msg.pose.pose.orientation
        
        # 存储完整 Pose 数据
        self.current_pose = msg.pose.pose

    def get_position(self):
        # 手动将雷达初始位置置为0
        self.current_position.x += Init_Pose[0]
        self.current_position.y += Init_Pose[1]
        self.current_position.z += Init_Pose[2]
        return self.current_position

    def get_pose(self):
        return self.current_pose

    def get_orientation_quaternion(self):
        return self.current_orientation

    def get_orientation_euler(self):
        # tf转换四元数成欧拉角
        (roll, pitch, yaw) = tf.transformations.euler_from_quaternion(
            [self.current_orientation.x,
             self.current_orientation.y,
             self.current_orientation.z,
             self.current_orientation.w]
        )
        # 欧拉角转换成角度,通过初始角度使其手动置为0
        roll = roll * 180 / pi + Init_Quaternion[0]
        pitch = pitch * 180 / pi + Init_Quaternion[1]
        yaw = yaw * 180 / pi + Init_Quaternion[2]
        return (roll, pitch, yaw)


# -------------------------- 主函数 --------------------------
if __name__ == "__main__":
    flag = True

    # 实例化
    odom_reader = LivoxOdometryReader(topic_name="/Odometry")
    
    # 设置循环频率（10Hz）
    rate = rospy.Rate(10)
    
    while flag is True:
        # 获取数据
        position = odom_reader.get_position()
        orientation_euler = odom_reader.get_orientation_euler()
        
        # 打印位置和欧拉角
        print("\n当前位置:")
        print(f"  X: {position.x:.3f} m")
        print(f"  Y: {position.y:.3f} m")
        print(f"  Z: {position.z:.3f} m")
        
        print("\n当前姿态:")
        print(f"  Roll:  {orientation_euler[0]:.3f}")
        print(f"  Pitch: {orientation_euler[1]:.3f}")
        print(f"  Yaw:   {orientation_euler[2]:.3f}")
        
        rate.sleep()

if __name__ == '__main__':
    Odom_listener()