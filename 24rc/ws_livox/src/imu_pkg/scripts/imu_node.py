#!/usr/bin/python3
# coding=utf-8
import rospy
from sensor_msgs.msg import Imu
import tf.transformations as tf_trans

def IMUCallback(data):
    # print(data)

    # if data.orientation_coveriance[0] < 0:
    #     rospy.logwarn("No orientation data")
    #     return

    # 使用TF将四元数转换为欧拉角
    quaternion = [
        data.orientation.x,
        data.orientation.y,
        data.orientation.z,
        data.orientation.w
    ]
    (roll, pitch , yaw) = tf_trans.euler_from_quaternion(quaternion)
    # 打印欧拉角
    # rospy.loginfo("Euler Angles: Roll: %f, Pitch: %f, Yaw: %f", roll, pitch, yaw)

    # 打印角速度
    angular_velocity = [
        data.angular_velocity.x,
        data.angular_velocity.y,
        data.angular_velocity.z
    ]
    # rospy.loginfo("Angular Velocity: x: %f, y: %f, z: %f", angular_velocity[0], angular_velocity[1], angular_velocity[2])

    # 打印线性加速度
    linear_acceleration = [
        data.linear_acceleration.x,
        data.linear_acceleration.y,
        data.linear_acceleration.z
    ]
    # rospy.loginfo("Linear Acceleration: x: %f, y: %f, z: %f", linear_acceleration[0], linear_acceleration[1], linear_acceleration[2])

    # 统一输出三个角度
    rospy.loginfo("Euler Angles: Roll: %f, Pitch: %f, Yaw: %f \nAngular Velocity: x: %f, y: %f, z: %f \nLinear Acceleration: x: %f, y: %f, z: %f",
                  roll, pitch, yaw, angular_velocity[0], angular_velocity[1], angular_velocity[2], linear_acceleration[0], linear_acceleration[1], linear_acceleration[2])

def imu_listener():
    rospy.init_node('imu_node', anonymous=True)
    rospy.Subscriber("/livox/imu", Imu, IMUCallback, queue_size=10)
    rospy.spin()

if __name__ == '__main__':
    imu_listener()
