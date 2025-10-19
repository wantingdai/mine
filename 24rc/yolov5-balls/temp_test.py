# Author: Ethan Lee
# 2024/7/5 下午9:49

import math


def trans_cond(body_x, body_y, odom_x, odom_y, odom_yaw):  # 修正坐标（局部转世界）
    tmp = math.sqrt(body_x ** 2 + body_y ** 2)
    theta = math.asin(body_y / tmp) if body_x >= 0 else -math.asin(body_y / tmp)  # Test
    odom_yaw = math.radians(odom_yaw)
    rou = theta + odom_yaw
    r_x = tmp * math.cos(rou) if body_x >= 0 else -tmp * math.cos(rou)
    r_y = tmp * abs(math.sin(rou))  # 铁是正的
    world_x = r_x + odom_x
    world_y = r_y + odom_y
    return world_x, world_y


if __name__ == '__main__':
    body_x = 7.6
    body_y = 10
    u, v = trans_cond(body_x, body_y, -1.4, -5.1, -18.4)
    print(u, v)
