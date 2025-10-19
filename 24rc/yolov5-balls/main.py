"""
Principal Point: 970.2665,553.7746
IntrinsicMatrix: # 内参矩阵
[1351.8822, 0, 970.2665;
 0, 1349.8975, 553.7746;
 970.2665, 553.7746, 1]
"""
import numpy as np
import cv2
import serial.tools.list_ports
import get_ball
import silo
from apply_on_R2 import loadPolicy
import rc_utils
import os
import ros_comm
import time

# TODO: silo顶层模型加练，ground ball 加练的模型得用上，新摄像头模型加练, 看球加上深度滤除

if __name__ == '__main__':

    """
    @ 以下是四个步骤的判断语句，初始状态均为0表示关闭使能
    """
    # 这个位置的各个参数都是标志位，如果想专修某个模块的部分，可以直接在这里对他们操作，将他们置1
    # 高台看底下红球点位的部分，修改Start = 1
    # 机械臂抓球点位，修改ON_1 = 1
    # 决策框的部分，修改ON_2 = 1
    SIDE = 'r'  # 'r'和‘b'
    # SIDE = 'b'
    MAKERS = ['START', 'ON_1', 'ON_2', 'OFF_1', 'OFF_2', 'Catch_Ball', 'Finish_Ball', 'RESET', 'Send_INT', 'BACK',
              'GETE', 'RETRY', 'LIDAR', 'SETUP', 'OK', 'ERROR', 'CLICK', 'BEGIN', 'END', 'EYE']
    retry = False
    setup = False
    proc_ctrl = {m: False for m in MAKERS}  # 用bool列表进行流程标志位控制
    sent = False  # 用于处理博弈时的结果（只发送有效值一次）
    obs_avoid_time_start = 0
    decision = 9
    CAM_WARMUP_FRAMES = 6
    # CAM_WARMUP_FRAMES = 0
    cam_warmup_cnt = 0
    # TODO: 以下优化掉
    model = loadPolicy()
    WIDTH = 640
    HEIGHT = 360
    ####################

    '''手动调试'''
    # proc_ctrl['START'] = True
    # proc_ctrl['ON_1'] = True
    # proc_ctrl['ON_2'] = True
    # proc_ctrl['ON_3'] = True
    # proc_ctrl['Catch_Ball'] = True
    # proc_ctrl['LIDAR'] = True  # Start at Zone 3
    # proc_ctrl['RETRY'] = True
    # proc_ctrl['SETUP'] = True
    # proc_ctrl['OK'] = True
    # setup = True
    ''''''

    '''@相机初始化'''
    pipeline, align = rc_utils.init_realsense(WIDTH, HEIGHT)  # 深度相机初始化
    # cap2 = cv2.VideoCapture(0)  # 要么是6，要么是0
    # cap2.set(3, WIDTH)
    # cap2.set(4, HEIGHT)

    # ROS 通信模块初始化
    ros_recver = ros_comm.ROSMsgRecver()
    ros_sender = ros_comm.ROSMsgSender()
    # 不知道为什么silo第一次识别不到，预执行一次
    # ros_sender.send('1,0,0,0')
    # time.sleep(0.3)
    # ros_sender.send('0,0,0,0')

    # 串口配置
    # find_serial()
    serial_port = serial.Serial(port="/dev/ttyUSB0", baudrate=115200,
                                timeout=0.2)  # port表示串口(串口是找串口函数中得到的串口)，115200表示波特率,0.2表示时间间隔
    ser = rc_utils.SerialCtrl(serial_port, ros_recver)
    ser.clear_log()

    # 取球实例化
    get = get_ball.GetBall(SIDE)  # 取球
    dec = silo.Silo(SIDE)  # decide
    print('主程序主线程正在运行！！！')

    while True:
        # proc_ctrl['START'] = True  # Temp
        # proc_ctrl['ON_1'] = True  # Temp
        # proc_ctrl['ON_2'] = True  # Temp
        """
        @ 以下是串口对数据进行实时接收
        """
        msg = ser.receive_data()
        if msg is not None:
            print(f'msg type={type(msg)}, msg={msg}')
        if msg in proc_ctrl.keys():
            proc_ctrl[msg] = True

        if proc_ctrl['SETUP']:  # 摁下一次按键
            # proc_ctrl['CLICK'] = not (proc_ctrl['CLICK'])
            proc_ctrl['CLICK'] = True
            proc_ctrl['OK'] = False
            proc_ctrl['SETUP'] = False

        if proc_ctrl['CLICK'] and proc_ctrl['OK'] and proc_ctrl['BEGIN'] is False:  # 如果是奇数次按下按键
            os.system('~/r2_start.sh')  # 启动
            setup = rc_utils.setup_self_check(ser, pipeline)  # 视觉部分进行自检
            proc_ctrl['BEGIN'] = True  # 用于区分偶数次按下按键和0次按下按键

        # if proc_ctrl['CLICK'] is False and proc_ctrl['BEGIN'] is True:  # 如果是偶数次且不为0次按下按键
        #     os.system('~/end_r2.sh')  # 关闭
        #     proc_ctrl['BEGIN'] = False

        if not setup:  # 视觉部分自检错误则重新来过
            continue

        if setup is False:  # 视觉部分自检错误或电控部分自检错误则重新自检
            continue

        # if proc_ctrl['LIDAR']:
        #     os.system('gnome-terminal -e "roslaunch fast_lio mapping_mid360.launch"')
        #     proc_ctrl['LIDAR'] = False

        if proc_ctrl['RESET']:  # 自动复位
            os.system('clear')
            ser.clear_log()
            print('执行复位！')

            proc_ctrl = {m: False for m in MAKERS}
            proc_ctrl['BEGIN'] = True
            get.reset_yzy_idea()
            get.reset_find_and_get()
            dec.reset()
            dec.silo_our_side_cnt = np.array([0 for _ in range(5)])
            cv2.destroyAllWindows()

        if proc_ctrl['END']:
            os.system('~/end_r2.sh')  # 关闭
            proc_ctrl['END'] = False

        if proc_ctrl['EYE']:
            os.system('~/r2_start.sh')  # start
            proc_ctrl['EYE'] = False

        """
        @ 以下是对深度摄像头进行设置和读取操作
        """
        if cam_warmup_cnt < CAM_WARMUP_FRAMES:  # 预热相机，减少色差和亮度影响
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            cam_warmup_cnt += 1
            continue

        frames = pipeline.wait_for_frames()
        aligned_frames = align.process(frames)

        # Get aligned frames
        aligned_depth_frame = aligned_frames.get_depth_frame()
        color_frame = aligned_frames.get_color_frame()

        # Validate that both frames are valid
        if not aligned_depth_frame or not color_frame:
            print('NO aligned_depth_frame or color_frame')
            continue

        # Convert images to numpy arrays
        depth_image = np.asanyarray(aligned_depth_frame.get_data())
        color_image = np.asanyarray(color_frame.get_data())

        """
        @ 以下是对机器人进行操作
        """
        if proc_ctrl['RETRY']:  # 重试
            os.system('clear')
            ser.clear_log()
            print('执行重试！')

            proc_ctrl = {m: False for m in MAKERS}
            get.reset_yzy_idea()
            get.reset_find_and_get()
            dec.reset()
            dec.silo_our_side_cnt = np.array([0 for _ in range(5)])
            cv2.destroyAllWindows()
            retry = True

            # proc_ctrl['ON_2'] = True  # Temp

        # 机器人此时朝下看球
        if proc_ctrl['START'] and proc_ctrl['ON_1'] is False:
            color_image = rc_utils.depth_filter(0, 8, depth_image, color_image)
            out_loop = get.yzy_idea_static_filter(color_image, aligned_depth_frame, ser, decision, ros_recver.read_recv('o'))  # 有球为真
            if not out_loop:
                continue
            # get.yzy_idea(color_image, aligned_depth_frame, ser, decision, ros_recver.read_recv('o'))
            # dtd_check_down = get.dtd_idea_check(camera2_img, proc_ctrl['ON_3'], WIDTH)  # 有球为真
            # dtd_check_down = True
            # dtd_check_up = True
            # proc_ctrl['Send_INT'] = not (dtd_check_up or dtd_check_down)  # 都无球为真
            proc_ctrl['Finish_Ball'] = False

            # To make following codes work, the camera shoul  d get down after vision send the position
            # if get.get_downhill(ros_recver.read_recv('o')):  # to stop sending
            #     proc_ctrl['Send_INT'] = False  # 复位INT
            #     cam_warmup_cnt = 0
            #     proc_ctrl['GETS'] = False
            #     proc_ctrl['START'] = False
            #     get.reset_yzy_idea()

        # if proc_ctrl['Send_INT']:  # 中断，重试
        #     proc_ctrl['START'] = False
        #     proc_ctrl['ON_3'] = False
        #     get.reset_dtd_idea()
        #     ser.send_interrupt()
        #     cv2.destroyAllWindows()

        # if proc_ctrl['OFF_1']:
        #     proc_ctrl['ON_1'] = True  # 应对可能出现的OFF_1没来但是拿到球并且看到一眼超阈的情况
        # 机器人此时没有抓取到球

        if proc_ctrl['ON_1']:  # 确保这部分的ON_1进入是能够导向RETRY部分的ON_1避免逻辑错误
            if proc_ctrl['START']:
                proc_ctrl['Send_INT'] = False  # 复位INT
                cam_warmup_cnt = 0
                proc_ctrl['GETS'] = False
                proc_ctrl['START'] = False
                get.reset_yzy_idea()
                continue  # 更新图像
                # cv2.destroyAllWindows()
            """
            @ 对深度摄像头进行设置
            """
            # depth_colormap = cv2.applyColorMap(cv2.convertScaleAbs(depth_image, alpha=0.03), cv2.COLORMAP_JET)
            # img = rc_utils.depth_filter(0, 6.8, depth_image, color_image)
            img = color_image

            if not proc_ctrl['OFF_1']:  # 没有接收到捡球结束的命令
                if proc_ctrl['BACK']:  # 除了球超阈会给出BACK指令之外，如果连续抓四次没有抓到球也会直接给BACK指令
                    get.reset_yzy_idea()
                    decision = 9  # 只使用一次决策，即如果从球框前抓球没有抓到则退回到高台重新看球，不进入决策部分

                    obs_avoid_time = time.time() - obs_avoid_time_start
                    if not ros_recver.read_recv('s') or obs_avoid_time > 7:  # 借用silo接收避障信息
                        ser.send_retry()  # 持续发送E
                    elif ros_recver.read_recv('s') and obs_avoid_time <= 7:
                        ser.send_obs_avoid()  # 避障

                    if proc_ctrl['GETE']:  # 接收到GETE，不再发送
                        proc_ctrl['ON_1'] = False  # ON_1给False会让R2只等待START的命令
                        proc_ctrl['BACK'] = False  # 不再发E
                        proc_ctrl['GETE'] = False  # 判断位的状态重置
                        cv2.destroyAllWindows()
                if proc_ctrl['BACK'] is False and proc_ctrl['ON_1'] is True:  # 确保发送的球的坐标是在ON_1状态下发送的
                    # 避免在收到GETE之后回退的过程中又看到了红球导致卡住
                    success = get.find_and_get(img, aligned_depth_frame, ser)  # 暂存区捡球
                    if success is False:  # ROI内一旦没有球
                        proc_ctrl['BACK'] = True  # BACK标志位给True，表示回退
                        obs_avoid_time_start = time.time()
                        ser.send_retry()  # change chassis mode

            elif proc_ctrl['OFF_1']:  # 接收到了捡球结束的命令
                print("Get Ball!\r\n")
                cv2.destroyAllWindows()
                ser.send_transition()  # 过渡状态

                get.reset_find_and_get()

                proc_ctrl['Catch_Ball'] = True  # 标志位，进入过渡，让底盘能够接受命令
                proc_ctrl['OFF_1'] = False  # 将OFF_1重新设置为0
                proc_ctrl['ON_1'] = False  # 将ON_1重新设置为0
                obs_avoid_time_start = time.time()
                ser.send_transition()  # change chassis mode
                # cam_warmup_cnt = 0

        if proc_ctrl['Catch_Ball']:
            obs_avoid_time = time.time() - obs_avoid_time_start
            if not ros_recver.read_recv('s') or obs_avoid_time > 7:  # 借用silo接收避障信息
                ser.send_transition()  # 持续发送数据表示取到球，本质上是用于做过渡的一个通信命令
            elif ros_recver.read_recv('s') and obs_avoid_time <= 7:
                ser.send_obs_avoid()  # 避障
            proc_ctrl['ON_2'] = dec.arrive_dec_pos(ros_recver.read_recv('o'))  # 判断是否到达观察范围
            # print(f'ON_2: {proc_ctrl["ON_2"]}')

        if proc_ctrl['ON_2']:  # R2已经跑到谷仓对应点位
            # 滤除不满足深度要求的点（以黑色填充）
            # 设置深度阈值 (单位为米)
            img = rc_utils.depth_filter(0, 5, depth_image, color_image)
            # img = None
            if proc_ctrl['Catch_Ball']:
                proc_ctrl['Catch_Ball'] = False  # 关闭持续发送
                cam_warmup_cnt = 0
                continue  # update images
            if not proc_ctrl['OFF_2']:  # 此处表示R2还没放完球
                # cv2.imshow('img', img)
                if not sent:
                    # noinspection PyUnboundLocalVariable
                    while True:  # TODO: 判断里程计到达位置后，自动进行决策。方法放在silo中
                        # TODO: 收到清零信号后开始运行fast_lio
                        if retry:  # 重试
                            # decision = dec.YXS()
                            decision = dec.make_retry_decision(img)
                            retry = False
                        elif not retry:
                            # decision = dec.YXS()
                            decision = dec.make_decision(img, ros_recver, ros_sender)
                        if decision != -1:
                            break
                    # if decision == 9:  # 如果静止不动则持续发送(如果策略有等待)
                    #     pass
                    # else:  # 否则发送一次数据结束
                    #     sent = True
                    sent = True
                ser.send_decision(decision)  # 不改变值，持续发送
            elif proc_ctrl['OFF_2']:  # 此处表示R2已经放完球
                cv2.destroyAllWindows()
                print("Finish!")
                proc_ctrl['Finish_Ball'] = True
                proc_ctrl['ON_2'] = False
                proc_ctrl['OFF_2'] = False
                sent = False

                # cam_warmup_cnt = 0

        if proc_ctrl['Finish_Ball']:
            ser.send_decision(5)  # 同上，作为过渡位
            # proc_ctrl['Start'] = True

        if cv2.waitKey(1) == 27:
            break

    pipeline.stop()
    cv2.destroyAllWindows()
