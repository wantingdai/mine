# 从ros中获取信息
import socket
import threading
import time


class ROSMsgRecver:
    def __init__(self):
        # TODO: 用父类优化一下
        # 里程计：
        self.serversocket_odom = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 socket 对象
        self.host_odom = socket.gethostname()  # 获取本地主机名
        self.port_odom = 12345  # 设置端口号
        self.serversocket_odom.bind((self.host_odom, self.port_odom))  # 绑定端口
        self.serversocket_odom.listen(5)  # 设置最大连接数，超过后排队
        self.recver_odom_thread = threading.Thread(target=self.recver_odom)
        self.recver_odom_thread.start()
        self.odom_vals = None

        # 框数据：
        self.serversocket_silo = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 socket 对象
        self.host_silo = socket.gethostname()  # 获取本地主机名
        self.port_silo = 23456  # 设置端口号
        self.serversocket_silo.bind((self.host_silo, self.port_silo))  # 绑定端口
        self.serversocket_silo.listen(5)  # 设置最大连接数，超过后排队
        self.recver_silo_thread = threading.Thread(target=self.recver_silo)
        self.recver_silo_thread.start()
        self.silo_inside_num = []

    def recver_odom(self):  # 建立客户端连接
        while True:
            try:
                clientsocket, addr = self.serversocket_odom.accept()
                msg = clientsocket.recv(1024)  # 1024是一个参数，用于指定recv方法一次性最多接收的数据量，单位是字节
                ros_odom_msg = msg.decode('utf-8')
                clientsocket.close()
                self.odom_vals = self.split_ros_msg(ros_odom_msg)
            except KeyboardInterrupt:
                clientsocket.close()

    def recver_silo(self):  # 建立客户端连接
        while True:
            try:
                clientsocket, addr = self.serversocket_silo.accept()
                msg = clientsocket.recv(1024)  # 1024是一个参数，用于指定recv方法一次性最多接收的数据量，单位是字节
                ros_silo_msg = msg.decode('utf-8')
                self.silo_inside_num = eval(ros_silo_msg)
                clientsocket.close()
            except KeyboardInterrupt:
                clientsocket.close()


    def read_recv(self, mode):
        while True:
            try:
                if mode == 'o':
                    if self.odom_vals is not None:
                        return self.odom_vals
                    print('发送端未启动！')
                elif mode == 's':
                    if self.silo_inside_num is not None:
                        # return self.silo_inside_num  # Temp danxiangsai
                        return False  # No need obs
                    print('发送端未启动！')
            except TypeError:
                continue

    @staticmethod
    def split_ros_msg(msg):
        variables = msg.split(',')
        '''
        里程计：f'{pos[0]},{pos[1]},{pos[2]},{roll},{pitch},{yaw}'
        框内球数：(list) [0, 0, 0, 0, 0]
        '''
        return [float(variable) for variable in variables]


class ROSMsgSender:
    def __init__(self):
        # 发出数据
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 socket 对象
        self.host_send = socket.gethostname()  # 获取本地主机名
        self.port_send = 34567  # 设置端口号

    def send(self, ros_msg):
        self.serversocket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)  # 创建 socket 对象
        try:
            self.serversocket.connect((self.host_send, self.port_send))  # 连接服务，指定主机和端口
            self.serversocket.send(ros_msg.encode('utf-8'))  # 发送数据
        except ConnectionRefusedError:
            print('接收端未启动！')
        finally:
            self.serversocket.close()


if __name__ == '__main__':
    recver = ROSMsgRecver()
    sender = ROSMsgSender()
    while True:
        # time.sleep(1)
        odom = recver.read_recv('o')
        x = odom[0]
        y = odom[1]
        yaw = odom[-1]
        # print(f'YAW = {yaw:+07.2f}')
        msg = f'1,{x},{y},{yaw}'
        # msg = f'1'
        # msg = f'1,0,0,0'
        sender.send(msg)
        # time.sleep(0.5)
        # sender.send('0')
        print(f'SILO = {recver.read_recv("s")}')
        # time.sleep(5)
        # time.sleep(1)
        # sender.send('2, 2')  # command = 2, dec = 2
