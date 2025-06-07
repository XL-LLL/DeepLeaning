# encoding=utf-8
import time
from serial_connect import SerialConnect
import threading


class ArmController():

    def __init__(self):
        # 创建串口对象
        self.ser = SerialConnect()
        self.ser.on_connected_changed(self.serial_on_connected_changed)

        # 发送的数据队列
        self.msg_list = []
        # 是否连接成功
        self._isConn = True
        # 初始化机械臂状态参数
        self.init_arm_params()

    def init_arm_params(self):
        self.arm_state = 0
        self.running_flag = 0
        self.air_pump_state = 0
        self.electron_state = 0
        self.pos_x = 0
        self.pos_y = 0
        self.pos_z = 0

    # 串口连接状态回调函数
    def serial_on_connected_changed(self, is_connected):
        if is_connected:
            print("Com Connected")
            self._isConn = True
            self.ser.connect()
            self.ser.on_data_received(self.on_data_received)

            time.sleep(1)
            # 通信线程创建启动
            sendThread = threading.Thread(name="send_thread", target=self.send_msg)
            sendThread.setDaemon(True)
            sendThread.start()
            # 开启机械臂数据获取线程
            self.get_arm_data()
        else:
            self._isConn = False
            print("Com DisConnected")

    # 串口通信线程发送函数
    def send_msg(self):
        while True:
            if len(self.msg_list) > 0 and self._isConn:
                self.ser.write(self.msg_list[0])
                time.sleep(0.01)
                self.msg_list.remove(self.msg_list[0])

    # 串口数据包构建方法
    def generateCmd(self, cmd, data):
        buffer = [0] * (len(data) + 4)
        buffer[0] = 0x5A
        buffer[1] = cmd
        buffer[2] = len(data)
        for i in range(len(data)):
            buffer[3 + i] = data[i]
        # 校验位
        check = 0
        for i in range(len(buffer)):
            check += buffer[i]
        buffer[len(data) + 3] = check & 0xFF
        # for i in range(len(buffer)):
        #     print(hex(int(buffer[i])))
        return buffer

    # 复位
    def reset(self):
        data_body = [0] * 0
        state_pack = self.generateCmd(0x11, data_body)
        self.msg_list.append(state_pack)

    # 设置机械臂绝对位置
    def set_arm_absolute_pos(self, x, y, z):
        pos_x = x.to_bytes(4, byteorder='little', signed=True)
        pos_y = y.to_bytes(4, byteorder='little', signed=True)
        pos_z = z.to_bytes(4, byteorder='little', signed=True)
        pos_data = pos_x + pos_y + pos_z
        data_pack = self.generateCmd(0x1D, pos_data)
        self.msg_list.append(data_pack)

    # 控制气泵开关
    def set_air_pump_swicth(self, switch):
        if self.air_pump_state != switch:
            data_body = [0] * 1
            data_body[0] = switch & 0xff
            state_pack = self.generateCmd(0x14, data_body)
            self.msg_list.append(state_pack)

    # 控制电磁阀开关
    def set_electro_swicth(self, switch):
        if self.electron_state != switch:
            data_body = [0] * 1
            data_body[0] = switch & 0xff
            state_pack = self.generateCmd(0x15, data_body)
            self.msg_list.append(state_pack)

    # 获取机械臂数据线程
    def get_arm_data(self):
        print("start get_arm_data thread")
        self.get_arm_data_thread = threading.Thread(target=self.get_arm_data_callback, name="get_arm_data")
        self.get_arm_data_thread.setDaemon(False)
        self.get_arm_data_thread.start()

    # 获取机械臂数据
    def get_arm_data_callback(self):
        while True:
            if self._isConn:
                # 获取机械臂状态
                data_body = [0] * 0
                state_pack = self.generateCmd(0x10, data_body)
                self.msg_list.append(state_pack)
                time.sleep(0.1)

    # 获取机械臂数据读取回调函数
    def on_data_received(self, data):
        if len(data) == 24:
            if data[1] == 0x10:
                self.arm_state = data[3]
                self.runing_flag = data[4]
                pump_array = '{:08b}'.format(data[6])
                self.air_pump_state = pump_array[7]
                self.electron_state = pump_array[6]
                self.pos_x = int.from_bytes(data[11:15], byteorder='little', signed=True)
                self.pos_y = int.from_bytes(data[15:19], byteorder='little', signed=True)
                self.pos_z = int.from_bytes(data[19:23], byteorder='little', signed=True)
                # print("air_pump = {},ele = {}".format(self.air_pump_state,self.electron_state))
                # print("pos_x = {},pox_y = {},pox_z = {}".format(self.pos_x,self.pos_y,self.pos_z))

if __name__ == '__main__':
    connect = ArmController()
    time.sleep(2)
    connect.reset()
    flag = 0
    # connect.set_arm_absolute_pos(1960,1960,-2220)





