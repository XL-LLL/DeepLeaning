# encoding=utf-8
from arm_controller import ArmController
import time


class ArmMove:
    """控制机械臂移动抓取"""
    def __init__(self):
        self.ac = ArmController()
        time.sleep(2)
        self.grip_flag = False  # 机械臂抓取标志
        self.reset()

    def grip(self):
        """控制机械臂末端执行器进行抓取：包括打开气泵和打开吸盘"""
        if self.grip_flag is False:
            self.ac.set_electro_swicth(0)  # 打开气泵
            self.ac.set_air_pump_swicth(1)  # 打开吸盘
            self.grip_flag = True
            print('i will grip')

    def loose(self):
        """控制机械臂末端执行器进行松开：包括关闭气泵和松开吸盘"""
        if self.grip_flag is True:
            self.ac.set_electro_swicth(1)
            self.ac.set_air_pump_swicth(0)
            self.grip_flag = False
            print('i am loose')

    def arm_pick(self, x, y, z):
        """控制机械臂末端执行器到达指定位置，然后执行抓取动作

        :param x: 机械臂x轴位置坐标，基座
        :param y: 机械臂y轴位置坐标，小臂
        :param z: 机械臂z轴位置坐标，大臂
        :return: 无
        """
        self.go_to(x, y, z)
        self.grip()
        self.go_to(x, y + 2500, z + 2000)

    def arm_release(self, x, y, z):
        """控制机械臂末端执行器到达指定位置，然后执行松开动作

        :param x: 机械臂x轴位置坐标，基座
        :param y: 机械臂y轴位置坐标，小臂
        :param z: 机械臂z轴位置坐标，大臂
        :return: 无
        """
        self.go_to(x, y, z)
        self.loose()
        self.go_to(0, 0, 0)

    def go_to(self, x, y, z):
        """单纯的控制机械臂末端执行器到达指定位置，不执行抓取或者松开操作

        :param x: 机械臂x轴位置坐标，基座
        :param y: 机械臂y轴位置坐标，小臂
        :param z: 机械臂z轴位置坐标，大臂
        :return: 无
        """
        self.ac.set_arm_absolute_pos(x, y, z)
        # 因为控制机械臂末端执行器到达指定位置时会有些许误差，所以如果实际到达位置与指定位置误差在上下2个单位之间，就视为到达
        for i in range(20):
            if (self.ac.pos_x in [x - 2, x - 1, x, x + 1, x + 2]) and (
                    self.ac.pos_y in [y - 2, y - 1, y, y + 1, y + 2]) and (
                    self.ac.pos_z in [z - 2, z - 1, z, z + 1, z + 2]):
                break
            time.sleep(0.5)

    def reset(self):
        """控制机械臂复位

        :return:
        """
        self.ac.reset()
        time.sleep(3)
        for i in range(40):
            # 如果机械臂的状态在[1, 2, 3, 4, 5]之间，代表机械臂正在复位过程中
            if self.ac.arm_state in [1, 2, 3, 4, 5]:
                time.sleep(0.5)


if __name__ == '__main__':
    am = ArmMove()
    am.arm_pick(0, 100, -3200)  # 物块抓取
    am.arm_release(9000, 2000, -1000)  # 物块松开