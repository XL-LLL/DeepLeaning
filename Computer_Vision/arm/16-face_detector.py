#!/usr/bin/env python
# -*- coding: utf-8 -*-
import rospy
import cv2
import numpy as np
from sensor_msgs.msg import Image
import cv_bridge


class FaceDetector:
    def __init__(self):
        rospy.on_shutdown(self.cleanup)

        # 创建cv_bridge
        self.bridge = cv_bridge.CvBridge()
        self.image_pub = rospy.Publisher("cv_bridge_image", Image, queue_size=1)
        # self.image_sub = rospy.Subscriber("/camera/color/image_raw", Image, self.image_callback, queue_size=1)
        self.image_sub = rospy.Subscriber("/camera/rgb/image_rect_color", Image, self.image_callback, queue_size=1)

        # 获取haar特征的级联表的XML文件，文件路径在launch文件中传入
        cascade_1 = rospy.get_param("~cascade_1", "./cascades/haarcascade_frontalface_alt.xml")
        cascade_2 = rospy.get_param("~cascade_2", "./cascades/haarcascade_eye_tree_eyeglasses.xml")
	#haarcascade_profileface.xml

        # 使用级联表初始化haar特征检测器
        self.cascade_1 = cv2.CascadeClassifier(cascade_1)
        self.cascade_2 = cv2.CascadeClassifier(cascade_2)

        # 设置级联表的参数，优化人脸识别，可以在launch文件中重新配置
        self.haar_scaleFactor = rospy.get_param("~haar_scaleFactor", 1.2)
        self.haar_minNeighbors = rospy.get_param("~haar_minNeighbors", 2)
        self.haar_minSize = rospy.get_param("~haar_minSize", 40)
        self.haar_maxSize = rospy.get_param("~haar_maxSize", 60)
        self.color = (50, 255, 50)

    def image_callback(self, data):
        # 使用cv_bridge将ROS的图像数据转换成OpenCV的图像格式
        cv_image = self.bridge.imgmsg_to_cv2(data,"bgr8")
        frame = np.array(cv_image, dtype=np.uint8)

        # 创建灰度图像
        grey_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # 创建平衡直方图，减少光线影响
        grey_image = cv2.equalizeHist(grey_image)

        # 尝试检测人脸
        faces_result = self.detect_face(grey_image)

        # 在opencv的窗口中框出所有人脸区域
        if len(faces_result) > 0:
            for face in faces_result:
                x, y, w, h = face
                cv2.rectangle(cv_image, (x, y), (x + w, y + h), self.color, 2)

        # 将识别后的图像转换成ROS消息并发布
        self.image_pub.publish(self.bridge.cv2_to_imgmsg(cv_image, "bgr8"))

    def detect_face(self, input_image):
        # 首先匹配正面人脸的模型
        if self.cascade_1:
            faces = self.cascade_1.detectMultiScale(input_image,
                                                    self.haar_scaleFactor,
                                                    self.haar_minNeighbors,
                                                    cv2.CASCADE_SCALE_IMAGE,
                                                    (self.haar_minSize, self.haar_maxSize))

        # 如果正面人脸匹配失败，那么就尝试匹配戴眼镜人脸的模型
        if len(faces) == 0 and self.cascade_2:
            faces = self.cascade_2.detectMultiScale(input_image,
                                                    self.haar_scaleFactor,
                                                    self.haar_minNeighbors,
                                                    cv2.CASCADE_SCALE_IMAGE,
                                                    (self.haar_minSize, self.haar_maxSize))

        return faces

    def cleanup(self):
        print("强制结束程序。。")
        cv2.destroyAllWindows()


if __name__ == '__main__':
    try:
        # 初始化ros节点
        rospy.init_node("face_detector")
        follower = FaceDetector()
        rospy.loginfo("人脸识别已经启动。。。")
        rospy.loginfo("请打开opencv节点订阅消息。。。")
        rospy.spin()
    except KeyboardInterrupt:
        print("强制结束程序。。")
        cv2.destroyAllWindows()
