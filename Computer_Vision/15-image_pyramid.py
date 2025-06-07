#-*-coding:utf-8 -*-
import cv2 as cv
import sys
msg = """


1 : 高斯金字塔
2 : 拉普拉斯金字塔

点击窗口右上角退出

"""

key=sys.argv[1]

print(msg)

#高斯金字塔
def pyramid_image(image):
    cv.imshow("yuan",image)
    level = 3#金字塔的层数
    temp = image.copy()#拷贝图像
    pyramid_images = []
    for i in range(level):
        dst = cv.pyrDown(temp)
        pyramid_images.append(dst)
        cv.imshow("pyramid"+str(i), dst)
        temp = dst.copy()
    return pyramid_images

#拉普拉斯金字塔
def lpls_image(image):
    image=cv.resize(image,(224,224))  #要求：拉普拉斯金字塔时，图像大小必须是2的n次方*2的n次方，不然会报错。
    pyramid_images = pyramid_image(image)
    level = len(pyramid_images)
    for i in range(level-1, -1, -1):#数组下标从0开始 i从金字塔层数-1开始减减
        if (i-1)<0:#原图
            expand = cv.pyrUp(pyramid_images[i])
            lpls = cv.subtract(image, expand)
            cv.imshow("lpls_%s" % i, lpls)
        else:
            expand = cv.pyrUp(pyramid_images[i], dstsize=pyramid_images[i-1].shape[:2])
            lpls = cv.subtract(pyramid_images[i-1], expand)
            cv.imshow("lpls_%s" % i, lpls)

if key == '1':  
    img = cv.imread("./image/1.png")
    pyramid_image(img)
    cv.waitKey(0)
    cv.destroyAllWindows()
elif key == '2' : 
    img = cv.imread("./image/1.png")
    lpls_image(img)
    cv.waitKey(0)
    cv.destroyAllWindows()