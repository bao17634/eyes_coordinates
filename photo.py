import cv2
import requests
from io import BytesIO
import numpy
from PIL import Image



def load_image(image_url):
    cap = cv2.VideoCapture(image_url)
    if cap.isOpened():
        ret, image = cap.read()

        cv2.imshow("原图", image)

        start_identify(image)


def start_identify(image):
    global face_x, face_y, face_width, face_height
    global eye_x, eye_y, eye_width, eye_height


    # 创建人脸检测级联分类器对象实例
    face_detect = cv2.CascadeClassifier('D:\python_object\eyes_coordinates\haarcascade_frontalface_default.xml')

    # 创建眼部检测级联分类器对象实例
    eye_detect = cv2.CascadeClassifier('D:\python_object\eyes_coordinates\haarcascade_eye.xml')

    # BRG转灰度图
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # 调用识别人脸接口
    # 参数说明，scaleFactor：图像缩放比例，可理解为相机的X倍镜，
    # minNeighbors：对特征检测点周边多少有效点同时检测，这样可避免因选取的特征检测点太小而导致遗漏
    # minSize：特征检测点的最小尺寸
    #返回的人脸对象的个数
    faces = face_detect.detectMultiScale(image_gray, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5))

    if len(faces) > 0:

        face_x, face_y, face_width, face_height = faces[0]

        print("脸部数据：", face_x, face_y, face_width, face_height)
        # # 检测眼睛
        eyes = eye_detect.detectMultiScale(image_gray, scaleFactor=1.1, minNeighbors=3, minSize=(5, 5))

        if len(eyes) > 0:
            eye_x, eye_y, eye_width, eye_height = eyes[0]

            print("眼睛数据：", eye_x, eye_y, eye_width, eye_height)

            cv2.rectangle(image, (face_x, eye_y), (face_x + face_width, eye_y + eye_height), (0, 0, 0), -1)

    # 显示图片
    cv2.imshow('Face And Eye Detect', image)

    # 销毁窗口
    cv2.waitKey()

if __name__ == '__main__':
    image_url = "https://gimg2.baidu.com/image_search/src=http%3A%2F%2Fdpic.tiankong.com%2Fda%2F2i%2FQJ6671395910.jpg&refer=http%3A%2F%2Fdpic.tiankong.com&app=2002&size=f9999,10000&q=a80&n=0&g=0n&fmt=jpeg?sec=1624699268&t=721fd6087c4f5f7313539138e53506b3"

    load_image(image_url)
