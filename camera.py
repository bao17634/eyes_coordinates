import cv2


def main():
    # 加载人脸检测联级文件
    face_detect = cv2.CascadeClassifier('D:\python_object\eyes_coordinates\haarcascade_frontalface_default.xml')

    # 加载眼部检测联级文件
    eye_detect = cv2.CascadeClassifier('D:\python_object\eyes_coordinates\haarcascade_eye.xml')

    # 打开摄像头
    camera = cv2.VideoCapture(0, cv2.CAP_DSHOW)

    while (True):
        # 从摄像头读取图片
        ret, image = camera.read()
        if not ret:
            print('failed to read camera data.')
            break

        # BRG转灰度图
        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

        # 检测人脸
        faces = face_detect.detectMultiScale(image_gray, scaleFactor=1.15, minNeighbors=5, minSize=(5, 5))
        for (x, y, width, height) in faces:
            # 根据检测结果绘制矩形框
            cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), 2)

        # 检测眼睛
        eyes = eye_detect.detectMultiScale(image_gray, scaleFactor=2.50, minNeighbors=3, minSize=(5, 5))

        print(eyes)

        for (x, y, width, height) in eyes:

            # 根据检测结果绘制矩形框
            cv2.rectangle(image, (x, y), (x + width, y + height), (255, 0, 0), -1)

        # 显示图片
        cv2.imshow('Face And Eye Detect', image)

        # 捕捉按键，如果是Q退出
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # 释放摄像头
    camera.release()

    # 销毁窗口
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
