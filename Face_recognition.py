from operator import mod
import os
import numpy as np
import insightface
import cv2
from facenet import Facenet
from PIL import Image


# 实时人脸预测模块
def predict_image_realtime(image1, model, path_name='loss'):  
    # 确保没有超出边界
    if image1.shape[0] ==0 or image1.shape[1] ==0:
        print(1)
        return [-1]
    image1 = Image.fromarray(cv2.cvtColor(image1,cv2.COLOR_BGR2RGB))
    output1 = model.load_model_vector(image1)
    for dir_item in os.listdir(path_name):
        #从初始路径开始叠加，合并成可识别的操作路径
        if dir_item.endswith('.txt'):
            full_path = os.path.abspath(os.path.join(path_name, dir_item))
            output2 = np.loadtxt(full_path)
            probability = np.linalg.norm(output1-output2, axis=1)
            print(dir_item + ':', probability)
            if probability < 1:
                return [1, dir_item.split('.')[0]]
                    
    return [-1, -1]


# 利用文本存储特征向量，循环计算人脸识别 2.0
def predict_image_for_server_2(image1_path, model, path_name='loss'):  
    image1 = Image.open(image1_path)
    output1 = model.load_model_vector(image1)
    for dir_item in os.listdir(path_name):
        #从初始路径开始叠加，合并成可识别的操作路径
        if dir_item.endswith('.txt'):
            full_path = os.path.abspath(os.path.join(path_name, dir_item))
            print('full_path', full_path)
            output2 = np.loadtxt(full_path)
            loss = np.linalg.norm(output1-output2, axis=1)
            print(dir_item + ':', loss)
            if loss < 1:
                return [1, dir_item.split('.')[0]]
                    
    return [-1, -1]


# 循环便利图片预测 1.0
def predict_image_for_server_1(image1_path, model, path_name='img'):  
    image1 = Image.open(image1_path)
    for dir_item in os.listdir(path_name):
        #从初始路径开始叠加，合并成可识别的操作路径
        full_path = os.path.abspath(os.path.join(path_name, dir_item))
        if dir_item.endswith('.jpg') or dir_item.endswith('.png'):
            image2 = Image.open(full_path)
            probability = model.detect_image(image1,image2)
            print(dir_item + ':', probability)
            if probability < 1:
                return [1, dir_item.split(';')[0]]
                    
    return [-1, -1]


# 摄像头处理类
class Handler:
    def __init__(self):
        self.detector = insightface.model_zoo.get_model('retinaface_mnet025_v2')
        self.detector.prepare(ctx_id=-1)


if __name__ == '__main__':

    model = Facenet()
                     
    color = (0, 255, 0)
    
    cap = cv2.VideoCapture(0)
    
    handler = Handler()

    while True:
        ret, frame = cap.read()
        font = cv2.FONT_HERSHEY_SIMPLEX

        bboxes, _ = handler.detector.detect(frame)

        # print(bboxes.shape[0])
        if bboxes.shape[0] == 0:
            cv2.putText(frame, "no face", (20, 350), font, 0.8, (0, 0, 0), 1, cv2.LINE_AA)
        else:
            for i in range(bboxes.shape[0]):
                faceRect = bboxes[i]
                x, y, w, h, _ = faceRect.astype(np.int32)
                
                #截取脸部图像提交给模型识别这是谁
                image = frame[y - 10: h + 10, x - 10: w + 10]
                res = predict_image_realtime(model=model, image1=image)


                if res[0] == -1:
                    cv2.rectangle(frame, (x - 10, y - 10), (w + 10,h + 10), color, thickness = 2)
                    #文字提示是谁
                    cv2.putText(frame,'NULL', 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽  
                else:
                    cv2.rectangle(frame, (x - 10, y - 10), (w + 10, h + 10), color, thickness = 2)
                    #文字提示是谁
                    cv2.putText(frame,res[1], 
                                (x + 30, y + 30),                      #坐标
                                cv2.FONT_HERSHEY_SIMPLEX,              #字体
                                1,                                     #字号
                                (255,0,255),                           #颜色
                                2)                                     #字的线宽     
        
        cv2.imshow("face recognition", frame)
        
        k = cv2.waitKey(10)

        if k & 0xFF == ord('q'):
            break
 
    cap.release()
    cv2.destroyAllWindows()