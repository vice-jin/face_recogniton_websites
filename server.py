#!/usr/bin/env python3
from facenet import Facenet
import cv2
import os
import numpy as np
from scipy.misc import face, imread
from flask import Flask, request, render_template
from werkzeug.utils import secure_filename
from waitress import serve
from Face_recognition import predict_image_for_server_2, Handler
from PIL import Image,ImageOps
from utils.utils import allowed_file, remove_file_extension, save_image

# import numpy
# numpy.load()

# app是Flask的实例，它接收包或者模块的名字作为参数，但一般都是传递__name__。
# 让flask.helpers.get_root_path函数通过传入这个名字确定程序的根目录，以便获得静态文件和模板文件的目录。
app = Flask(__name__)
# 生成24位随机字符
app.secret_key = os.urandom(24)
# os.path.abspath(__file__) 作用： 获取当前脚本的完整路径
# os.path.dirname: 去掉文件名，返回目录 
# APP_ROOT: 设置当前目录为主目录
APP_ROOT = os.path.dirname(os.path.abspath(__file__))
# 设置相关的路径
uploads_path = os.path.join(APP_ROOT, 'img')
tmp_path = os.path.join(APP_ROOT, 'uploads')
# 允许上传的文件格式
# set: 无序不重复元素集
allowed_set = set(['png', 'jpg', 'jpeg', 'pgm'])
model = Facenet()
model.detect_image(image_1=Image.open('test_1.jpg'), image_2=Image.open('test_2.jpg'))
handler = Handler()
# 使用app.route装饰器会将URL和执行的视图函数的关系保存到app.url_map属性上。
# 处理URL和视图函数的关系的程序就是路由，这里的视图函数就是get_iamge。

# 从http标准看来get,post,put,delete对应的就是对这个资源的查，改，增，删四个操作，
# 因此我们可以理解为get一般是用来获取/查询服务器资源信息，post一般是用于更新服务器资源信息。即get是向服务器发送取数据的一种请求，而post是向服务器提交数据的一种请求。
# 对于GET方式的请求，浏览器会把http header和data一并发送出去，服务器响应200（返回数据）：发一次。
# 而对于POST，浏览器先发送header，服务器响应100 continue，浏览器再发送data，服务器响应200 ok（返回数据）：发两次
@app.route('/upload', methods=['POST', 'GET'])
def get_image():
    # 通过POST请求获取图像文件，将图像提供给FaceNet模型，然后保存原始图像。
    # 以及它从FaceNet模型中嵌入到指定文件夹中的结果。
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template(
                template_name_or_list="warning.html",
                status="没有文件被 POST request!"
            )

        file = request.files['file']
        filename = file.filename

        if filename == "":
            return render_template(
                template_name_or_list="warning.html",
                status="请选择文件!"
            )
        
        if file and allowed_file(filename=filename, allowed_set=allowed_set):
            filename = secure_filename(filename=filename)
            # 读取图像文件为RGB的numpy数组
            # img = imread(name=file, mode='RGB')
            image = Image.open(file)
            image = ImageOps.exif_transpose(image)
            # 保存裁剪后的脸图像到 'uploads/'
            img_path = save_image(img=image, filename=filename, uploads_path=uploads_path)

            # det_img = cv2.imread(img_path)
            # det_img = cv2.cvtColor(det_img,cv2.COLOR_BGR2RGB)
            # bboxes, _ = handler.detector.detect(det_img)
            # if bboxes.shape[0] == 0:
            #     # os.remove(os.path.join(uploads_path, filename))
            #     return render_template(
            #     template_name_or_list="upload_result.html",
            #     status="上传失败！没有检测出人脸"
            #     )         
            # else:
            #     for i in range(bboxes.shape[0]):
            #         faceRect = bboxes[i]
            #         x, y, w, h, _ = faceRect.astype(np.int64)
                
            #         face_img = det_img[y - 10: h + 10, x - 10: w + 10]
            #         # 覆盖了原图
            #         face_img_path = save_image(img=face_img, filename=filename, uploads_path=uploads_path)
            #         # os.remove(os.path.join(uploads_path, filename))


            image = Image.open(img_path)
            # image = Image.open(face_img_path)
            model.save_model_vector(image=image, filename=filename)

            # 移除扩展名
            filename = remove_file_extension(filename=filename)


            # 模板文件就是html文件，需要放在templates文件夹中。
            # 传变量到模板中，可以把变量定义成字典，然后在render_template中，通过关键词参数的方式传递过去
            return render_template(
                template_name_or_list="upload_result.html",
                status="文件上传成功耶!"
            )

        else:
            return render_template(
                template_name_or_list="upload_result.html",
                # status="Image upload was unsuccessful! No human face was detected!"
                status="上传失败！没有检测出人脸"
            )
    else:
        return render_template(
            template_name_or_list="warning.html",
            status="没有接收到 POST HTTP!"
        )


@app.route('/predictImage', methods=['POST', 'GET'])
def predict_image():
    # 通过POST请求获取一个图像文件，将图像提供给FaceNet模型，然后得到embedding结果送去和embedding数据库比较。
    # 过程中不保存图片。
    # 然后呈现一个html页面，显示预测结果。
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template(
                template_name_or_list="warning.html",
                status="没有文件被 POST request!"
            )

        file = request.files['file']
        filename = file.filename

        # print(file)
        # print(filename)

        if filename == "":
            return render_template(
                template_name_or_list="warning.html",
                status="请选择人脸图片!"
            )

        # 读取图像文件为RGB的numpy数组
        # image = cv2.imread(file)
        # image = imread(name=file, mode='RGB')
        image = Image.open(file)
        image = ImageOps.exif_transpose(image)
        # print('image:', image)

        img_path = save_image(img=image, filename=filename, uploads_path=tmp_path)
        # print('img_path:', img_path)

        det_img = cv2.imread(img_path)
        det_img = cv2.cvtColor(det_img,cv2.COLOR_BGR2RGB)
        bboxes, _ = handler.detector.detect(det_img)
        if bboxes.shape[0] == 0:
            return render_template(
            template_name_or_list="upload_result.html",
            status="上传失败！没有检测出人脸"
            )         
        else:
            for i in range(bboxes.shape[0]):
                faceRect = bboxes[i]
                x, y, w, h, _ = faceRect.astype(np.int64)
                if y - 10 >0  and  x - 10 >0:
                    face_img = det_img[y - 10: h + 10, x - 10: w + 10]
                    # 覆盖了原图
                    face_img_path = save_image(img=face_img, filename=filename, uploads_path=tmp_path)
                    # os.remove(os.path.join(uploads_path, filename))
                else:
                    face_img_path = img_path

        # 如果检测到人脸
        # identity = predict_image_for_server_1(image1_path=img_path, model=model)
        identity = predict_image_for_server_2(image1_path=face_img_path, model=model)
        # print('identity:', identity)

        # 无问题            
        if identity[0] == -1:
            return render_template(
                template_name_or_list='predict_result.html',
                identity="操作成功! 但是系统无匹配身份!"
            )
        else:
            print('identity', identity[1])
            return render_template(
                template_name_or_list='predict_result.html',
                # identity=identity[1]
                identity=identity[1]
            )
    else:
        return render_template(
            template_name_or_list="warning.html",
            status="没有接收到 POST HTTP!"
        )


@app.route("/live", methods=['GET'])
def face_detect_live():
    # 通过网络摄像头实时检测人脸。
    #                      
    color = (0, 255, 0)
    # handler = Handler()

    cap = cv2.VideoCapture(0)
    
    while True:
        _, frame = cap.read()
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
                # image = frame[y - 10: h + 10, x - 10: w + 10]
                # res = predict_image_realtime(model=model, path_name='img', image1=image)

                # if res[0] == -1:
                cv2.rectangle(frame, (x - 10, y - 10), (w + 10,h + 10), color, thickness = 2)
                #     #文字提示是谁
                #     cv2.putText(frame,'NULL', 
                #                 (x + 30, y + 30),                      #坐标
                #                 cv2.FONT_HERSHEY_SIMPLEX,              #字体
                #                 1,                                     #字号
                #                 (255,0,255),                           #颜色
                #                 2)                                     #字的线宽  
                # else:
                #     cv2.rectangle(frame, (x - 10, y - 10), (w + 10, h + 10), color, thickness = 2)
                #     #文字提示是谁
                #     cv2.putText(frame,res[1], 
                #                 (x + 30, y + 30),                      #坐标
                #                 cv2.FONT_HERSHEY_SIMPLEX,              #字体
                #                 1,                                     #字号
                #                 (255,0,255),                           #颜色
                #                 2)                                     #字的线宽     
        cv2.imshow("face recognition", frame)
        
        k = cv2.waitKey(10)

        if k & 0xFF == ord('q'):
            break
 
    cap.release()


@app.route("/")
def index_page():
    # 使 'index.html' 网页手动上传图像。
    return render_template(template_name_or_list="index.html")


@app.route("/predict")
def predict_page():
    # 使 'predict.html' 页面手动上传预测图片。
    return render_template(template_name_or_list="predict.html")


if __name__ == '__main__':
    # server 和 FaceNet Tensorflow 的配置

    # waitress WSGI server上启动flask应用程序
    serve(app=app, host='0.0.0.0', port=9999)
