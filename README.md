# face_recogniton_websites
a web in masked face recognition system for undergraduate final test (本科毕设：可遮挡的人脸识别系统）


本项目是用于毕业设计的代码，创建一个以浏览器为接口的可遮挡人脸识别系统，目前测试下来，LFW数据集识别成功率99.8%， RA数据集识别成功率94%，没有进行压力测试。<br/>此项目已实现完全开源，希望各位使用者可以在此代码的基础上给予项目模块的扩建和网页的美化，提升系统的实现效率。
### 安装
***
#### 环境配置
+ python 3.6.8
+ flask 1.0
+ insightface 0.1.5
+ macOS
+ 此为我实验的环境配置，其他系统版本获取可以使用

安装相关的依赖：
```shell
pip3 install -r requirements.txt
```
### 使用
***
+ 下载WebFace数据集放到datasets文件夹下，Webface数据集下载链接: https://pan.baidu.com/s/1gIz39WBZXs-hVVv3VjHbfA?pwd=19cr 提取码: 19cr 
+ 下载LFW数据集放到lfw文件夹下，LFW数据集下载链接: https://pan.baidu.com/s/1G4k3Z7tKWtdz87VoPX1Nqw?pwd=2982 提取码: 2982 
+ 运行train.py生成人脸识别模型
+ 运行predict.py验证人脸识别系统是否可以正确使用
+ 运行server.py开启人脸识别系统，网站默认网址为localhost:9999


| 技术金字塔 |                            |
| :--------- | :------------------------- |
| 网络模型   | MobileNet                  |
| 损失函数   | Triplet Loss               |
| 学习率优化 | SGD                        |
| 学习率调整 | warm up + Cosine annealing |
| 预训练模型 | FaceNet                    |

### 效果
***


