# face_recogniton_websites
a web in masked face recognition system for undergraduate final test (本科毕设：可遮挡的人脸识别系统）


本项目是用于毕业设计的代码，创建一个以浏览器为接口的可遮挡人脸识别系统，目前测试下来，LFW数据集识别成功率99.8%， RA数据集识别成功率94%，没有进行压力测试。
此项目已实现完全开源，希望各位使用者可以在此代码的基础上给予项目模块的扩建和网页的美化，提升系统的实现效率。


| 语言及框架  |   |
| :----  | :----  |
| Python  |  系统编写语言 |
| Flask | Web框架，搭建浏览器接口 |




|  技术金字塔   |   |
|  :----  | :----  |
| 网络模型  | MobileNet |
| 损失函数 | Triplet Loss |
| 学习率优化 | SGD |
| 学习率调整 | warm up + Cosine annealing |
| 预训练模型 | FaceNet  |
