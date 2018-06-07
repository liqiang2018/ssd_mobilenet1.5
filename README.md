# mobilenet

#### 项目介绍
第九章作业
利用slim框架和object_detection框架，做一个物体检测的模型
#### 数据集

https://gitee.com/ai100/quiz-w8-data.git
155张图片
数据集中的物品分类如下：
computer
monitor
scuttlebutt
water dispenser
drawer chest
#### 代码
新增object_detection\dataset_tools\create_data.py
做对应修改使得程序满足要求正常运行
#### 预训练模型

object_detection框架提供了一些预训练的模型以加快模型训练的速度，不同的模型及检测框架的预训练模型不同，常用的模型有resnet，mobilenet以及最近google发布的nasnet，检测框架有faster_rcnn，ssd等，本次作业使用mobilenet模型ssd检测框架，其预训练模型请自行在model_zoo中查找: https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md

ssd论文：https://arxiv.org/abs/1512.02325
mobilenet论文：https://arxiv.org/abs/1704.04861

#### 训练过程
#### 验证
