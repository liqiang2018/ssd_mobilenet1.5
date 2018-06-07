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

#### 1.训练过程
现在本机上运行2个sep确认代码正确
python object_detection/train.py --train_dir=model\train --pipeline_config_path=model/ssd_mobilenet_v1_pets.config
![image](https://github.com/liqiang2018/coco/blob/master/image/image/clipboard.png)
成功后会在train_dir下生产checkpoint
![image](https://github.com/liqiang2018/coco/blob/master/image/image/2.png)
####2. 验证
python object_detection/eval.py --logtostderr 
--pipeline_config_path=F:\\code\\CNN\\models\\research\\models\\faster_rcnn_resnet101_voc07.config 
--checkpoint_dir=models/train
 --eval_dir=models/eval
![image](https://github.com/liqiang2018/coco/blob/master/image/image/3.png)
验证整个后会在eval目录下生成验证文件
![image](https://github.com/liqiang2018/coco/blob/master/image/image/4.png)
####3 导出训练好的模型
python object_detection/export_inference_graph.py --input_type image_tensor --pipeline_config_path model/ssd_mobilenet_v1_pets.config --trained_checkpoint_prefix model\train\model.ckpt-2 --output_directory data\exported_graphs
![image](https://github.com/liqiang2018/coco/blob/master/image/image/5.png)
会将模型导入saved_model/saved_model.pb
![image](https://github.com/liqiang2018/coco/blob/master/image/image/6.png)
####4.导出的模型运行inference
python inference.py --output_dir=data --dataset_dir=F:\data\quiz-w8-data
会在output目录下生产output.png文件，由于只训练2step，效果很不好
![image](https://github.com/liqiang2018/coco/blob/master/image/image/7.png)
准备在tinymind上运行
但tinymind各种问题，也是无语，在tinymind上耗了好久，没什么进展
![image](https://github.com/liqiang2018/coco/blob/master/image/image/8.png)
等tinymind可以运行的时候，效果应该会更好,
数据集上传似乎有问题，导致运行的时候找不到数据源
https://www.tinymind.com/executions/g1fok4vz
