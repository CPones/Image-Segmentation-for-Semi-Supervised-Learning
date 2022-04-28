# Image-Segmentation-for-Semi-Supervised-Learning
【高引】飞桨深度学习云训练平台一键fork：https://aistudio.baidu.com/aistudio/projectdetail/2148971

# 目录
* [1. 项目介绍](#1-项目介绍)
* [2. 数据集](#2-数据集)
   * [2.1 数据集划分](#21-数据集划分)
   * [2.2 文件目录](#22-文件目录)
   * [2.3 图像可视化](#23-图像可视化)
* [3. 项目运行](#3-项目运行)
   * [3.1 模型训练](#31-模型训练)
   * [3.2 模型预测](#32-模型预测)
   * [3.3 结果记录](#33-结果记录)
* [4. 总结提升](#4-总结提升)
* [5. 参考引用](#5-参考引用)


# 1. 项目介绍
半监督学习（Semi-Supervised Learning）是指通过大量无标注数据和少量有标注数据完成模型训练，解决具有挑战性的模式识别任务。近几年，随着计算硬件性能的提升和大量大规模标注数据集的开源，基于深度卷积神经网络(Deep Convolutional Neural Networks, DCNNs)的监督学习研究取得了革命性进步。然而，监督学习模型的优异性能要以大量标注数据作为支撑，可现实中获得数量可观的标注数据十分耗费人力物力（例如：获取像素级标注数据）。于是，半监督学习逐渐成为深度学习领域的热门研究方向，只需要少量标注数据就可以完成模型训练过程，更适用于现实场景中的各种任务。

图像分割属于像素级别的分类任务，基于深度学习的图像分割研究在自动驾驶、医学图像分析、工业自动化等各种场景都具有重要意义，研究者在全卷积神经网络（Full Connection Network， FCN）的基础上提出一系列不同的改进方法，本项目数据集来自于“第三届中国AI+创新创业大赛：半监督学习目标定位竞赛”，使用三种网络结构来训练模型，特点如下：

- Unet：Encoder-Decoder结构，同层级通道concat以保留上下文信息。
- PSPnet：特征金字塔结构，聚合不同尺度下的上下文信息。
- Deeplab：空洞卷积以增大特征图的感受野。

网络结构请查看[model.md](model.md)

# 2. 数据集
## 2.1 数据集划分
1. 训练数据集包括50,000幅像素级有标注的图像，共包含500个类，每个类100幅图像。[点击下载](https://aistudio.baidu.com/aistudio/datasetdetail/95703)；
2. A榜测试数据集包括11,878幅无标注的图像。[点击下载](https://aistudio.baidu.com/aistudio/datasetdetail/95703)
3. B榜测试数据集包括10,989幅无标注的图像。[点击下载](https://aistudio.baidu.com/aistudio/datasetdetail/100087)


## 2.2 文件目录
```
|---|train_image #原始图像
|---|---|n014443537 #文件夹，类别
|---|---|---|n01443537_2.png #图像
|---|---|---|……
|---|---|n01491361
|---|---|---|n01443537_176.png #图像
|---|---|---|……
……

|---|train_50k_mask #标注图像
|---|---|n014443537 #文件夹，类别
|---|---|---|n01443537_2.png #图像
|---|---|---|……
|---|---|n01491361
|---|---|---|n01443537_176.png #图像
|---|---|---|……
……

|---|val_image #测试提交
|---|---|ILSVRC2012_val_00000001.JPEG
|---|---|ILSVRC2012_val_00000004.JPEG
|---|---|……


```
## 2.3 图像可视化
|原始图像|标注图像|
|:-:|:-:|
|![](https://ai-studio-static-online.cdn.bcebos.com/a74cf67e3ab741b99aed5ed7734c5d8f8d40219eab7b4cafa59e96e17f701714)|![](https://ai-studio-static-online.cdn.bcebos.com/af685ee04f5f4acbaaba5141ce2c558440729d14762c412a94f02bb70a2e1c1d)|
|![](https://ai-studio-static-online.cdn.bcebos.com/40dd8e974a1d4981a9fc1444fa229a3d95efbf6cd16c4081a0009fc567c0605e)|![](https://ai-studio-static-online.cdn.bcebos.com/1f80bca410944e2c803959c7e75f32208da50e2a796a4331a0e4fd1db10b7f1f)|

# 3. 项目运行
数据集下载后存放于`data`文件夹下，整个项目的文件目录和功能介绍如下：
```
|--train.py       #模型训练
|--predict.py     #模型预测
|--Unet.py        #Unet模型
|--PSPnet.py      #PSPnet模型
|--Deeplab.py     #Deeplab模型
|--dataset.py     #数据读取器
|--data
    |--train_image        #训练集、验证集，原始图像
    |--train_50k_mask     #训练集、验证集，标注图像
    |--val_image          #测试集，输入图像
    |--val_label          #测试集，预测图像
|--output
    |--final.pdparams     #参数保存文件
    |--final.pdopt        #参数保存文件
```

## 3.1 模型训练
可选择三种模型`UNet`、`PSPnet`和`Deeplabv3`进行训练，注意数据集的路径应保持一致。

```
!python train.py --model='UNet'  \
                 --eval_num=1000 \
                 --batch_size=4  \
```

## 3.2 模型预测
可选择三种模型`UNet`、`PSPnet`和`Deeplabv3`进行预测，注意数据集的路径应保持一致。

```
!python predict.py --model='Unet' \
                   --checkpoint_path='output/final'  \
                   --eval_num=1000  \
                   --batch_size=4  \
```

预测结果如下图：

![](https://ai-studio-static-online.cdn.bcebos.com/91531674a8024bdca1a2f41552c518869096ed29d04a416d8d2ad2d2ef6f1713)

## 3.3 结果记录

* 1.U-Net网络模型：精度可以达到 0.4-0.5

* 2.PSPNet网络模型：精度可以达到 0.5-0.6

* 3.DeepLabv3网络模型：精度可以达到 0.7-0.8。

##############################

* 2021年07月03日
* 网络骨架：U-net基线
* 项目编号：000
* 编译环境：飞桨paddle
* 定义参数：batch_size=8，epoch=5
* 运行时间：起始（10:00），结束（14:00）
* IoU成绩：0.37496

##############################

* 2021年07月04日
* 网络骨架：U-net
* 项目编号：005
* 编译环境：飞桨paddle
* 定义参数：batch_size=4，epoch=10
* 运行时间：起始（20:20），结束（22:00）
* 学习率：1e-3
* IoU成绩：0.5309

##############################

* 2021年07月08日
* 网络模型：PSPnet
* 项目编号：008
* 编译环境：飞桨paddle
* 定义参数：batch_size=16，epoch=30
* 运行时间：起始（9:00），结束（13:00）
* 学习率：1e-3
* IoU成绩:0.62947  /epoch=15
* IoU成绩:0.62471  /epoch=16
* IoU成绩:0.61097  /epoch=18
* IoU成绩:0.61975  /epoch=22
* IoU成绩:0.61975  /epoch=32

##############################

* 2021年07月13日
* 项目编号：011
* 编译环境：飞桨paddle
* 定义参数：batch_size=8，epoch=30
* 运行时间：起始（22:00），结束（2:00）
* 学习率：1e-3
* 网络模型：Deeplabv3
* IoU成绩:0.75799

##############################


# 4. 总结提升
提升图像预测精度最有效的方法是更换模型结构，优秀的网络模型决定了精度的上限，随着网络复杂程度的提高，模型训练时长以及模型参数大小都会有大的飞跃，在确定选用哪种网络模型后，模型参数调整可以小幅度提升模型精度的效果，可以总结为“模型结构粗调，模型参数微调”。

- 图像分割领域目前使用DeepLab系列的模型居多，本项目亦可验证结论：DeepLabv3>PSPNet>UNet，其中DeepLabv3训练时间过长，可以跳转到[基于PaddleSeg的图像分割模型训练项目](https://aistudio.baidu.com/aistudio/projectdetail/2249209)

- 模型参数选择集中在优化器、学习率、损失函数等方面，多次实验发现Momentum优化器、lr=0.001、CrossEntropyLoss损失函数对本项目比较友好，当然还可以从数据增强、激活函数、模型融合等方面考虑。


# 5. 参考引用
>[图像分割的打怪升级之路——UNet、PSPNet、Deeplab](https://aistudio.baidu.com/aistudio/projectdetail/2148971)

>[第三届中国AI+创新创业大赛：半监督学习目标定位竞赛](https://aistudio.baidu.com/aistudio/competition/detail/78)

>[基于PaddleSeg的图像分割模型训练项目，精度0.75左右](https://aistudio.baidu.com/aistudio/projectdetail/2249209)

