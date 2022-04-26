# Image-Segmentation-for-Semi-Supervised-Learning
【高引】飞桨深度学习云训练平台一键fork：https://aistudio.baidu.com/aistudio/projectdetail/2148971

# 目录
* [1. 项目介绍](#1-项目介绍)
* [2. 数据集](#2-数据集)
* [3. 模型组网](#3-模型组网)
   * [3.1 Unet](#31-Unet)
   * [3.2 PSPnet](#32-PSPnet)
   * [3.3 Deeplabv3](#33-Deeplabv3)
* [4. 项目运行](#4-项目运行)
   * [4.1 模型训练](#41-模型训练)
   * [4.2 模型预测](#42-模型预测)
   * [4.3 结果记录](#43-结果记录)
* [5. 总结提升](#5-总结提升)
* [6. 参考引用](#6-参考引用)


# 1. 项目介绍

# 2. 数据集

# 3. 模型组网
## 3.1 Unet
U-Net网络结构因为形似字母“U”而得名，最早是在医学影像的细胞分割任务中提出，结构简单适合处理小数量级的数据集。比较于FCN网络的像素相加，U-Net是对通道进行concat操作，保留上下文信息的同时，加强了它们之间的语义联系。整体是一个Encode-Decode的结构，如下图所示。
<center>
<img src="https://img-blog.csdnimg.cn/a8304e5957a34a1a92a804bb9bfd231e.png" width=100%>
</center>

* **知识点1**:下采样Encode包括conv和max pool，上采样Decode包括up-conv和conv。
* **知识点2**:U-Net特点在于灰色箭头，利用通道融合使上下文信息紧密联系起来。


## 3.2 PSPnet
Pyramid Scene Parsing Network（PSPNet）网络结构形似金字塔而被命名，能够聚合不同尺度下的上下文信息，在场景解析上有很好的效果。PSPNet的精髓在于pyramid parsing module的构建，能够增大深层区域的感受野。
<center>
    <img src="https://img-blog.csdnimg.cn/a9f97cbcd2a444659b28bbfa8c6d01c6.png" width=100%>
</center>

* **知识点1**:多尺度特征融合可以提高模型性能，深层网络中包含更多的语义信息和较小的位置信息。
* **知识点2**:input image需要通过CNN网路提取特征，这里使用的是飞桨预训练的resnet50网络。
<center>
    <img src="https://img-blog.csdnimg.cn/cfc26d218064444dbd2ac73fd7d05ea2.png" width=100%>
</center>

* **知识点3**:PSPmodule将CNN的输出划成四个通道，然后进行上采样，全局特征和局部特征进行融合得到2C通道。
<center>
    <img src="https://img-blog.csdnimg.cn/539fc019ec5b4639b3b49b38a613ecd4.png" width = 100%>
</center>


## 3.3 Deeplabv3
空洞卷积（Dilatee/Atrous Convolution）是一种特殊的卷积算子，针对卷积神经网络在下采样时图像分辨率降低、部分信息丢失而提出的卷积思路，通过在卷积核中添加空洞以获得更大的感受野。

![](https://ai-studio-static-online.cdn.bcebos.com/cedf91586d0045c5a031ba5cfc2204996933d4d72c934dd7adc2a5f4366fec58)

* 3x3卷积核，dilation rate 分别为1， 2， 4，空洞部分填充零。
* 输入大小为[H，W]，卷积核大小为{FH，FW]，填充为P，步幅为S，计算输出大小
$$
\begin{aligned}
&O H=\frac{H+2 P-F H}{S}+1 \\
&O W=\frac{W+2 P-F W}{S}+1
\end{aligned}
$$
* 3x3卷积核可以等效为5x5，假设卷积核大小k=3，空洞数d=2，则等效卷积核k‘，感受野RF，第一层感受野为3，Si为之前所有层步长的乘积。RFa=3，RFb=5，RFc=8。
$$
\begin{aligned}
&k^{\prime}=k+(k-1) \times(d-1)\\
&R F_{i+1}=R F_{i}+\left(k^{\prime}-1\right) \times S_{i}\\
&S_{i}=\prod_{i=1}^{i} \text { Stride }_{i}
\end{aligned}
$$
$$
\begin{aligned}
&k^{\prime}=3+(3-1) \times(2-1)=5\\
&R F_{i+1}=3+(5-1) \times S_{i}\\
&S_{i}=\prod_{i=1}^{i} \text { Stride }_{i}=1
\end{aligned}
$$

![](https://ai-studio-static-online.cdn.bcebos.com/c5a720628d9442a68674f5fe7a7789c129ba12712c254c13a62e54c1bbf1b8b2)


# 4. 项目运行

## 4.1 模型训练

## 4.2 模型预测

## 4.3 结果记录

# 5. 总结提升

# 6. 参考引用
