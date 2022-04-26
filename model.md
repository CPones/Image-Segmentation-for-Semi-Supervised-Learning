# 一、Unet
U-Net网络结构因为形似字母“U”而得名，最早是在医学影像的细胞分割任务中提出，结构简单适合处理小数量级的数据集。比较于FCN网络的像素相加，U-Net是对通道进行concat操作，保留上下文信息的同时，加强了它们之间的语义联系。整体是一个Encode-Decode的结构，如下图所示。
<center>
<img src="https://img-blog.csdnimg.cn/a8304e5957a34a1a92a804bb9bfd231e.png" width=100%>
</center>

* **知识点1**:下采样Encode包括conv和max pool，上采样Decode包括up-conv和conv。
* **知识点2**:U-Net特点在于灰色箭头，利用通道融合使上下文信息紧密联系起来。

# 二、PSPnet
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
    <img src="https://img-blog.csdnimg.cn/539fc019ec5b4639b3b49b38a613ecd4.png" width=100%>
</center>

# 三、Deeplabv3
## 3.3 Deeplabv3
空洞卷积（Dilatee/Atrous Convolution）是一种特殊的卷积算子，针对卷积神经网络在下采样时图像分辨率降低、部分信息丢失而提出的卷积思路，通过在卷积核中添加空洞以获得更大的感受野。
<center>
    <img src="https://ai-studio-static-online.cdn.bcebos.com/cedf91586d0045c5a031ba5cfc2204996933d4d72c934dd7adc2a5f4366fec58" width=100%>
</center>

<center>
    <img src="https://ai-studio-static-online.cdn.bcebos.com/c5a720628d9442a68674f5fe7a7789c129ba12712c254c13a62e54c1bbf1b8b2" width=100%>
</center>

