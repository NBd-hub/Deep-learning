论文名称：Rethinking ImageNet Pre-training

pdf链接：https://arxiv.org/pdf/1811.08883.pdf

源码链接：

摘要与网络结构框图：

​		摘要：在目标检测和实例分割两个领域，使用随机初始化方法训练的模型，在 COCO 数据集上取得了非常鲁棒的结果。其结果并不比使用了 ImageNet 预训练的方法差，即使那些方法使用了 MaskR-CNN 系列基准的超参数。在以下三种情况，得到的结果仍然没有降低：（1）仅使用 10％ 的训练数据；（2）使用更深和更宽的模型以及（3）使用多个任务和指标。

实验表明，使用 ImageNet 的预训练模型可以在训练早期加快收敛速度，但不一定能带来正则化的效果或最终提高目标任务的准确率。文中阐述了在不使用任何额外数据的情况下，COCO 数据集上物体检测结果为 50.9 AP 的方法，而这一结果与使用 ImageNet 预训练的 COCO 2017 竞赛方法结果相同。根据这些观察还发现，使用 ImageNet 预训练这一方法并不能带来最终效果的提升。

![image-20211009133034385](/Users/mengzezheng/Desktop/1.png)



如图1所示：灰色的曲线为进行过imagenet预训练的模型，而红色的曲线则是从头开始训练的模型，从图中可知，使用 ImageNet 的预训练模型可以在训练早期加快收敛速度，能迅速达到目标准确率，而从头开始训练的模型因为要学习低级层级的特征如边缘，纹理等，需要训练模型的时间就会变长，但是最终的训练结果几乎和进行预训练的模型的准确率相差无几。

​		网络结构框图：采用 Mask R-CNN，ResNet 或 ResNeXt，并采用特征金字塔网络（FPN）作为我们模型的主体结构

​							（在faster R—cnn中从一张224 * 224的图片通过VGG网络之后下降到7 * 7的feature map之后，对于语义分割来说，可					能会丢掉一些语义信息。总的来说，当图片尺寸大时语义信息弱，但是分辨率高，而小尺寸时语义信息强而分辨率低。从而就					需要使用fpn 来解决这个问题。）

​								FPN结构图：

​													![image-20211009140137479](/Users/mengzezheng/Desktop/2.png)



​							针对输入的图片进行resize裁剪，裁剪完的图片拥有明显的寓意信息，将该特征图片进行上采样，将图片进行扩大后再加					入没有裁剪完成的图片，这样一来合成的图片就成功拥有了明显的语义信息与高分辨率。

​							

​								Mask-R-Cnn结构图：

​									![image-20211009145127248](/Users/mengzezheng/Desktop/3.png)

​								

​							RPN结构图：

​							                   <img src="/Users/mengzezheng/Desktop/4.png" alt="image-20211009151309131" style="zoom:80%;" />



​								fpn/resnet结构图：

​											![image-20211009152424383](/Users/mengzezheng/Desktop/5.png)







​								fpn/resnet+mask r—cnn结构图：

​	

​												![image-20211009152254550](/Users/mengzezheng/Desktop/6.png)



​						输入图像进入fpn，产生feature map后进入Rpn 进行锚框标注与二分类的锚框消除，接着进入rol align进行锚框标注下的				图片进行裁剪，最后进行边缘框回归，图像分类，以及对像素进行分类。



主要创新点：

​		1.针对标准化的改变：batch Normalization是一种常见的标准化方式，但是对于物体检测时尺寸大的图片，受限于显卡的显存，就只能降低batch-size。

​			如下图所示，当batch size逐渐变小时，GN的错误率变化不明显，而BN的错误率明显上涨，所以这里将Batch Normalization

改成GN和SyncBN



​							![image-20211009154156239](/Users/mengzezheng/Desktop/7.png)

​				GN：

​							![这里写图片描述](/Users/mengzezheng/Desktop/8.png)





​				而SyncBN是一种多机多卡分布式的一种方式。通过多块GPU来增加batch-size。



​		2.收敛

​				关于收敛的问题，如果我们希望从零开始随机初始化训练我们的算法所需的时间，要小于那个使用ImageNet预训练初始化所			需的时间，这是不现实也是不公平的。如果忽视这个事实，那我们很可能得到的是不正确的结论。

​				典型的ImageNet 预训练涉及到了百万张图像的上百个epoch的迭代训练。这种大规模的学习过程除了能够学到高阶语义信息			外，还能够学到一些低阶的图像的特征。因此，在fine-tuning的时候就不需要重新学习这些低阶的图像特征描述了。因此，我们在			比较两个收敛速度的时候，需要选取那些训练周期较长的model。



步骤与结果分析：

​		设置网络架构，以ResNet或者ResNeXt为基础的Mask-RCNN，主要使用GN和SyncBN来取代BN的操作。所有的超参数设置参照Detectron设置。初始学习率为0.02，权重衰减为0.0001 ，动量参数值为0.9. 所有的模型训练在8GPU上，batch-size为2.  





​		**从头开始训练以匹配准确性**

​		实验中，我们发现当只使用 COCO 数据集时，从头开始训练的模型性能是能够匹配预训练模型的性能。我们在 COCO train2017 上训练模型，并在 COCO val2017 上验证模型的性能。训练数据集共有 118k 张图片，而验证集包含 5k 张图片。对于检测任务，我们评估了 bbox 和 AP（Aversage Precision）指标；对于实例分割，我们以 mask AP 作为评价标准。

​			

​			**Baselines with GN and SyncBN**

下图 3，图 4 和图 5分别展示了 ResNet50+GN，ResNet101+GN 以及 ResNet50+SynaBN 在验证集上的 bbox 和 AP 性能曲线。每张图上我们可以对比随机初始化训练的模型和经预训练微调的模型之间的性能差异。可以看到，在标准的 COCO 数据集上，从头开始训练的模型在 bbox 和 AP 上的表现，完全可以匹配经预训练微调的模型性能。而 ImageNet 数据集的预训练主要能够加速模型的收敛速度，并不会或很少提高模型最终的检测性能。



![img](https://t12.baidu.com/it/u=1678706684,633556480&fm=173&app=25&f=JPG?w=390&h=318&s=79AC3C72010F654F0C54E4DE0000E0B1)

图 3 在 COCO val2017 数据集上，以 ResNet50+GN 为主体结构的 Mask R-CNN 模型的 bbox 和 AP 性能曲线。

![img](https://t11.baidu.com/it/u=992748280,3423570702&fm=173&app=25&f=JPG?w=364&h=304&s=58A83C72190E654D0CDDD1DA0000C0B1)

图 4 在 COCO val2017 数据集上，以 ResNet101+GN 为主体结构的 Mask R-CNN 模型的 bbox 和 AP 性能曲线。

![img](https://t11.baidu.com/it/u=237584740,2850160584&fm=173&app=25&f=JPG?w=367&h=305&s=3DAC7C32010F654D10D4D1DA0000A0B1)

图 5 在 COCO val2017 数据集上，以 ResNet50+SyncBN 为主体结构的 Mask R-CNN 模型的 bbox 和 AP 性能曲线。

**Multiple detection metric**

下图 6 进一步比较了两种情况下模型在多种检测指标上的性能，包括分别在 IoU 阈值为 0.5 和 0.75的情况下，Mask R-CNN 模型的 box-level AP，segmentation-level AP。



![image-20211009161359397](/Users/mengzezheng/Desktop/9.png)



​		图 6 从头开始训练 Mask R-CNN+FPN+GN 为结构的模型与经预训练的模型之间在多种检测指标上的性能对比，其中AP50是指当预测框与真实框的lou值大于50时才被认为是true

**Models without BN/GN--VGG nets**

为了研究模型性能的泛化能力，以 VGG-16 作为主体结构，参考先前 Faster R-CNN 模型的实现过程，没有引入 FPN 架构，并采用标准的超参数方案，从头开始训练模型，并将其与在 ImageNet 上预训练的模型性能进行比较分析。我们发现，即使经 ImageNet 预训练的模型，其收敛的速度也很缓慢，而从头开始训练的模型最终也能达到与之相匹配的检测性能。



**用更少的数据从头开始训练**

实验过程中，还发现了随着数据量的减少，从头开始训练的模型性能并不会随之下降，仍然还能取得与预训练模型相匹配的性能。



**35k COCO training samples vs 10k COCO training samples**

分别从 COCO 数据集中随机选择 35k 和 10k 张训练数据，用于从头开始训练或基于预训练模型进行微调操作。下图 7 展示了二者在更少的训练数据上的 bbox 和 AP 性能对比。可以看到，尽管用更少的数据，从头开始训练的模型最终也能赶上预训练模型的性能。此外，经 ImageNet 预训练并不会有助于防止过拟合现象的发生。





![image-20211009161921662](/Users/mengzezheng/Desktop/10.png)



图 7 以更少的 COCO 样本训练的 Mask R-CNN+ResNet50-FPN+GN 模型在 val2017 上的 bbox 和 AP 性能。左图：以 35k COCO 样本训练，采用默认的超参数设置，模型在改变学习率的过程中发生了过拟合现象。中图：以 35k COCO 样本训练，采用与随机初始化模型相同的超参数设置。右图：以 10k COCO 样本训练，采用与随机初始化模型相同的超参数设置。





启示：

​	1.在不需要对模型结构进行大幅度修改的情况下，可以在一个新的任务中从头开始训练一个模型。

​	2.从头开始训练一个模型通常需要更多的迭代步数才能获得充分的收敛。

​	3.从头开始训练的模型性能能够匹配的上经预训练的模型性能，即使是在只有 10k COCO 训练数据的情况下。

​    4.经 ImageNet 预训练的模型，在一个新的任务中能够加快收敛速度。

​	5.经 ImageNet 预训练的模型并不一定有助于减少过拟合现象的发生，除非我们采用非常小的数据。

​	6.如果我们的目标是比定位和分类更敏感的任务，那么 ImageNet 预训练对于模型的帮助将变得更小。

