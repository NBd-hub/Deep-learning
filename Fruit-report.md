# Fruits-360数据集训练图像识别实验报告

### 	一.进行训练数据集处理

​				Fruits训练集是正规的格式（种类文件夹+图片），所以引用了dataset.ImageFolder()函数进行了数据加载，并用torch.utils.data.DataLoader引用数据，dataloder为迭代器进行batch的迭代，需要用enumerate进行调用。（<u>enumerate() 函数用于将一个可遍历的数据对象(如列表、元组或字符串)组合为一个索引序列</u>）：迭代出来的数据为索引+batch*data。



## 	二.进行数据预处理

​		引用imagenet标准化数据参数，后若遇到图像复杂的数据集可以自行进行数据集标准化：

```
import numpy as np

import cv2
import os

img_h, img_w = 32, 32

img_h, img_w = 32, 32   #经过处理后你的图片的尺寸大小
means, stdevs = [], []
img_list = []

imgs_path = "./data/sharedata/cat-dog/training/cats/"#数据集的路径采用绝对引用
imgs_path_list = os.listdir(imgs_path)

len_ = len(imgs_path_list)
i = 0
for item in imgs_path_list:
    img = cv2.imread(os.path.join(imgs_path,item))
    img = cv2.resize(img,(img_w,img_h))
    img = img[:, :, :, np.newaxis]
    img_list.append(img)
    i += 1
    print(i,'/',len_)    

imgs = np.concatenate(img_list, axis=3)
imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

#BGR --> RGB ， CV读取的需要转换，PIL读取的不用转换

means.reverse()
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))			
```

最好能够对图片进行翻转处理，使训练起来的模型更加具有泛化性。



## 		三.定义模型

​				这次用的模型为ALexNet，该模型拥有11层，但是模型比较落后，**之后对VGG，残差网络需要进行理解与实践**

​			不然训练出来的模型无法与高精度模型进行pk。

## 		四.定义损失函数与优化器

​				损失函数用的是交叉熵函数，**题外话：查询NLLLoss()损失函数与交叉熵损失有什么区别** ，优化器用的是SGD的随机梯度下降，**可以换用Adam优化器进行试试** 。

## 		五.进行模型训练

​				在模型中加入训练集进行训练，设置epoch为10，最后训练出来的精度为99%，并且用torch.save函数进行存储训练好的模型，上传阿里云端。

## 		六.加载图像测试集

​				获取测试集图像的路径，并用image.open进行将路径转化为image data

## 		七.进行测试集的训练

​				加入训练模型，并且将预测数据写入指定文件：

​				

```python
root = '/ilab/datasets/local/fruits/test'

f_l = os.listdir(root)
i=0
with open('/home/ilab/submission','w') as f:
    for filename in f_l:
        filepath = os.path.join(root,filename)
        img = Image.open(filepath)
        x = transformer(img).reshape((1,3,224,224))
        y = model(x)
        re=filename+' '+train_data.classes[int(y.argmax(dim=1))]+'\n'
        f.write(re)
```

​					

**针对python 的文件操作并不是很熟，这部分要进行加强** 

最后测出来的分数为96.8

#### 总结：如果将模型修改为高精度模型并且运用学习率衰退，并将图像预处理做的更好一些，应该得分会更高

