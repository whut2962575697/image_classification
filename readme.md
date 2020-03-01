# Fashion-mnist分类任务报告
## 1.解题思路说明
### 1.1待解决的问题
针对`Fashion-MNIST`数据集，设计、搭建、训练机器学习模型，能够尽可能准确地分辨出测试数据地标签。
### 1.2整体思路/方案
本次任务我们使用了开源的深度卷积神经网络resnet34作为我们的baseline backone, 同时通过消融实验，设计了数据增强方法。经过实验调试，我们对比了不同的backbone网络的性能，以及各种超参数对实验结果的影响，最终选择了最优的模型。
具体实现方案如下。
#### 1.2.1选择baseline
考虑到本次任务的原始数据分辨率小(28x28),过深过大的网络可能会导致发生过拟合(overfiting)现象，我们选择了一个参数量较少的深度模型resnet34作为此次的baseline backbone。
#### 1.2.2设计数据增强
经过实验我们发现使用resnet34网络，仍然会发生一定程度的过拟合，数据增强是解决过拟合一个比较好的手段，它的本质是在一定程度上扩充训练数据样本，避免模型拟合到训练集中的噪声，所以设计一个好的数据增强方案尤为必要。在CV任务中，常用的数据增强包括RandomCrop(随机扣取)、Padding(补丁)、RandomHorizontalFlip(随机水平翻转)、ColorJilter(颜色抖动)等。还有一些其他高级的数据增强技巧，比如RandomEreasing(随机擦除)、MixUp、CutMix、AutoAugment，以及最新的AugMix和GridMask等。在此次任务中我们通过实验对比，选择了一个较合适的数据增强方案。
#### 1.2.3对比其他backbone网络
由于resnet34网络提取特征的能力有限，我们在设计了一个合适的数据增强方案后，将backbone换成了其他更强的backbone,例如efficientnet,wideresnet(wrn)等，经过实验发现wrn40-4效果更好。
#### 1.2.4参数调优
针对各种参数的选择，我们利用控制变量法与网格搜索方法，选取最优参数，为了节省训练时间我们选用了收敛更快的Adam优化器
##### (1)学习率(learning rate)的选择
我们分别尝试了3e-2+warmup(初始3e-4)+Cosine衰减、3e-3+warmup(初始3e-5)+Cosine衰减、3e-4+warmup(初始3e-6)+Cosine衰减
##### (2)batch size的选择
考虑到训练时间和机器性能，我们使用128batch size
##### (3)输入图像大小的选择
原则上图像分辨率高对网络识别的效果越好，但是由于机器性能和训练时间限制，我们选择32x32大小的分辨率
##### (4)迭代epoch选择
我们开始设置了一个比较大的epoch，后面观察到网络的收敛的情况，最终选择400个epoch
#### 1.2.5测试增强方法
一个常用提高精度的方法测试时增强（test time augmentation, TTA），可将准确率提高若干个百分点，测试时将原始图像造出多个不同版本，包括不同区域裁剪和更改缩放程度等，并将它们输入到模型中；然后对多个版本进行计算得到平均输出，作为图像的最终输出分数。这种技术很有效，因为原始图像显示的区域可能会缺少一些重要特征，在模型中输入图像的多个版本并取平均值，能解决上述问题。
### 1.3数据处理
#### 1.3.1数据转换
由于原始数据为单通道图片，所以我们有两种选择方案
- 默认1通道图片进行训练
- 将图片转换为3通道图片进行训练

使用默认的单通道图片进行训练无法使用预训练模型，所以比较好的方法是将图片转换为3通道图片进行训练，这样可以用到一些backbone在其他数据集上的预训练模型，这种迁移学习的方法能够加快网络收敛速度并在一定程度上提高性能。
#### 1.3.2数据增强
通过实验对比，选择了一个如下数据增强方案：
- Resize 36x36
- RandomCrop(随机) 32x32
- RandomHorizontalFlip(随机水平翻转)
- RandomEreasing(随机擦除)
- AutoAugment
- CutMix
- Normalation 
### 1.4模型训练
模型训练策略：   
- WarmUp 10 epoch 
- CosineAnnealingLR Scheduler
- Adam 3e-4 + weight_decay 5e-4 / Ranger Optimizer(no warm up) 4e-3
- epoch 400

### 1.5结果分析
| train_size | batch_size | lr | backbone | tricks | test_loss | test_acc |
| :-----| ----: | :----: | :----: | :----: | :----: | :----: |
| 32x32 | 128 | 3e-4 | wrn40_4 | WarmUp + CosineAnnealingLR + RandomCrop + RandomHorizontalFlip + RandomErasing + AutoAugment + CutMix | 0.1739 | 0.9579 |
| 32x32 | 128 | 3e-4 | wrn40_4 | WarmUp + CosineAnnealingLR + RandomCrop + RandomHorizontalFlip + RandomErasing + AutoAugment + CutMix + TTA | 0.1739 | 0.9607 |
| 32x32 | 128 | 4e-3 | wrn40_4(Mish) | Ranger + CosineAnnealingLR + RandomCrop + RandomHorizontalFlip + RandomErasing + AutoAugment + CutMix | 0.1490 | 0.9601 |
| 32x32 | 128 | 4e-3 | wrn40_4(Mish) | Ranger + CosineAnnealingLR + RandomCrop + RandomHorizontalFlip + RandomErasing + AutoAugment + CutMix + TTA | 0.1490 | 0.9621 |     

ps: 参数调一调应该能更高，由于时间和机器限制，大部分参数没仔细调，而且不加TTA最好的是结果是0.9609，但是模型没保存下来，懒得再训了，就用了0.9601，有条件的可以试试增加分辨率，应该会提高一些
## 2.数据和模型的使用
### 2.1数据说明
Fashion-MNIST的目的是要成为MNIST数据集的一个直接替代品。作为算法作者，你不需要修改任何的代码，就可以直接使用这个数据集。Fashion-MNIST的图片大小，训练、测试样本数及类别数与经典MNIST完全相同。
#### 类别标注
在Fashion-mnist数据集中，每个训练样本都按照以下类别进行了标注：
| 标注编号 | 描述 |
| --- | --- |
| 0 | T-shirt/top（T恤）|
| 1 | Trouser（裤子）|
| 2 | Pullover（套衫）|
| 3 | Dress（裙子）|
| 4 | Coat（外套）|
| 5 | Sandal（凉鞋）|
| 6 | Shirt（汗衫）|
| 7 | Sneaker（运动鞋）|
| 8 | Bag（包）|
| 9 | Ankle boot（踝靴）|
#### 数据集大小
| 数据集 | 样本数量 |
| --- | --- |
| 训练集 | 60000|
| 测试集 | 10000|
### 2.2 预训练模型的使用
- resnet34使用image-net上的预训练模型
- efficientnet使用image-net上的预训练模型
- wrn40-4不使用预训练模型
## 3.项目运行环境
该框架支持cpu,单gpu,多gpu/syncbn,支持日志系统，集成了多种实用的tricks方便易用,计划不久开源。项目运行在python3.6, pytorch>0.4。
### 3.1项目所需的工具包/框架
具体依赖如下
- python==3.6
- albumentations==0.4.3
- imageio==2.6.1
- imgaug==0.2.6
- pandas==0.25.1
- hickle==3.4.5
- tqdm==4.36.1
- opencv_python==4.1.1.26
- scikit_image==0.15.0
- mlconfig==0.0.4
- visdom==0.1.8.9
- torch==1.0.0
- torchvision==0.2.0
- yacs==0.1.6
- numpy==1.17.4
- scipy==1.3.1
- Pillow==6.2.1
- skimage==0.0
### 3.2项目运行的资源环境
6G gtx-1060
## 4.项目运行方法

### 4.1项目的文件结构
详情见github
### 4.2项目运行步骤


