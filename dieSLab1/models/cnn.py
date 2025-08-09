import torch.nn as nn

class CNN(nn.Module):
    """
    简单卷积神经网络（CNN）
    结构：Conv2d -> ReLU -> MaxPool -> Conv2d -> ReLU -> MaxPool -> Flatten -> FC -> ReLU -> FC
    """

    def __init__(self, in_channels=3, num_classes=10, img_size=28):
        """
        初始化卷积神经网络（CNN）模型的构建。此部分定义了模型中的卷积层、池化层和全连接层。
        
        参数：
        in_channels (int): 输入图像的通道数，默认为3，适用于RGB彩色图像。
                           若输入为灰度图像，可以将该参数设置为1。
        num_classes (int): 输出类别的数量，默认为10，适用于数字分类（例如MNIST数据集，数字0到9）。
        img_size (int): 输入图像的大小（高度和宽度），默认为28，适用于MNIST图像。
                        该参数用于计算卷积和池化操作后特征图的尺寸。
        """
        super().__init__()  # 调用父类nn.Module的初始化方法。它负责处理神经网络中一些基本操作，如模型保存和加载。

        # 定义模型的各个层：

        # nn.Conv2d(in_channels, out_channels, kernel_size, padding)：
        # 第一层卷积层，将输入的in_channels通道数的图像通过卷积核（卷积核的大小为3x3）处理，
        # 输出16个特征图，每个特征图的大小经过卷积后不变（因为使用了padding=1，保持图像大小不变）。
        # padding=1是为了保持输出特征图的空间尺寸与输入图像相同。
        self.conv1 = nn.Conv2d(in_channels, 8, kernel_size=3, padding=1)

        # nn.ReLU()：ReLU激活函数。ReLU激活函数的输出是：f(x) = max(0, x)，
        # 这使得卷积操作的输出在负值部分被截断，保持正值部分不变。
        self.relu1 = nn.ReLU()

        # nn.MaxPool2d(2)：最大池化层。池化层的作用是对输入进行下采样，以减少空间维度，减轻计算量。
        # 这里使用了2x2的池化核，步幅默认为2，意味着每次将特征图的2x2区域压缩成一个值。
        # 该操作将空间尺寸减半。
        self.pool1 = nn.MaxPool2d(2)

        # 第二层卷积层：将8个特征图的输入通过3x3卷积核，输出16个特征图。经过padding=1，保持输出图像的尺寸。
        self.conv2 = nn.Conv2d(8, 16, kernel_size=3, padding=1)

        # 第二个ReLU激活函数，类似于第一个ReLU。
        self.relu2 = nn.ReLU()

        # 第二个最大池化层，用于进一步减少特征图的空间尺寸。
        self.pool2 = nn.MaxPool2d(2)

        # nn.Flatten()：展平层。将多维的特征图展平为一维向量，方便输入到全连接层。
        self.flatten = nn.Flatten()

        # 第一层全连接层：将展平后的特征向量传入全连接层，输出32个神经元的激活值。
        # 输入特征的大小由卷积和池化操作后特征图的尺寸决定。通过公式计算，输出特征的尺寸为16 * (img_size // 4) * (img_size // 4)。
        # 假设输入图像的尺寸为28x28，经过两次2x2池化后，特征图的尺寸为7x7。
        self.fc1 = nn.Linear(16 * (img_size // 4) * (img_size // 4), 32)

        # 第三个ReLU激活函数，用于引入非线性。
        self.relu3 = nn.ReLU()

        # 第二层全连接层：从32个神经元输出到num_classes个类别的得分。这里的num_classes默认为10，适用于10类分类任务（例如MNIST）。
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, x):
        """
        定义前向传播过程，描述数据通过网络的流动过程。

        参数：
        x (Tensor): 输入数据，形状为[batch_size, in_channels, img_size, img_size]。
                    这是一个4维张量，代表一批输入图像的批次，每个图像有in_channels个通道，大小为img_size x img_size。

        返回：
        Tensor: 输出数据，形状为[batch_size, num_classes]，表示每个样本在每个类别上的预测得分。
                这些得分通常会通过Softmax等操作转换为每个类别的概率。
        """
        # 第一层卷积：输入数据x经过卷积层conv1，输出16个特征图
        x = self.conv1(x)

        # 第一层ReLU激活：对卷积层的输出进行非线性激活
        x = self.relu1(x)

        # 第一层池化：对激活后的特征图进行最大池化，下采样尺寸
        x = self.pool1(x)

        # 第二层卷积：将池化后的特征图传入第二层卷积，输出16个特征图
        x = self.conv2(x)

        # 第二层ReLU激活：对卷积层的输出进行非线性激活
        x = self.relu2(x)

        # 第二层池化：对激活后的特征图进行最大池化
        x = self.pool2(x)

        # 将池化后的特征图展平为一维向量，方便传入全连接层
        x = self.flatten(x)

        # 第一层全连接层：将展平后的特征传入全连接层，输出32个神经元的激活值
        x = self.fc1(x)

        # 第三层ReLU激活：对全连接层的输出进行非线性激活
        x = self.relu3(x)

        # 第二层全连接层：输出类别的得分，num_classes个类别的得分
        x = self.fc2(x)

        # 最终返回每个输入样本在每个类别的得分（未进行Softmax等转换）
        return x
