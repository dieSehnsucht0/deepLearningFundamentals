import torch.nn as nn

class MLP(nn.Module):
    """
    多层感知机（MLP）模型，用于小图片分类任务。它由多个全连接层组成，
    每一层之间使用ReLU激活函数引入非线性。
    
    结构：输入展平 -> 第一全连接层 -> ReLU激活函数 -> 第二全连接层 -> 输出
    """

    def __init__(self, input_size, hidden_size=32, num_classes=10):
        """
        初始化多层感知机模型的构建。此部分定义了模型中的各个层以及其参数。
        
        参数：
        input_size (int): 输入数据的大小，通常是输入图像的像素数。
                           例如，若是28x28的灰度图像，input_size应为28*28=784。
        hidden_size (int): 隐藏层的神经元个数，默认为32。隐藏层决定了模型的学习能力，
                           神经元越多，模型能够拟合的数据复杂度也越高。
        num_classes (int): 输出类别的数量，默认为10，适用于MNIST数据集（数字分类：0到9）。
        """
        super().__init__()  # 调用父类nn.Module的初始化方法。它负责处理神经网络中一些基本操作，如模型保存和加载。

        # 定义模型的各个层：
        
        # nn.Flatten()：将输入的多维张量展平成一维向量。对于输入图像数据，展平是为了能够将其传入全连接层。
        # 比如，输入是一个28x28的图片，经过展平后就变成一个784长度的向量。
        self.flatten = nn.Flatten()

        # nn.Linear(input_size, hidden_size)：第一层全连接层，它将输入的特征（input_size个元素）
        # 映射到隐藏层（hidden_size个神经元）。每个神经元有一组权重和一个偏置参数，学习这些参数是网络训练的核心。
        # 全连接层的作用是线性变换：y = xW + b，其中W是权重矩阵，b是偏置项，x是输入。
        self.fc1 = nn.Linear(input_size, hidden_size)

        # nn.ReLU()：ReLU激活函数。ReLU（Rectified Linear Unit）是一个常用的激活函数，
        # 它的输出是：f(x) = max(0, x)，即输入小于0时输出0，输入大于0时输出本身。
        # ReLU激活函数能够使得神经网络引入非线性，使得模型能够拟合复杂的关系。
        self.relu = nn.ReLU()

        # nn.Linear(hidden_size, num_classes)：第二层全连接层，将隐藏层的输出映射到输出层。
        # 输出层的大小为num_classes，代表了最终分类任务的类别数。
        # 对于MNIST数据集，num_classes是10，因为我们有10个数字类别（0-9）。
        self.fc2 = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        """
        定义前向传播过程，也就是数据通过神经网络时的流动过程。
        在这个过程中，数据会依次经过各层的计算，最终得到网络的输出。

        参数：
        x (Tensor): 输入数据，形状为[batch_size, channels, height, width]。这是一个4维张量，
                    对于图片数据，通常为一个批次的图像。这里的`batch_size`代表一次性输入的图像数量，
                    `channels`代表图像的通道数（如灰度图为1，RGB彩色图为3），`height`和`width`是图像的高和宽。

        返回：
        Tensor: 输出数据，形状为[batch_size, num_classes]，表示每个输入样本在每个类别上的预测得分。
                对于分类任务，通常这些得分将经过Softmax等操作转化为概率值。
        """
        # 第一部分：输入数据展平。输入图像为一个4维张量（batch_size, channels, height, width），
        # 在传入全连接层之前需要展平成2维张量（batch_size, input_size），
        # 这样每个图像的所有像素值就会变成一个长向量，方便进行全连接操作。
        x = self.flatten(x)

        # 第二部分：通过第一层全连接层（fc1）。这一层会将展平后的输入向量（形状为[batch_size, input_size]）
        # 映射到一个新的空间，得到大小为[batch_size, hidden_size]的输出。
        # 其中hidden_size是隐藏层神经元的数量。
        x = self.fc1(x)

        # 第三部分：应用ReLU激活函数。通过ReLU将输入的线性变换结果进行非线性激活，生成下一层的输入。
        x = self.relu(x)

        # 第四部分：通过第二层全连接层（fc2）。这一层将隐藏层的输出（形状为[batch_size, hidden_size]）
        # 映射到输出层（形状为[batch_size, num_classes]），输出每个类别的得分。
        # 这些得分通常会在后续通过Softmax转化为每个类别的概率。
        x = self.fc2(x)

        # 最终，返回网络的输出，它表示每个类别的得分。
        return x
