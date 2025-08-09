import torch
import torch.nn as nn

class SimpleViT(nn.Module):
    """
    极简Vision Transformer (ViT) 模型。该模型的结构基于Transformer，并使用了图像切分成小块（patch）的方法来处理图像数据。

    结构：Patch切分 -> 线性映射 -> 加位置编码 -> Transformer编码器 -> 分类头
    """

    def __init__(self, image_size=28, patch_size=7, num_classes=10, dim=16, depth=1, heads=2, mlp_dim=32, channels=3):
        """
        初始化Vision Transformer模型的构建。此部分定义了模型中的各个层和参数。

        参数：
        image_size (int): 输入图像的大小（例如，对于MNIST，通常是28x28）。该参数需要能够被patch_size整除。
        patch_size (int): 每个图像块（patch）的大小，默认为7。用于切分图像的区域。
        num_classes (int): 输出类别的数量，默认为10，适用于10分类任务（如MNIST）。
        dim (int): Transformer编码器的嵌入维度（即每个token的维度），默认为16。
        depth (int): Transformer编码器的层数（即编码器堆叠的层数），默认为1。
        heads (int): 每个Transformer层中多头自注意力机制的头数，默认为2。
        mlp_dim (int): Transformer中的前馈网络的维度，默认为32。
        channels (int): 输入图像的通道数，默认为3（适用于RGB图像）。
        """
        super().__init__()

        # 检查输入图像大小是否能被patch_size整除
        assert image_size % patch_size == 0, "Image dimensions must be divisible by the patch size."

        # 计算每个patch的数量和每个patch的维度
        num_patches = (image_size // patch_size) ** 2  # 计算图像中有多少个patch
        patch_dim = channels * patch_size * patch_size  # 计算每个patch的维度（通道数 * patch的宽高）

        # 保存一些参数
        self.patch_size = patch_size
        self.dim = dim

        # Patch切分后的映射：将每个patch映射到一个向量空间
        self.to_patch_embedding = nn.Linear(patch_dim, dim)  # 每个patch通过线性变换映射到dim维空间

        # 位置编码：每个patch有一个位置编码，Transformer需要知道每个patch在图像中的相对位置
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # 加入1个cls token的位置信息

        # cls token：用于表示图像的全局特征，这个token会在整个Transformer过程中传递
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # 生成一个初始的cls token

        # Transformer编码器：堆叠若干个Transformer Encoder层
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=heads, dim_feedforward=mlp_dim, batch_first=True),
            num_layers=depth  # 堆叠的Transformer层数
        )

        # 分类头：将Transformer的输出映射到类别数
        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),  # 对Transformer的输出进行LayerNorm
            nn.Linear(dim, num_classes)  # 映射到类别数
        )

    def forward(self, x):
        """
        定义前向传播过程，描述数据通过网络的流动过程。

        参数：
        x (Tensor): 输入数据，形状为[batch_size, channels, height, width]。
                    对于图像数据，通常为一个批次的图像，每张图像有channels个通道，大小为height x width。

        返回：
        Tensor: 输出数据，形状为[batch_size, num_classes]，表示每个样本在每个类别上的预测得分。
                这些得分通常会通过Softmax等操作转化为每个类别的概率。
        """
        B, C, H, W = x.shape  # 获取批次大小、通道数、高度和宽度

        patch_size = self.patch_size

        # 1. Patch切分：将输入图像切分为若干小块（patch）
        x = x.unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)  # unfold操作用于切分图像
        x = x.contiguous().view(B, C, -1, patch_size, patch_size)  # 展平为一系列patch
        x = x.permute(0, 2, 1, 3, 4).contiguous().view(B, -1, C * patch_size * patch_size)  # 转置并展平成一维向量

        # 2. Patch embedding：将每个patch映射到一个向量空间
        x = self.to_patch_embedding(x)  # 使用线性层将每个patch映射到一个维度为dim的向量

        # 3. 拼接cls token：将一个特殊的"cls token"加入到输入序列的前面
        cls_tokens = self.cls_token.expand(B, -1, -1)  # 扩展cls token到batch size
        x = torch.cat((cls_tokens, x), dim=1)  # 将cls token和patch嵌入拼接

        # 4. 加位置编码：为每个patch和cls token加上位置编码，帮助模型理解各个patch的位置
        x = x + self.pos_embedding[:, :x.size(1)]  # 对输入加上位置编码，确保位置编码与输入序列长度匹配

        # 5. Transformer编码：使用Transformer编码器处理输入的序列
        x = self.transformer(x)  # 将包含位置编码和cls token的序列输入Transformer

        # 6. 取cls token的输出：Transformer输出序列的第一个位置是cls token的输出
        out = self.mlp_head(x[:, 0])  # 获取第一个token的输出（即cls token），并通过分类头进行处理

        return out  # 返回最终的分类结果

# 定义ViT模型
def ViT(num_classes=10):
    """
    返回一个SimpleViT模型实例，适用于指定类别数的图像分类任务。
    """
    return SimpleViT(num_classes=num_classes)
