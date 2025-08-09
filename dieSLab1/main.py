import os
import csv
import random
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from models.mlp import MLP
from models.cnn import CNN
from models.vit import ViT
from train import train_model
from config import model_configs, dataset_names

# 设置随机种子，以保证实验可复现
random.seed(42)  # 设置Python的随机数种子
torch.manual_seed(42)  # 设置PyTorch的随机数种子

# 设置设备，优先用CUDA（GPU），如果没有就用CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 创建保存实验结果的文件夹
os.makedirs("results/loss_curves", exist_ok=True)  # 创建保存loss曲线的目录
os.makedirs("results", exist_ok=True)  # 创建保存最终结果的目录

def get_data(dataset, batch_size=8):
    """
    获取指定数据集的训练集和测试集，并根据输入的batch_size进行数据加载。

    参数：
    dataset (str): 需要加载的数据集，支持 "mnist" 或 "cifar10"。
    batch_size (int): 每个batch的大小，默认值为8。
    """
    if dataset == "mnist":
        # 对MNIST数据集的图像进行预处理：将灰度图像转换为3通道图像，并转为Tensor格式
        transform = transforms.Compose([
            transforms.ToTensor(),  # 将图像转换为Tensor
            transforms.Lambda(lambda x: x.repeat(3, 1, 1))  # 重复通道，转为3通道（伪RGB）
        ])
        # 下载并加载训练集和测试集
        train_set = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
        test_set = datasets.MNIST(root="./data", train=False, download=True, transform=transform)
        # 图像的通道数是3，输入大小是3*28*28（每个图像有3个28x28的像素）
        in_channels = 3
        input_size = 3 * 28 * 28
        img_size = 28
    elif dataset == "cifar10":
        # 对CIFAR-10数据集的图像进行预处理：随机水平翻转、转换为Tensor格式，并进行归一化
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),  # 随机水平翻转图像
            transforms.ToTensor(),  # 将图像转换为Tensor
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # 对图像进行归一化处理
        ])
        # 下载并加载训练集和测试集
        train_set = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
        test_set = datasets.CIFAR10(root="./data", train=False, download=True, transform=transform)
        # 图像的通道数是3，输入大小是3*32*32（每个图像有3个32x32的像素）
        in_channels = 3
        input_size = 3 * 32 * 32
        img_size = 32
    else:
        raise ValueError("Unknown dataset")  # 如果输入的数据集不是"mnist"或"cifar10"，则抛出错误
    """
    返回：
    train_loader (DataLoader): 训练集的DataLoader。
    test_loader (DataLoader): 测试集的DataLoader。
    in_channels (int): 图像的通道数。
    input_size (int): 图像输入的大小（展平后的特征数）。
    img_size (int): 输入图像的大小（宽/高）。
    """
    return DataLoader(train_set, batch_size=batch_size, shuffle=True), DataLoader(test_set, batch_size=batch_size), in_channels, input_size, img_size

# 保存实验结果的列表
results = []

# 遍历数据集，进行实验
for dataset in dataset_names:
    num_classes = 10  # 设置每个数据集的类别数

    # MLP
    for batch_size in model_configs["MLP"]["batch_sizes"]:
        # 获取训练和测试数据
        train_loader, test_loader, in_channels, input_size, img_size = get_data(dataset, batch_size)
        for lr in model_configs["MLP"]["learning_rates"]:
            for hidden_size in model_configs["MLP"]["hidden_sizes"]:
                for epoch in model_configs["MLP"]["epochs"]:
                    # 实例化MLP
                    model = MLP(input_size=input_size, hidden_size=hidden_size, num_classes=num_classes)
                    # 设置标签，用于标识当前实验的参数
                    tag = f"{dataset}_mlp_bs{batch_size}_lr{lr}_hs{hidden_size}_ep{epoch}"
                    # 训练模型并获取准确率
                    acc = train_model(
                        model, train_loader, test_loader,
                        optim.Adam(model.parameters(), lr=lr),  # 使用Adam优化器
                        nn.CrossEntropyLoss(),  # 使用交叉熵损失函数
                        device,  # 使用指定的设备（GPU或CPU）
                        epoch,  # 训练的epoch数
                        tag  # 实验的标识
                    )
                    # 将结果保存到results列表中
                    results.append([dataset, "MLP", batch_size, lr, hidden_size, epoch, acc])

    # CNN
    for batch_size in model_configs["CNN"]["batch_sizes"]:
        # 获取训练和测试数据
        train_loader, test_loader, in_channels, input_size, img_size = get_data(dataset, batch_size)
        for lr in model_configs["CNN"]["learning_rates"]:
            for epoch in model_configs["CNN"]["epochs"]:
                # 实例化CNN
                model = CNN(in_channels=in_channels, num_classes=num_classes, img_size=img_size)
                # 设置标签，用于标识当前实验的参数
                tag = f"{dataset}_cnn_bs{batch_size}_lr{lr}_ep{epoch}"
                # 训练模型并获取准确率
                acc = train_model(
                    model, train_loader, test_loader,
                    optim.Adam(model.parameters(), lr=lr),  # 使用Adam优化器
                    nn.CrossEntropyLoss(),  # 使用交叉熵损失函数
                    device,  # 使用指定的设备（GPU或CPU）
                    epoch,  # 训练的epoch数
                    tag  # 实验的标识
                )
                # 将结果保存到results列表中
                results.append([dataset, "CNN", batch_size, lr, "-", epoch, acc])

    # ViT
    for batch_size in model_configs["ViT"]["batch_sizes"]:
        # 获取训练和测试数据
        train_loader, test_loader, in_channels, input_size, img_size = get_data(dataset, batch_size)
        for lr in model_configs["ViT"]["learning_rates"]:
            for epoch in model_configs["ViT"]["epochs"]:
                # 实例化ViT模型
                model = ViT(num_classes=num_classes)
                # 设置标签，用于标识当前实验的参数
                tag = f"{dataset}_vit_bs{batch_size}_lr{lr}_ep{epoch}"
                # 训练模型并获取准确率
                acc = train_model(
                    model, train_loader, test_loader,
                    optim.Adam(model.parameters(), lr=lr),  # 使用Adam优化器
                    nn.CrossEntropyLoss(),  # 使用交叉熵损失函数
                    device,  # 使用指定的设备（GPU或CPU）
                    epoch,  # 训练的epoch数
                    tag  # 实验的标识
                )
                # 将结果保存到results列表中
                results.append([dataset, "ViT", batch_size, lr, "-", epoch, acc])

# 将实验结果保存为CSV文件
results_file = "results/performance.csv"
with open(results_file, "w", newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        "Dataset", "Model", "Batch Size", "Learning Rate", "Hidden Size", "Epochs", "Test Accuracy"
    ])  # 写入CSV文件的表头
    for row in results:
        writer.writerow(row)  # 写入每一行的实验结果

print(f"实验完成，结果保存至 {results_file}")
