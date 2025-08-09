import torch
import matplotlib.pyplot as plt
from torch.utils.tensorboard import SummaryWriter

writers={}

def evaluate(model, loader, device):
    """
    评估模型在给定数据集上的准确率。

    参数：
    model (nn.Module): 需要评估的模型。
    loader (DataLoader): 数据加载器，用于加载测试集或验证集数据。
    device (torch.device): 设备（CPU或GPU），模型和数据将被加载到此设备。

    返回：
    float: 模型在测试集上的准确率。
    """
    model.eval()  # 设置模型为评估模式（禁用Dropout和BatchNorm）
    correct = 0  # 初始化正确预测的数量
    total = 0  # 初始化总样本数
    with torch.no_grad():  # 在评估时禁用梯度计算（减少内存消耗）
        for x, y in loader:
            x, y = x.to(device), y.to(device)  # 将输入和标签转移到指定设备
            pred = model(x)  # 获取模型的预测
            correct += (pred.argmax(1) == y).sum().item()  # 统计预测正确的样本数
            total += y.size(0)  # 累加样本总数
    return correct / total  # 返回准确率

def plot_losses(losses, path):
    """
    绘制训练损失曲线并保存为图片。

    参数：
    losses (list): 每个epoch的训练损失。
    path (str): 图片保存路径。

    """
    plt.plot(losses)  # 绘制损失曲线
    plt.xlabel("Epoch")  # x轴标签：Epoch
    plt.ylabel("Loss")  # y轴标签：Loss
    plt.title("Training Loss")  # 图表标题
    plt.grid(True)  # 显示网格
    plt.savefig(path)  # 保存图表为图片
    plt.close()  # 关闭当前图表，释放内存

def plot_accs(accs, path):
    """
    绘制测试准确率曲线并保存为图片。

    参数：
    accs (list): 每个epoch的测试准确率。
    path (str): 图片保存路径。

    """
    plt.plot(accs)  # 绘制准确率曲线
    plt.xlabel("Epoch")  # x轴标签：Epoch
    plt.ylabel("Accuracy")  # y轴标签：Accuracy
    plt.title("Test Accuracy")  # 图表标题
    plt.grid(True)  # 显示网格
    plt.savefig(path)  # 保存图表为图片
    plt.close()  # 关闭当前图表，释放内存

#初始化TensorBoard writer
def init_tensorboard_writer(label):
    """
    初始化TensorBoard writer

    参数：
    label (str): 实验标签

    返回：
    SummaryWriter: TensorBoard writer实例
    """
    writer = SummaryWriter(f'runs/{label}')
    writers[label] = writer
    return writer

# 记录训练指标
def log_metrics(label, epoch, loss, accuracy):
    """
    记录训练指标到TensorBoard

    参数：
    label (str): 实验标签
    epoch (int): 当前epoch
    loss (float): 训练损失
    accuracy (float): 测试准确率

    """
    if label in writers:
        writer = writers[label]
        writer.add_scalar('Loss/train', loss, epoch)
        writer.add_scalar('Accuracy/test', accuracy, epoch)

# 关闭TensorBoard writer
def close_tensorboard_writer(label):
    """
    关闭TensorBoard writer

    参数：
    label (str): 实验标签

    """
    if label in writers:
        writers[label].close()
        del writers[label]