import torch
import os
from utils import evaluate, plot_losses, plot_accs,init_tensorboard_writer, log_metrics, close_tensorboard_writer

def train_model(model, train_loader, test_loader, optimizer, criterion, device, epochs, label):
    """
    训练模型，并在每个epoch结束后评估模型性能。该函数还会保存最佳模型，并绘制损失和准确率曲线。

    参数：
    model (nn.Module): 要训练的模型。
    train_loader (DataLoader): 训练集的DataLoader。
    test_loader (DataLoader): 测试集的DataLoader。
    optimizer (torch.optim.Optimizer): 用于优化模型参数的优化器。
    criterion (torch.nn.Module): 损失函数，用于计算模型的误差。
    device (torch.device): 训练时使用的设备（例如GPU或CPU）。
    epochs (int): 训练的总轮数。
    label (str): 用于标识当前模型训练的标签，将用于文件命名等。

    返回：
    best_acc (float): 模型在测试集上的最佳准确率。
    """
    torch.cuda.empty_cache()  # 清空CUDA缓存，防止内存泄漏
    model.to(device)  # 将模型加载到指定设备（CPU或GPU）
    losses = []  # 用于记录每个epoch的损失
    accs = []  # 用于记录每个epoch的准确率
    best_acc = 0.0  # 初始化最佳准确率为0

    # 创建保存结果的文件夹
    os.makedirs("results/loss_curves", exist_ok=True)  # 如果文件夹不存在，则创建
    os.makedirs("results/models", exist_ok=True)  # 如果文件夹不存在，则创建
    # 初始化 TensorBoard writer
    writer = init_tensorboard_writer(label)

    # 写入网络结构（拿一批数据作为示例输入）
    sample_data, _ = next(iter(train_loader))
    try:
        writer.add_graph(model, sample_data.to(device))
    except Exception as e:
        print(f"add_graph failed: {e}")
        print("模型结构如下：")
        print(model)  # 打印模型结构
    # 训练过程
    for epoch in range(epochs):
        model.train()  # 将模型设置为训练模式
        total_loss = 0  # 初始化该epoch的总损失

        # 遍历训练集的所有batch
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)  # 将输入和标签转移到指定设备（GPU或CPU）
            optimizer.zero_grad()  # 清空梯度
            pred = model(x)  # 前向传播，得到模型的预测结果
            loss = criterion(pred, y)  # 计算损失
            loss.backward()  # 反向传播，计算梯度
            optimizer.step()  # 更新模型参数

            total_loss += loss.item()  # 累加当前batch的损失

        avg_loss = total_loss / len(train_loader)  # 计算该epoch的平均损失
        acc = evaluate(model, test_loader, device)  # 在测试集上评估模型的准确率
        log_metrics(label, epoch, avg_loss, acc)  # 记录训练指标

        print(f"[{label}] Epoch {epoch+1}/{epochs} - Loss: {avg_loss:.4f}, Test Acc: {acc:.4f}", flush=True)

        # 保存最佳模型（基于测试准确率）
        if acc > best_acc:
            best_acc = acc
            torch.save(model.state_dict(), f"results/models/{label}_best.pth")  # 保存模型权重

        losses.append(avg_loss)  # 记录每个epoch的平均损失
        accs.append(acc)  # 记录每个epoch的准确率

        torch.cuda.empty_cache()  # 每个epoch结束后清空CUDA缓存，防止内存泄漏

    close_tensorboard_writer(label)  # 关闭TensorBoard writer

    # 绘制损失曲线和准确率曲线，并保存到文件
    plot_losses(losses, f"results/loss_curves/{label}_loss.png")  # 绘制并保存损失曲线
    plot_accs(accs, f"results/loss_curves/{label}_acc.png")  # 绘制并保存准确率曲线

    return best_acc  # 返回最佳准确率
