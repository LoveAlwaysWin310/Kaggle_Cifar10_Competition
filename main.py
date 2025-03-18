import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torchvision.transforms as T
import torchvision.models as models
from dataset.dataset import TrainDataset, TestDataset
from tqdm import tqdm
import matplotlib.pyplot as plt

# 记录每个 epoch 的 loss 和 acc
train_losses, train_accuracies = [], []
val_losses, val_accuracies = [], []

# 暂存原始函数引用
_original_train_epoch = None
_original_evaluate = None

def custom_train_epoch(*args, **kwargs):

    global train_losses, train_accuracies
    loss, acc = _original_train_epoch(*args, **kwargs)
    train_losses.append(loss)
    train_accuracies.append(acc)
    return loss, acc

def custom_evaluate(*args, **kwargs):

    global val_losses, val_accuracies
    loss, acc = _original_evaluate(*args, **kwargs)
    val_losses.append(loss)
    val_accuracies.append(acc)
    return loss, acc

num_epochs = 20
batch_size = 64
lr =  1e-3


# 数据预处理与增强
train_transform = T.Compose([
    T.Resize((224, 224)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])

test_transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225])
])


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc="Training", leave=True)
    for images, labels in loop:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item() * images.size(0)  # pytorch中的loss默认是平均损失，因此要乘以样本数得到总损失
        _, predicted = torch.max(outputs, 1)  # 找到最大的概率作为输出
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

        loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# 验证函数
def evaluate(model, dataloader, criterion, device):
    model.eval()  #评估模式
    running_loss = 0.0
    correct = 0
    total = 0

    loop = tqdm(dataloader, desc="Validating", leave=True)
    with torch.no_grad():
        for images, labels in loop:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            running_loss += loss.item() * images.size(0)
            _, predicted = torch.max(outputs, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            loop.set_postfix(loss=loss.item(), acc=100 * correct / total)

    epoch_loss = running_loss / total
    epoch_acc = correct / total
    return epoch_loss, epoch_acc

# 预测函数
def predict(model, dataloader, device):
    CIFAR10_LABELS = {
        0: "airplane",
        1: "automobile",
        2: "bird",
        3: "cat",
        4: "deer",
        5: "dog",
        6: "frog",
        7: "horse",
        8: "ship",
        9: "truck"
    }

    model.eval()
    predictions = []
    image_ids = []

    loop = tqdm(dataloader, desc="Predicting", leave=True)
    image_ids.extend(dataloader.dataset.image_ids)
    with torch.no_grad():
        for images, _ in loop:
            images = images.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            predicted_labels = [CIFAR10_LABELS[label.item()] for label in predicted]
            predictions.extend(predicted_labels)

            loop.set_postfix()

    return image_ids, predictions


def save_predictions(ids, predictions, filename="predictions.csv"):
    # 保存预测结果到 CSV 文件
    ids = list(np.array(ids, dtype=int) + 1)
    df = pd.DataFrame({
        'id': ids,
        'label': predictions
    })
    df.to_csv(filename, index=False)
    print(f"预测结果已保存到 {filename}")

def main():
    # 读取数据
    full_train_dataset = TrainDataset(csv_file='dataset/trainLabels.csv', root_dir='dataset/train', transform=train_transform)
    test_dataset = TestDataset(image_dir='dataset/test', transform=test_transform)

    # 划分验证集
    train_size = int(0.8 * len(full_train_dataset))
    val_size = len(full_train_dataset) - train_size
    train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=2)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)

    # 设备配置
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # 加载预训练 ResNet50
    model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(in_features=2048, out_features=10)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)


    # 训练和验证
    for epoch in range(num_epochs):
        print(f"\nEpoch [{epoch+1}/{num_epochs}]")
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")

    # 保存训练后的模型
    torch.save(model.state_dict(), "resnet50_model.pth")
    print("模型已保存！")

    # 进行预测
    ids, predictions = predict(model, test_loader, device)

    # 保存预测结果
    save_predictions(ids, predictions)


if __name__ == '__main__':

    _original_train_epoch = train_epoch
    _original_evaluate = evaluate

    train_epoch = custom_train_epoch
    evaluate = custom_evaluate

    main()

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(range(1, len(train_losses)+1), train_losses, label='Train Loss', marker='o')
    plt.plot(range(1, len(val_losses)+1), val_losses, label='Val Loss', marker='s')
    plt.title('Loss vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)

    # -- 绘制 Accuracy --
    plt.subplot(1, 2, 2)
    plt.plot(range(1, len(train_accuracies)+1), train_accuracies, label='Train Acc', marker='o')
    plt.plot(range(1, len(val_accuracies)+1), val_accuracies, label='Val Acc', marker='s')
    plt.title('Accuracy vs. Epoch')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True)

    plt.tight_layout()
    plt.show()
