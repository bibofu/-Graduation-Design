import torchvision
import torch
# 准备数据集
from torch.utils.data import DataLoader

from model import *

train_data = torchvision.datasets.CIFAR10(
    root="./dataset2",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

test_data = torchvision.datasets.CIFAR10(
    root="./dataset2",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

train_data_size = len(train_data)
test_data_size = len(test_data)

print("训练数据集的长度为:{}".format(train_data_size))
print("测试数据集的长度为:{}".format(test_data_size))

# 利用dataloader加载数据集
train_data_loader = DataLoader(
    dataset=train_data,
    batch_size=64
)

test_data_loader = DataLoader(
    dataset=test_data,
    batch_size=64
)

# 创建神经网络
model = Model()

# 损失函数
loss_fn = nn.CrossEntropyLoss()

# 优化器
learning_rate = 0.01
optimizer = torch.optim.SGD(model.parameters(), lr=learning_rate)

# 记录训练的次数
total_train_step = 0

# 记录测试的次数
total_test_step = 0

# 训练的轮数
epoch = 10

for i in range(epoch):
    print("-----第{}轮训练开始了-----".format(i + 1))

    # 训练步骤开始
    for data in train_data_loader:
        imgs, targets = data
        outputs = model(imgs)
        loss = loss_fn(outputs, targets)

        # 优化器优化模型
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_train_step = total_train_step + 1
        if total_train_step % 100 == 0:
            print("训练次数为:{},loss:{}".format(total_train_step, loss.item()))

    # 训练一轮后测试
    total_test_loss = 0.0
    with torch.no_grad():
        for data in test_data_loader:
            imgs, targets = data
            outputs = model(imgs)
            loss = loss_fn(outputs, targets)
            total_test_loss = total_test_loss + loss.item()
    print("测试集的loss:{}".format(total_test_loss))
