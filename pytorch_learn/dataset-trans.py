import torchvision

# 数据集格式转化 img->tensor
dataset_trans = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)
 
train_set = torchvision.datasets.CIFAR10(
    root="./dataset2",
    train=True,
    transform=dataset_trans,
    download=True
)

test_set = torchvision.datasets.CIFAR10(
    root="./dataset2",
    train=False,
    transform=dataset_trans,
    download=True
)

# print(train_set[0])
# print(train_set.classes)
# print(len(train_set.classes))
#
# img, target = train_set[100]
# print(img)
# print(target)
print(test_set[0])