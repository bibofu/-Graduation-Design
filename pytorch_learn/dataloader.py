import torchvision
from torch.utils.data import DataLoader

# 数据集格式转化 img->tensor
dataset_trans = torchvision.transforms.Compose(
    [torchvision.transforms.ToTensor()]
)

test_trainSet = torchvision.datasets.CIFAR10(
    root="./dataset2",
    train=False,
    transform=dataset_trans,
    download=True
)

test_loader = DataLoader(
    dataset=test_trainSet,
    batch_size=64,
    shuffle=True,
    num_workers=0,
    drop_last=False

)

img, target = test_trainSet[0]
print(img.shape)
print(target)

for data in test_loader:
    imgs, targets = data
    print(imgs.shape)
    print(targets)
