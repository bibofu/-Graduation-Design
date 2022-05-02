import torch
import torchvision
from torch.nn import Conv2d
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(
    "./dataset2",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True
)

dataloader = DataLoader(
    dataset=dataset,
    batch_size=64
)


class BiModel(torch.nn.Module):
    def __init__(self):
        super(BiModel, self).__init__()
        self.conv1 = Conv2d(in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0)

    def forward(self, x):
        x = self.conv1(x)
        return x


model = BiModel()

for data in dataloader:
    imgs , targets = data
    outdata=model(imgs)
    print(imgs.shape)
    print(outdata.shape)

