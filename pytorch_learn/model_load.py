import torch

# 方式一，保存方式一，加载模型
# model = torch.load("vgg16_method1.pth")
# print(model)

# 方式二，保存方式二
import torchvision

vgg16 = torchvision.models.vgg16(pretrained=False)
vgg16.load_state_dict(torch.load("vgg16_method2.pth"))

print(vgg16)
