import torch
import torch.nn as nn


class MyModel(nn.Module):
    def __init__(self):
        super(MyModel, self).__init__()

    def forward(self, input):
        output = input + 1
        return output


model = MyModel()
x = torch.tensor(1.0)
output = model(x)
print(output)