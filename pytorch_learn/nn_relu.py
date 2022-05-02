import torch
from torch.nn import ReLU

input = torch.tensor([[1, -0.5],
                      [-1, 3]])
print(input.shape)

input = torch.reshape(input, (-1, 1, 2, 2))

print(input.shape)


class Model(torch.nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.relu1 = ReLU()

    def forward(self, input):
        output = self.relu1(input)
        return output


model = Model()
res = model(input)
print(res)
