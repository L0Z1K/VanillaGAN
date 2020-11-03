import torch
import torch.nn as nn

class Loss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss = nn.BCELoss()
    
    def forward(self, x, y):
        return self.loss(x, y)

if __name__ == "__main__":
    x = torch.rand(3, 1)
    y = torch.rand(3, 1)
    loss = Loss()
    
    print(loss(x, y))