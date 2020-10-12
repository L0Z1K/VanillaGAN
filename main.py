import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from torch.autograd import Variable

transform=transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5,), std=(0.5,))
])

mnist_train = datasets.MNIST(root="MNIST/",
                             train=True,
                             download=True,
                             transform=transform)

mnist_test = datasets.MNIST(root="MNIST/",
                            train=False,
                            download=True,
                            transform=transform)

class Generator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(100, 256),
            nn.ReLU(),
            nn.Linear(256, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, 784),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.model(x)
        return x

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()

        self.model = nn.Sequential(
            nn.Linear(784, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        x = self.model(x)
        return x

batch_size=128
data_train = DataLoader(dataset=mnist_train,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)

G = Generator()
D = Discriminator()

optim_G = torch.optim.Adam(G.parameters(), lr=0.001)
optim_D = torch.optim.Adam(D.parameters(), lr=0.001)
criterion = nn.BCELoss()

device = 'cuda' if torch.cuda.is_available() else 'cpu'

Tensor = torch.cuda.FloatTensor if device is 'cuda' else torch.FloatTensor

total_epochs = 1
for epoch in range(total_epochs):
    for i, (x,_) in enumerate(data_train):
        real = Tensor(x.size(0), 1).fill_(1.0)
        fake = Tensor(x.size(0), 1).fill_(0.0)
        
        x = x.view(x.size(0), -1)

        noise = torch.rand(batch_size, 100)
        fake_img = G(noise)

        '''
        # Train Generator
        optim_G.zero_grad()
        g_cost = criterion(D(fake_img), real)
        print(g_cost)
        g_cost.backward(retain_graph=True)
        optim_G.step()

        '''
        # Train Discriminator
        optim_D.zero_grad()
        d_cost = criterion(D(torch.cat((x, fake_img))), torch.cat((real, fake)))
        d_cost.backward(retain_graph=True)
        optim_D.step()
        
        break