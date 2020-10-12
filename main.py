import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torchvision.utils import save_image, make_grid
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
            nn.Dropout(p=0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self,x):
        x = self.model(x)
        return x

batch_size=100
data_train = DataLoader(dataset=mnist_train,
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

G = Generator().to(device)
D = Discriminator().to(device)

optim_G = torch.optim.Adam(G.parameters(), lr=0.0001)
optim_D = torch.optim.Adam(D.parameters(), lr=0.0001)
criterion = nn.BCELoss()

print("[+] Train Start")
total_epochs = 100
total_batch = len(data_train)
for epoch in range(total_epochs):
    avg_cost = [0, 0]
    for i, (x,_) in enumerate(data_train):
        real = (torch.FloatTensor(x.size(0), 1).fill_(1.0)).to(device)
        fake = (torch.FloatTensor(x.size(0), 1).fill_(0.0)).to(device)
        
        x = x.view(x.size(0), -1).to(device)

        noise = torch.rand(batch_size, 100).to(device)

        fake_img = G(noise)
        # Train Generator
        optim_G.zero_grad()
        g_cost = criterion(D(fake_img), real)
        g_cost.backward()
        optim_G.step()

        fake_img = fake_img.detach().to(device)
        # Train Discriminator
        optim_D.zero_grad()
        y_pred= torch.cat((x, fake_img))
        y_pred = D(y_pred)
        y = torch.cat((real, fake))
        d_cost = criterion(y_pred, y)
        d_cost.backward()
        optim_D.step()

        avg_cost[0] += g_cost
        avg_cost[1] += d_cost
    avg_cost[0] /= total_batch
    avg_cost[1] /= total_batch
    print("Epoch: %d, Generator: %f, Discriminator: %f"%(epoch+1, avg_cost[0], avg_cost[1]))
    
    fake_img = fake_img.reshape([100, 1, 28, 28])
    img_grid = make_grid(fake_img, nrow=10, normalize=True)
    save_image(img_grid, "image/%d.png"%(epoch+1))
