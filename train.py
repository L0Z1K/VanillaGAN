import torch
import torch.nn as nn
from torchvision.utils import save_image, make_grid
from torch.autograd import Variable

from dataloader import *
from model import *
from loss import *

dataset = Dataset()
data_loader = DataLoader(dataset)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

G = Generator().to(device)
D = Discriminator().to(device)

optim_G = torch.optim.Adam(G.parameters(), lr=0.0001)
optim_D = torch.optim.Adam(D.parameters(), lr=0.0001)
criterion = Loss()

print("[+] Train Start")
total_epochs = 200
total_batch = len(data_loader.dataloader)
for epoch in range(total_epochs):
    avg_cost = [0, 0]
    for x, _ in data_loader.dataloader:
        real = (torch.FloatTensor(x.size(0), 1).fill_(1.0)).to(device)
        fake = (torch.FloatTensor(x.size(0), 1).fill_(0.0)).to(device)
        
        x = x.view(x.size(0), -1).to(device)

        noise = torch.randn(data_loader.dataloader.batch_size, 100, device=device)

        fake_img = G(noise)

        # Train Generator
        optim_G.zero_grad()
        g_cost = criterion(D(fake_img), real)
        g_cost.backward()
        optim_G.step()

        fake_img = fake_img.detach().to(device)
        # Train Discriminator
        optim_D.zero_grad()
        d_cost = criterion(D(torch.cat((x, fake_img))), torch.cat((real, fake)))
        d_cost.backward()
        optim_D.step()

        avg_cost[0] += g_cost
        avg_cost[1] += d_cost
    
    avg_cost[0] /= total_batch
    avg_cost[1] /= total_batch

    if (epoch+1) % 10 == 0:
        fake_img = fake_img.reshape([batch_size, 1, 28, 28])
        img_grid = make_grid(fake_img, nrow=10, normalize=True)
        save_image(img_grid, "result/%d.png"%(epoch+1))
        print("Epoch: %d, Generator: %f, Discriminator: %f"%(epoch+1, avg_cost[0], avg_cost[1]))