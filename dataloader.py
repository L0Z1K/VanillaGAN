from torchvision import datasets, transforms
import torch.utils.data as data

class Dataset(data.Dataset):
    def __init__(self):
        super().__init__()
        self.dataset = datasets.MNIST(root="MNIST/",
                                      train=True,
                                      download=True,
                                      transform=transforms.Compose([
                                          transforms.ToTensor(),
                                          transforms.Normalize(mean=(0.5,), std=(0.5,))
                                      ]))
    
    def __getitem__(self, index):
        return self.dataset[index]
    
    def __len__(self):
        return len(self.dataset)

class DataLoader:
    def __init__(self, dataset):
        super(data.DataLoader).__init__()
        
        self.dataloader = data.DataLoader(dataset=dataset,
                                          batch_size=100,
                                          shuffle=True,
                                          drop_last=True)

if __name__ == "__main__":
    dataset = Dataset()
    data_loader = DataLoader(dataset)

    print(data_loader.dataloader.batch_size)
