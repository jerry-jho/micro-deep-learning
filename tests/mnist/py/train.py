import torch
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
from torch import optim
from torch.autograd import Variable as var
from net import cnn

traindt = datasets.FashionMNIST(
    root='data',
    train=True,
    transform=ToTensor(),
    download=True,
)
testdt = datasets.FashionMNIST(root='data', train=False, transform=ToTensor())

ldrs = {
    'train':
    torch.utils.data.DataLoader(traindt,
                                batch_size=100,
                                shuffle=False,
                                num_workers=1),
    'test':
    torch.utils.data.DataLoader(testdt,
                                batch_size=100,
                                shuffle=False,
                                num_workers=1),
}

Cnn = cnn()
lossfunct = nn.CrossEntropyLoss()


optim = optim.Adam(Cnn.parameters(), lr=0.01)

numepchs = 3
Cnn.train()

# Train the model
ttlstp = len(ldrs['train'])

for epoch in range(numepchs):
    for a, (imgs, lbls) in enumerate(ldrs['train']):
        ax = var(imgs)
        ay = var(lbls)
        outp = Cnn(ax)[0]
        losses = lossfunct(outp, ay)

        optim.zero_grad()
        losses.backward()

        optim.step()

        if (a + 1) % 100 == 0:
            print('Epoch [{}/{}], Step [{}/{}], Loss: {:.4f}'.format(
                epoch + 1, numepchs, a + 1, ttlstp, losses.item()))

torch.save(Cnn, 'cnn.pth')


