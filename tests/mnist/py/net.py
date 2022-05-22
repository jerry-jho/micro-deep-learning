import torch.nn as nn

class cnn(nn.Module):
    def __init__(self):
        super(cnn, self).__init__()
        self.conv0 = nn.Sequential(
            nn.Conv2d(
                in_channels=1,
                out_channels=16,
                kernel_size=5,
                stride=1,
                padding=2,
            ),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.conv1 = nn.Sequential(
            nn.Conv2d(16, 32, 5, 1, 2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.out = nn.Linear(32 * 7 * 7, 10)

    def forward(self, y):
        y = self.conv0(y)
        y = self.conv1(y)
        y = y.view(y.size(0), -1)
        outp = self.out(y)
        return outp, y