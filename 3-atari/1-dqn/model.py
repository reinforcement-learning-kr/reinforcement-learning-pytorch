import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self, num_outputs):
        self.num_outputs = num_outputs
        super(Model, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=4,
                               out_channels=32,
                               kernel_size=8,
                               stride=4)

        self.conv2 = nn.Conv2d(in_channels=32,
                               out_channels=64,
                               kernel_size=4,
                               stride=2)

        self.conv3 = nn.Conv2d(in_channels=64,
                               out_channels=64,
                               kernel_size=3,
                               stride=1)

        self.fc1 = nn.Linear(7*7*64, 512)
        self.fc2 = nn.Linear(512, num_outputs)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        qvalue = self.fc2(x)
        return qvalue