#
import torch.nn as nn
import torch.nn.functional as F


IMF = 51     # For quickly modifying dimensional parameters


class NaberNet(nn.Module):
    def __init__(self):
        super(NaberNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        # self.bn = nn.BatchNorm2d(64)

        # Custom-made network
        self.conv1 = nn.Conv2d(3, 64, 3)
        self.conv2 = nn.Conv2d(64, 64, 3)
        self.conv3 = nn.Conv2d(64, 64, 3)
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.conv5 = nn.Conv2d(64, 64, 3)
        self.fc1 = nn.Linear(64 * IMF * IMF, 128)
        self.fc2 = nn.Linear(128, 84)
        self.fc3 = nn.Linear(84, 2)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))

        # print(x.shape)                # For debugging purposes.

        x = x.view(-1, 64 * IMF * IMF)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
