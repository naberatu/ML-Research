
import torch.nn as nn
import torch.nn.functional as F
IMF = 17     # For quickly modifying dimensional parameters


class Net(nn.Module):
    # CIFAR / STL Network
    def __init__(self):
        super(Net, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        # self.conv1 = nn.Conv2d(3, 32, 5)
        # self.conv2 = nn.Conv2d(32, 32, 5)
        # self.conv3 = nn.Conv2d(32, 32, 5)
        self.conv1 = nn.Conv2d(3, 96, 5)
        self.conv2 = nn.Conv2d(96, 96, 5)
        self.conv3 = nn.Conv2d(96, 96, 5)
        self.fc1 = nn.Linear(96 * IMF * IMF, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = x.view(-1, 96 * IMF * IMF)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
