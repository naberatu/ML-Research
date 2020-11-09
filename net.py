
import torch.nn as nn
import torch.nn.functional as F
IMF = 1     # For quickly modifying dimensional parameters


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(32, 32, 5)
        self.conv3 = nn.Conv2d(32, 32, 5)
        # self.conv4 = nn.Conv2d(32, 32, 5)
        self.fc1 = nn.Linear(32 * IMF * IMF, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        # print(x.shape)
        # x = F.relu(self.conv4(x))
        x = x.view(-1, 32 * IMF * IMF)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
