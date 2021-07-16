
import torch.nn as nn
import torch.nn.functional as F

# =============================================================
# MODEL LAYER PARAMETERS
# =============================================================
# cdims = [64, 64, 64, 64, 64]
# cdims = [64, 64, 128, 256, 512]

cdims = [32, 48, 64, 64, 64]     # Was used for nabernet_c1
# cdims = [30, 45, 60, 75, 90]     # Was used for nabernet_c2
fdims = [128, 84, 2]            # Was used for nabernet_c1
# fdims = [100, 64, 3]             # Was used for nabernet_c2
IMF = [59, 59, 59]
# =============================================================


# =============================================================
# NOTE MODEL CREATION
# =============================================================
class NaberNet(nn.Module):
    def __init__(self, group=0):        # NOTE: Don't forget to give a group number!
        super(NaberNet, self).__init__()
        self.pool = nn.MaxPool2d(2, 2)
        self.IMF = IMF[group]

        self.conv1 = nn.Conv2d(3, cdims[0], 3)
        self.conv2 = nn.Conv2d(cdims[0], cdims[1], 3)
        self.conv3 = nn.Conv2d(cdims[1], cdims[2], 3)
        self.conv4 = nn.Conv2d(cdims[2], cdims[3], 3)
        self.conv5 = nn.Conv2d(cdims[3], cdims[4], 3)
        self.fc1 = nn.Linear(cdims[4] * self.IMF * self.IMF, fdims[0])
        self.fc2 = nn.Linear(fdims[0], fdims[1])
        self.fc3 = nn.Linear(fdims[1], fdims[2])

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = self.pool(F.relu(self.conv2(x)))
        x = F.relu(self.conv3(x))
        x = self.pool(F.relu(self.conv4(x)))
        x = F.relu(self.conv5(x))

        # print(x.shape)                # For debugging purposes.

        x = x.view(-1, cdims[4] * self.IMF * self.IMF)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x
