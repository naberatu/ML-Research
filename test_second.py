# import torch
# import torch.nn as nn
# import torch.nn.functional as F
#
#
# # For this NN, we are trying to filter out the Red Green and Blue from the image.
# class SimpleNet(nn.Module):
#     def __init__(self, num_classes=10):
#         super(SimpleNet, self).__init__()
#
#         # 3 in-channels because of RGB
#         # 12 feature detectors using the out-channels
#         # Size 3 Kernel is a standard.
#         # Should stay as 1 unless we want to shrink image.
#         self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=3, stride=1, padding=1)
#         self.relu1 = nn.ReLU()
#
#         self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=3, stride=1, padding=1)
#         self.relu2 = nn.ReLU()
#
#         self.pool = nn.MaxPool2d(kernel_size=2)
#
#         self.conv3 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=3, stride=1, padding=1)
#         self.relu3 = nn.ReLU()
#
#         self.conv4 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=3, stride=1, padding=1)
#         self.relu4 = nn.ReLU()
#
#         self.fc = nn.Linear(in_features=16 * 16 * 24, out_features=num_classes)
#
#     def forward(self, input):
#         output = self.conv1(input)
#         output = self.relu1(output)
#
#         output = self.conv2(output)
#         output = self.relu2(output)
#
#         output = self.pool(output)
#
#         output = self.conv3(output)
#         output = self.relu3(output)
#
#         output = self.conv4(output)
#         output = self.relu4(output)
#
#         output = output.view(-1, 16 * 16 * 24)
#
#         output = self.fc(output)
#
#         return output
#
