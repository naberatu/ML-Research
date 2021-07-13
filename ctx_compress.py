import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F


# =============================================================
# NOTE: Begin Pruning Classifier
# =============================================================
dir_model = "C:\\Users\\elite\\PycharmProjects\\Pytorch\\models\\"

model_name = "nabernet_ctx"

model = torch.load(dir_model + model_name + ".tar")
print(model.conv1)
