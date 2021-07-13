import torch
from torch import nn
import torch.nn.utils.prune as prune
import torch.nn.functional as F
import numpy as np


# =============================================================
# NOTE: Begin Pruning Classifier
# =============================================================
dir_model = "/models_old\\"

model_name = "nabernet_ctx"
model = torch.load(dir_model + model_name + ".tar")

module = model.conv1

# prune.random_unstructured(module, name="weight", amount=0.3)
# prune.l1_unstructured(module, name="bias", amount=3)
# print(list(module.named_parameters()))
# print("=======================================")
# print(list(module.named_buffers()))

print(module.weight)
print("=======================================")
prune.ln_structured(module, name="weight", amount=0.5, n=2, dim=0)
print(module.weight)

print(model.state_dict().keys())
