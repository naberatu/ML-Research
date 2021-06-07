import math

import torch
import torch.nn.functional as tf
from torch.autograd import Function


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        div_zero_guard = 0.0001

        input = torch.split(input, 1)[0].squeeze()

        # self.inter = torch.dot(input.view(-1).float(), target.view(-1).float())     # Basically TP
        self.inter = torch.sum(input.view(-1).float() * target.view(-1).float())    # Basically TP
        self.union = torch.sum(input) + torch.sum(target) + div_zero_guard          # TP + FP + FN

        FP = torch.sum(input) - self.inter                          # Take the guess and remove the TP
        FN = torch.sum(target) - self.inter                         # Take the truth and remove the TP
        TN = self.union - self.inter - FP - FN                      # Take
        # TN = (256 * 256) - self.inter - FP - FN                      # Take
        TTP = 2. * self.inter.float()                               # 2 TP for simplicity

        dice = (TTP + div_zero_guard) / (self.inter + self.union.float())
        sens = self.inter / (self.inter + FN + div_zero_guard)
        spec = TN / (TN + FP + div_zero_guard)
        prec = self.inter / (self.inter + FP + div_zero_guard)
        gmean = math.sqrt(abs(sens * spec))
        f2 = (5 * prec * sens) / (4 * prec + sens + div_zero_guard)

        data = [dice, sens, spec, prec, gmean, f2]

        return data

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    s = []
    if input.is_cuda:
        for i in range(6):
            s.append(torch.FloatTensor(1).cuda().zero_())
    else:
        for i in range(6):
            s.append(torch.FloatTensor(1).zero_())

    max_val = 0
    # Accounts for average across n_classes.
    for i, c in enumerate(zip(input, target)):
        data = DiceCoeff().forward(c[0], c[1])
        for j, m in enumerate(s):
            m += data[j]
        # s[0] = s[0] + data[0]
        max_val = i

    for k in range(len(s)):
        s[k] = s[k] / (max_val + 1)

    # return s / (i + 1)
    return s
