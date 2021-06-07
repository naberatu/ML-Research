import torch
import torch.nn.functional as F
from tqdm import tqdm

from un_diceloss import dice_coeff


def fit_tensor(prediction, truth, num_classes):
    # Enforces non-negative tensors.
    prediction[prediction < 0] = 0
    truth[truth < 0] = 0

    # Enforces tensors within range of class totals.
    prediction[prediction >= num_classes] = num_classes - 1
    truth[truth >= num_classes] = num_classes - 1


def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.n_classes == 1 else torch.long
    n_val = len(loader)  # the number of batch
    # tot = 0
    metrics = [0]

    with tqdm(total=n_val, desc='Validation', unit='batch', leave=False) as pbar:
        for batch in loader:
            imgs, true_masks = batch['image'], batch['mask']
            imgs = imgs.to(device=device, dtype=torch.float32)
            true_masks = true_masks.to(device=device, dtype=mask_type)

            # Enforces 0 <= t < OUT_CH
            fit_tensor(imgs, true_masks, net.n_classes)

            with torch.no_grad():
                mask_pred = net(imgs)

            if net.n_classes > 1:
                # tot += F.cross_entropy(mask_pred, true_masks).item()
                metrics[0] += F.cross_entropy(mask_pred, true_masks).item()
                pred = torch.sigmoid(mask_pred)
                pred = (pred > 0.5).float()
                temp = dice_coeff(pred, true_masks)

                for i, t in enumerate(temp, 0):
                    if len(metrics) < 7:
                        metrics.append(temp[i].item())
                    elif i != 0:
                        metrics[i] += temp[i].item()

            pbar.update()

    net.train()
    for i in range(len(metrics)):
        metrics[i] = metrics[i] / n_val

    return metrics
    # return tot / n_val
