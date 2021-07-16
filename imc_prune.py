
# Environment imports
import torch
import warnings
import torch_pruning as tp
from torch.nn import modules
from torchvision.models.resnet import BasicBlock

warnings.filterwarnings("ignore")


def prune_model(name='', model=None, dir_models='', suffix='', im_size=224):
    print('\nPruning Model: ' + name + '...', end='\t')

    strategy = tp.strategy.L1Strategy()
    DG = tp.DependencyGraph().build_dependency(model, example_inputs=torch.randn(1, 3, im_size, im_size))

    def prune_conv(conv, amount=0.2):
        pruning_index = strategy(conv.weight, amount=amount)
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        plan.exec()

    block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
    limit = len(block_prune_probs) - 1
    blk_id = 0
    conv_index = 0
    prev_conv = False
    for m in list(model.modules())[1:]:
        if isinstance(m, modules.Conv2d):
            prev_conv = True
            conv_index += 1
        if isinstance(m, modules.Linear) and prev_conv:
            break

    for m in list(model.modules())[1:]:
        if isinstance(m, modules.Conv2d) and (conv_index > 1 or 'naber' not in name.lower()):
            prune_conv(m)
            conv_index -= 1
        elif isinstance(m, BasicBlock):
            prune_conv(m.conv1, block_prune_probs[blk_id])
            prune_conv(m.conv2, block_prune_probs[blk_id])
            if blk_id < limit - 1:
                blk_id += 1

    print('COMPLETE')

    # 5. Save Model
    print('Saving', name + suffix + '...', end='\t')
    filename = dir_models + name + suffix + ".pth"
    torch.save(model, filename)
    print('COMPLETE')
    return torch.load(filename)
