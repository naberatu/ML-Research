
# Environment imports
import torch
import warnings
import torch_pruning as tp

warnings.filterwarnings("ignore")


def prune_model(name='', model=None, dir_models='', suffix=''):
    print('\nPruning Model: ' + name + '...', end='\t')

    # 1. setup strategy (L1 Norm)           <-- can also try tp.strategy.RandomStrategy()
    strategy = tp.strategy.L1Strategy()

    # 2. build layer dependency for resnet18
    DG = tp.DependencyGraph()
    DG.build_dependency(model, example_inputs=torch.randn(1, 3, 224, 224))

    # 3. get a pruning plan from the dependency graph.
    pruning_idxs = strategy(model.conv1.weight, amount=0.4)     # or manually selected pruning_idxs=[2, 6, 9, ...]
    pruning_plan = DG.get_pruning_plan(model.conv1, tp.prune_conv, idxs=pruning_idxs)
    # print(pruning_plan)

    # 4. execute this plan (prune the model)
    pruning_plan.exec()
    print('COMPLETE')

    # 5. Save Model
    print('Saving', name + suffix + '...', end='\t')
    filename = dir_models + name + suffix + ".pth"
    torch.save(model, filename)
    print('COMPLETE')
    return torch.load(filename)
