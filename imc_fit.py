"""
This file was originally authored and provided by Mohammed Alnemari.
Modifications and adaptation by Nader Atout.
"""
import time
import torch
import torch.nn as nn
import pathlib
import logging

# Setup device
device = torch.device('cuda')


def fit(model, train_loader, test_loader, optimizer, epochs=10, criterion=torch.nn.CrossEntropyLoss(), best_acc=0.0,
        print_freq=10, save_model=True, save_params=True, quant=False, sub_folder='', model_name='Empty', divider=''):

    global device
    device = torch.device('cpu') if quant else torch.device('cuda')
    for epoch in range(epochs):
        adjust_learning_rate(optimizer, epochs)
        acc_train = train(model=model, model_name=model_name, train_loader=train_loader, optimizer=optimizer,
                          epochs=epoch, criterion=criterion, print_freq=print_freq, quant=quant, divider=divider)
        acc_test = test(model=model, model_name=model_name, test_data_loader=test_loader,
                        epoch=epoch, print_freq=print_freq, divider=divider, re_test=False, quant=quant)

        if acc_test > best_acc:
            if save_model:
                print("=== Model Saved with Accuracy: \t\t\t{:.1f}%".format(acc_test))
                model_save(model=model, model_name=model_name, sub_folder=sub_folder)
                best_acc = acc_test
            if save_params:
                print("=== Parameters Saved with Accuracy: \t{:.1f}%".format(acc_test))
                params_save(model=model, epoch=epoch, optimizer=optimizer, train_accuracy_1=acc_train,
                            test_accuracy_1=acc_test, model_name=model_name, sub_folder=sub_folder)
                best_acc = acc_test

    return acc_train, acc_test


def train(model=None, train_loader=None, optimizer=None, epochs=1, model_name='',
          criterion=nn.CrossEntropyLoss(), quant=False, print_freq=10, divider=''):

    path = pathlib.Path('logs/train_logger/')
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        print("\n> ERROR: Failed to make nested directory")

    global device
    device = torch.device('cpu') if quant else torch.device('cuda')

    # Print Training Epoch
    print()
    print(divider)
    print("> Training Epoch", epochs + 1)
    print(divider)

    # Generate logs
    file = str(path) + "/" +"__"+model_name+"__run__"+"_training.log"

    logger = logging.getLogger(name='train')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(train_loader), batch_time, data_time, losses, top1, prefix="Epoch: [{}]".format(epochs))

    # Execute training
    model.to(device)
    model.train()

    end = time.time()
    for i, x in enumerate(train_loader):
        data_time.update(time.time() - end)

        batch, label = x['img'], x['label']
        batch = batch.to(device)
        label = label.to(device)

        output = model(batch)
        loss = criterion(output, label)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, label, topk=(1, 5))
        losses.update(loss.item(), batch.size(0))
        top1.update(acc1[0], batch.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        if i % print_freq == 0:
            progress.print(i)
            msg = (
                'Epoch: [{0}][{1}/{2}]\t'
                'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
            )
            logger.info(msg.format(
                epochs, i, len(train_loader),
                batch_time=batch_time,
                data_time=data_time,
                loss=losses,
                top1=top1)
            )

    return top1.avg


def test(model=None, test_data_loader=None, model_name='', epoch=0, criterion=torch.nn.CrossEntropyLoss(),
         print_freq=10, divider='', re_test=False, quant=False):

    path = pathlib.Path('logs/test_logger/')
    try:
        path.mkdir(parents=True, exist_ok=True)
    except OSError:
        print("\n> ERROR: Failed to make nested directory")

    global device
    device = torch.device('cpu') if quant else torch.device('cuda')

    # Print Testing Round
    print()
    print("> Validation for Epoch ", epoch + 1)
    print(divider)

    file = str(path) + "/__" + model_name + "__run___test.log"
    if re_test:
        file = str(path) + "/__" + model_name + "_eval.log"

    logger = logging.getLogger(name='test')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    progress = ProgressMeter(len(test_data_loader), batch_time, losses, top1, prefix='Test: ')

    model.to(device)
    model.eval()
    with torch.no_grad():
        end = time.time()

    for i, x in enumerate(test_data_loader):

        batch, label = x['img'], x['label']
        batch = batch.to(device)
        label = label.to(device)

        output = model(batch)
        loss = criterion(output, label)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, label, topk=(1, 5))
        losses.update(loss.item(), batch.size(0))
        top1.update(acc1[0], batch.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end_time = time.time()

        if i % print_freq == 0:
            progress.print(i)

        msg = (
            'Epoch: [{0}][{1}/{2}]\t'
            'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
            'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
        )
        logger.info(msg.format(epoch, i, len(test_data_loader), loss=losses, batch_time=batch_time, top1=top1))

    return top1.avg


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, *meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def print(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        maxk = 2
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        return res


def adjust_learning_rate(optimizer, epoch, lr=0.001):
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
    return None


def model_save(model=None, model_name='', sub_folder=''):
    path_whole = str(pathlib.Path.cwd()) + sub_folder
    path_whole = pathlib.Path(path_whole)
    path_whole.mkdir(parents=True, exist_ok=True)
    file_whole = str(path_whole) + '\\' + model_name + '.pth'

    torch.save(model, file_whole)


def params_save(model=None, epoch=1, optimizer=None, train_accuracy_1=0.0, test_accuracy_1=0.0, model_name='', sub_folder=''):
    path_params = str(pathlib.Path.cwd()) + sub_folder
    path_params = pathlib.Path(path_params)
    path_params.mkdir(parents=True, exist_ok=True)
    filename = str(path_params) + '\\' + model_name + '_params' + '.pth.tar'

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_stat_dict': optimizer.state_dict(),
        'top1_accuracy_train': train_accuracy_1,
        'top1_accuracy_test': test_accuracy_1,
    }, filename)
