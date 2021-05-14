import time
import torch
import torch.nn as nn
import pathlib
import logging
# Fit Routine Provided by Mohammed Alnemari and modified by Nader Atout


def fit(model, train_data_loader, test_data_loader, optimizer, epochs=10, criterion=torch.nn.CrossEntropyLoss(),
        print_freq=10, save_model=True, save_params=True, best_acc=0, model_name="None"):

    for epoch in range(epochs):
        print("\n==========================================")
        print("> Starting Epoch", epoch + 1)
        print("==========================================")
        adjust_learning_rate(optimizer, epochs)
        train_accuracy1, train_accuracy5 = train(model, train_data_loader, optimizer, epoch, model_name, criterion, print_freq=print_freq)
        test_accuracy1, test_accuracy5 = test(model, test_data_loader, model_name, epoch=epoch, print_freq=print_freq)

        if test_accuracy1 > best_acc:
            if save_model:
                print("=== The model is saved with accuracy: {}".format(test_accuracy1))
                model_save(model, test_accuracy1, model_name)
                best_acc = test_accuracy1
            if save_params:
                print("=== The model parameters is saved with accuracy: {}".format(test_accuracy1))
                params_save(model, epoch, optimizer, train_accuracy1, train_accuracy5,
                            test_accuracy1, test_accuracy5, model_name=model_name)
                best_acc = test_accuracy1

    return train_accuracy1, test_accuracy1
    # return train_accuracy1, test_accuracy1, train_accuracy5, test_accuracy5


def train(model, train_data_loader, optimizer, epoch, model_name, criterion=nn.CrossEntropyLoss(), print_freq=10):
    path = pathlib.Path('./logs/train_logger/')
    try:
        path.mkdir(parents=True, exist_ok=True)

    except OSError:
        print("\n> ERROR: Failed to make nested directory")
    else:
        print("\n> Train Logger successfully created.")
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
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(train_data_loader), batch_time, data_time, losses, top1,
                             prefix="Epoch: [{}]".format(epoch))

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)
    model.train()

    end = time.time()
    for i, x in enumerate(train_data_loader):
        data_time.update(time.time() - end)

        # batch, label = x['img'], x['label']
        batch, label = x
        batch = batch.to(device)
        label = label.to(device)
        label = label.long()

        label = torch.argmax(label, dim=1)

        output = model(batch)
        loss = criterion(output, label)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, label, topk=(1, 5))
        losses.update(loss.item(), batch.size(0))
        top1.update(acc1[0], batch.size(0))
        top5.update(acc5[0], batch.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)

        if i % print_freq == 0:

            progress.print(i)
            msg = ('Epoch: [{0}][{1}/{2}]\t'
                   'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                   'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                   'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                   'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                   'Prec@5 {top1.val:.3f} ({top1.avg:.3f})\t'
                   )

            logger.info(msg.format(epoch, i, len(train_data_loader),
                                        batch_time=batch_time,
                                        data_time=data_time,
                                        loss=losses,
                                        top1=top1, top5=top5))
                                        # top1=top1))

    return top1.avg, top5.avg
    # return top1.avg


def test(model, test_data_loader, model_name, epoch, criterion=torch.nn.CrossEntropyLoss(), print_freq=10, valsplit=False):
    path = pathlib.Path('./logs/test_logger')
    try:
        path.mkdir(parents=True, exist_ok=True)

    except OSError:
        print("\n> ERROR: Failed to make nested directory")
    else:
        print("\n> Test Logger successfully created.")

    file = str(path) + "/" +"__"+model_name+"__run__"+"_test.log"

    logger = logging.getLogger(name='test')
    logger.setLevel(logging.INFO)
    fmt = logging.Formatter('%(message)s')
    file_handler = logging.FileHandler(file)
    file_handler.setFormatter(fmt)
    logger.addHandler(file_handler)

    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(len(test_data_loader), batch_time, losses, top1, prefix='Test: ')

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    model.to(device)
    model.eval()
    with torch.no_grad():
        end = time.time()

    for i, x in enumerate(test_data_loader):

        # batch, label = x['img'], x['label']
        batch, label = x
        batch = batch.to(device)
        label = label.to(device)
        label = label.long()

        label = torch.argmax(label, dim=1)

        output = model(batch)
        loss = criterion(output, label)

        # measure accuracy and record loss
        acc1, acc5 = accuracy(output, label, topk=(1, 5))
        losses.update(loss.item(), batch.size(0))
        top1.update(acc1[0], batch.size(0))
        top5.update(acc5[0], batch.size(0))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end_time = time.time()

        if i % print_freq == 0:
            progress.print(i)

        msg = ('Epoch: [{0}][{1}/{2}]\t'
               'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
               'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
               'Prec@5 {top1.val:.3f} ({top1.avg:.3f})\t'
               )
        logger.info(msg.format(epoch, i, len(test_data_loader), loss=losses, batch_time=batch_time, top1=top1, top5=top5))
        # logger.info(msg.format(epoch, i, len(test_data_loader), loss=losses, batch_time=batch_time, top1=top1))

    return top1.avg, top5.avg
    # return top1.avg


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
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        print(pred.shape)
        # exit()
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))

        print(res)
        return res


def adjust_learning_rate(optimizer, epoch, lr=0.001):
    # todo 5 update_list
    lr = lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * 0.1
    return None


# todo use os package to cheak folder and create one if it is not exist
def model_save(model, model_name):
    path_whole = str(pathlib.Path.cwd()) + '/' + 'results'
    path_whole = pathlib.Path(path_whole)
    path_whole.mkdir(parents=True, exist_ok=True)
    file_whole = str(path_whole) + '/' + model_name + '__whole__' + '.pth'

    torch.save(model, file_whole)


# def params_save(model, epoch, optimizer, train_accuracy_1, test_accuracy_1, model_name):
def params_save(model, epoch, optimizer, train_accuracy_1, train_accuracy_5, test_accuracy_1, test_accuracy_5, model_name):
    path_params = str(pathlib.Path.cwd()) + '/' + 'results'
    path_params = pathlib.Path(path_params)
    path_params.mkdir(parents=True, exist_ok=True)
    file_params = str(path_params) + "/" + model_name + '__params__' + '.pth.tar'

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.module.state_dict(),
        'optimizer_stat_dict': optimizer.state_dict(),
        'top1_accuracy_train': train_accuracy_1,
        'top5_accuracy_train': train_accuracy_5,
        'top1_accuracy_test': test_accuracy_1,
        'top5_accuracy_test': test_accuracy_5,

    }, file_params)
