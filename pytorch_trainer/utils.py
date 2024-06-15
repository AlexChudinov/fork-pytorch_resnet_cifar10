import torch


def save_checkpoint(state, filename='checkpoint.pth.tar'):
    """
    Save the training model
    """
    torch.save(state, filename)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.__val = None
        self.__sum = None
        self.__count = None
        self.reset()

    def reset(self):
        self.__sum = 0
        self.__count = 0

    def update(self, val, n=1):
        self.__val = val
        self.__sum += val * n
        self.__count += n

    @property
    def avg(self):
        return self.__sum / self.__count

    @property
    def val(self):
        return self.__val


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def _checkpoint_filename(model_name: str) -> str:
    return model_name + "-checkpoint.th"


def _history_filename(model_name: str) -> str:
    return model_name + "-history.data"
