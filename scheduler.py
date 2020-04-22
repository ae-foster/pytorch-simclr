import math

from torch.optim.lr_scheduler import _LRScheduler


class CosineAnnealingWithLinearRampLR(_LRScheduler):

    def __init__(self, optimizer, T_max, eta_min=0, last_epoch=-1, ramp_len=10):
        self.T_max = T_max
        self.eta_min = eta_min
        self.ramp_len = ramp_len
        super(CosineAnnealingWithLinearRampLR, self).__init__(optimizer, last_epoch)

    def get_lr(self):
        return self._get_closed_form_lr()

    def _get_closed_form_lr(self):
        cosine_lr = [self.eta_min + (base_lr - self.eta_min) *
                     (1 + math.cos(math.pi * self.last_epoch / self.T_max)) / 2
                     for base_lr in self.base_lrs]
        linear_lr = [base_lr * (1 + self.last_epoch) / self.ramp_len for base_lr in self.base_lrs]
        return [min(x, y) for x, y in zip(cosine_lr, linear_lr)]
