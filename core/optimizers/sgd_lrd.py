import torch
from torch.optim.optimizer import Optimizer


class SGD_LRD(Optimizer):
    """Stochastic Gradient Descent with momentum and Learning Rate Dropout

    https://github.com/HuangxingLin123/Learning-Rate-Dropout/blob/master/cifar10/sgd_lrd.py
    """

    def __init__(
        self,
        params,
        lr: float,
        momentum=0,
        dampening=0,
        weight_decay=0,
        dropout=0.0,
        nesterov=False,
    ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if momentum < 0.0:
            raise ValueError("Invalid momentum value: {}".format(momentum))
        if weight_decay < 0.0:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))

        defaults = dict(
            lr=lr,
            momentum=momentum,
            dampening=dampening,
            weight_decay=weight_decay,
            nesterov=nesterov,
            dropout=dropout,
        )
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        super(SGD_LRD, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SGD_LRD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault("nesterov", False)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group["weight_decay"]
            momentum = group["momentum"]
            dampening = group["dampening"]
            nesterov = group["nesterov"]

            for p in group["params"]:
                if p.grad is None:
                    continue
                d_p = p.grad.data

                ## mask
                m = torch.ones_like(p.data) * group["dropout"]
                mask = torch.bernoulli(m)

                if weight_decay != 0:
                    # d_p.add_(weight_decay, p.data)
                    d_p.add_(p.data, alpha=weight_decay)
                if momentum != 0:
                    param_state = self.state[p]
                    if "momentum_buffer" not in param_state:
                        buf = param_state["momentum_buffer"] = torch.clone(d_p).detach()
                    else:
                        buf = param_state["momentum_buffer"]
                        # buf.mul_(momentum).add_(1 - dampening, d_p)
                        buf.mul_(momentum).add_(d_p, alpha=1 - dampening)

                ##dropout learning rate
                lr_dropout = group["lr"] * mask
                I_buf = lr_dropout * buf.clone()

                # p.data.add_(-1, I_buf)
                p.data.add_(I_buf, alpha=-1)

                if nesterov:
                    d_p = d_p.add(momentum, buf)

        return loss
