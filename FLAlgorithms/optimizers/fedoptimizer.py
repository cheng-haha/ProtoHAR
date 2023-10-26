from torch.optim import Optimizer
import torch
from torch.optim import SGD

class MySGD(Optimizer):
    def __init__(self, params, lr):
        defaults = dict(lr=lr)
        super(MySGD, self).__init__(params, defaults)

    def step(self, closure=None, lamda = 0):
        loss = None
        if closure is not None:
            loss = closure

        for group in self.param_groups:
            # print(group)
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                if(lamda != 0):
                    p.data.add_(-lamda, d_p)
                else:     
                    p.data.add_(-group['lr'], d_p)
        return loss
        
class pFedIBOptimizer(Optimizer):
    def __init__(self, params, lr=0.01):
        # self.local_weight_updated = local_weight # w_i,K
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults=dict(lr=lr)
        super(pFedIBOptimizer, self).__init__(params, defaults)

    def step(self, k, apply=True, lr=None, allow_unused=False):
        grads = []
        # apply gradient to model.parameters, and return the gradients
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None and allow_unused:
                    continue
                # print( p.grad.data )
                grads.append(p.grad.data)
                if apply:
                    if lr == None:
                        p.data= p.data - ( group['lr'] / k ) * p.grad.data
                    else:
                        p.data=p.data - ( lr / k ) * p.grad.data
        return group['params'] , grads

class FedProxOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, lamda=0.1 ):
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults=dict(lr=lr, lamda=lamda )
        super(FedProxOptimizer, self).__init__(params, defaults)
        
    @torch.no_grad()
    def step(self, global_params, closure=None):
        loss=None
        if closure is not None:
            loss=closure
        for group in self.param_groups:
            for p, g in zip(group['params'], global_params):
                # w <=== w - lr * ( w'  + lambda * (w - w* ) )
                p.data = p.data - group['lr'] * (
                            p.grad.data + group['lamda'] * (p.data - g.data.clone()) )
        return group['params'], loss


class FEDLOptimizer(Optimizer):
    def __init__(self, params, lr=0.01, server_grads=None, pre_grads=None, eta=0.1):
        self.server_grads = server_grads
        self.pre_grads = pre_grads
        if lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr, eta=eta)
        super(FEDLOptimizer, self).__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure
        for group in self.param_groups:
            i = 0
            for p in group['params']:
                p.data = p.data - group['lr'] * \
                         (p.grad.data + group['eta'] * self.server_grads[i] - self.pre_grads[i])
                # p.data.add_(-group['lr'], p.grad.data)
                i += 1
        return loss






class ScaffoldOptimizer(Optimizer):
    def __init__(self, params, lr, weight_decay):
        defaults = dict(
            lr=lr, weight_decay=weight_decay
        )
        super().__init__(params, defaults)

    def step(self, server_control, client_control, closure=None):
        loss = None
        if closure is not None:
            loss = closure

        ng = len(self.param_groups[0]["params"])
        names = list(server_control.keys())

        # BatchNorm: running_mean/std, num_batches_tracked
        names = [name for name in names if "running" not in name]
        names = [name for name in names if "num_batch" not in name]

        t = 0
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue

                c = server_control[names[t]]
                ci = client_control[names[t]]

                # print(names[t], p.shape, c.shape, ci.shape)
                d_p = p.grad.data + c.data - ci.data
                p.data = p.data - d_p.data * group["lr"]
                t += 1
        assert t == ng
        return loss


