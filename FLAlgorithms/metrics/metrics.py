import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.loss import _Loss
import numpy as np
from torch.distributions import MultivariateNormal as MVN
softplus = nn.Softplus()
cosine_similarity = nn.CosineSimilarity(-1)
cross_entropy = nn.CrossEntropyLoss()
EPSILON = 1e-8


def gai_loss_md(pred, target, gmm, noise_var):
    I = torch.eye(pred.shape[-1])
    mse_term = -MVN(pred, noise_var*I).log_prob(target)
    balancing_term = MVN(gmm['means'], gmm['variances']+noise_var*I).log_prob(pred.unsqueeze(1)) + gmm['weights'].log()
    balancing_term = torch.logsumexp(balancing_term, dim=1)
    loss = mse_term + balancing_term
    loss = loss * (2 * noise_var).detach()
    return loss.mean()



def guassian_kernel(
        source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    n_samples = int(source.size()[0]) + int(target.size()[0])
    total = torch.cat([source, target], dim=0)

    L2_distance = ((
        total.unsqueeze(dim=1) - total.unsqueeze(dim=0)
    ) ** 2).sum(2)

    if fix_sigma:
        bandwidth = fix_sigma
    else:
        bandwidth = torch.sum(L2_distance.data) / (n_samples ** 2 - n_samples)
        bandwidth += 1e-8

    # print("Bandwidth:", bandwidth)
    bandwidth /= kernel_mul ** (kernel_num // 2)
    bandwidth_list = [bandwidth * (kernel_mul**i) for i in range(kernel_num)]
    kernel_val = [
        torch.exp(-L2_distance / band) for band in bandwidth_list
    ]
    return sum(kernel_val)


def mmd_rbf_noaccelerate(
        source, target, kernel_mul=2.0, kernel_num=5, fix_sigma=None):
    batch_size = int(source.size()[0])
    kernels = guassian_kernel(
        source, target,
        kernel_mul=kernel_mul, kernel_num=kernel_num,
        fix_sigma=fix_sigma
    )
    XX = kernels[:batch_size, :batch_size]
    YY = kernels[batch_size:, batch_size:]
    XY = kernels[:batch_size, batch_size:]
    YX = kernels[batch_size:, :batch_size]
    loss = torch.mean(XX + YY - XY - YX)
    return loss

def MMD(x, y, kernel, device='cpu'):
    """Emprical maximum mean discrepancy. The lower the result
       the more evidence that distributions are the same.

    Args:
        x: first sample, distribution P
        y: second sample, distribution Q
        kernel: kernel type such as "multiscale" or "rbf"
    """
    xx, yy, zz = torch.mm(x, x.t()), torch.mm(y, y.t()), torch.mm(x, y.t())
    rx = (xx.diag().unsqueeze(0).expand_as(xx))
    ry = (yy.diag().unsqueeze(0).expand_as(yy))
    
    dxx = rx.t() + rx - 2. * xx # Used for A in (1)
    dyy = ry.t() + ry - 2. * yy # Used for B in (1)
    dxy = rx.t() + ry - 2. * zz # Used for C in (1)
    
    XX, YY, XY = (torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device),
                  torch.zeros(xx.shape).to(device))
    
    if kernel == "multiscale":
        
        bandwidth_range = [0.2, 0.5, 0.9, 1.3]
        for a in bandwidth_range:
            XX += a**2 * (a**2 + dxx)**-1
            YY += a**2 * (a**2 + dyy)**-1
            XY += a**2 * (a**2 + dxy)**-1
            
    if kernel == "rbf":
      
        bandwidth_range = [10, 15, 20, 50]
        for a in bandwidth_range:
            XX += torch.exp(-0.5*dxx/a)
            YY += torch.exp(-0.5*dyy/a)
            XY += torch.exp(-0.5*dxy/a)
      
    return torch.mean(XX + YY - 2. * XY)


def CosineSim(param1, param2):
    #计算余弦相似度作为两个客户端之间的模型度量
    CosSim_score = torch.sum(param1*param2)/(torch.norm(param1)*torch.norm(param2)+1e-12)
    return CosSim_score
def CosineDistance(param1, param2):
    #计算余弦相似度作为两个客户端之间的模型度量
    CosSim_score = torch.sum(param1*param2)/(torch.norm(param1)*torch.norm(param2)+1e-12)
    return 1-CosSim_score



def flatten(source):
    return torch.cat([value.flatten() for value in source.parameters()])
def cosine_similarities(self,model_1, model_2):

    #计算余弦相似度作为两个客户端之间的模型度量
    model1 = self.flatten(model_1)
    model2 = self.flatten(model_2)
    cossin_score = torch.sum(model1*model2)/(torch.norm(model1)*torch.norm(model2)+1e-12)
    return cossin_score
