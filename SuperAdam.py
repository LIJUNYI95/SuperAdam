import math
import torch
from torch.optim.optimizer import Optimizer
from collections import defaultdict
from torch import Tensor
from typing import List


class SuperAdam(Optimizer):
    r"""Implements SuperAdam algorithm.
    """

    def __init__(self, params, use_adam=False, lr=1e-3, m = 100, gamma = 0.01, c = 5, beta= 0.999, \
        global_size = False, coordinate_global_size=False, eps=1e-3):
                
        defaults = dict(lr=lr, beta=beta, eps=eps, c=c, m=m, gamma=gamma,\
            global_size = global_size, coordinate_global_size=coordinate_global_size, use_adam=use_adam)
        super(SuperAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SuperAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        """Performs a single optimization step.
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []
            exp_avg_sqs = []

            grad_norm_sum = []

            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Super-Adam does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        
                        if group['global_size']:
                            state['grad_norm_sum'] = torch.zeros(1).cuda()

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    
                    if group['global_size']:
                        grad_norm_sum.append(state['grad_norm_sum'])


                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            superadam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   grad_norm_sum,
                   state_steps,
                   group['global_size'],
                   group['coordinate_global_size'],
                   group['use_adam'],
                   group['beta'],
                   group['lr'],
                   group['c'],
                   group['m'],
                   group['gamma'],
                   group['eps'])
        return
    
    @torch.no_grad()
    def update_momentum(self, closure=None):
        """Update momentum term (Only for Storm, which requires 
        to evaluate at two different points)
        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Adam does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    assert(len(state) != 0)

                    exp_avgs.append(state['exp_avg'])
                    
            eta = group['lr']/(group['m'] + state['step'])**(1/3)
            alpha = min(group['c'] * (eta)**2, 0.99) #superadam
            for i, _ in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg.add_(-grad).mul_(1 - alpha)
        return loss


def superadam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         grad_norm_sum: List[Tensor],
         state_steps: List[int],
         global_size: bool,
         coordinate_global_size: bool,
         use_adam: bool,
         beta: float,
         lr: float,
         c: float,
         m: float,
         gamma: float,
         eps: float):

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_sq.mul_(beta).addcmul_(grad, grad, value=1 - beta)

    if coordinate_global_size:
        exp_avg_sq_norm  = torch.norm(torch.cat([g.sqrt().reshape(-1) for g in exp_avg_sqs]))

    if global_size:
        grad_norm = torch.norm(torch.cat([g.reshape(-1) for g in params]))

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        if not use_adam:
            exp_avg.add_(grad)
        else:
            eta = lr/(m + step)**(1/2)
            alpha = c * eta
            exp_avg.mul_(1 - alpha).add_(grad, alpha= alpha)

        ### Different ways of choosing the H matrix
        bias_correction2 = 1 - beta ** step     
        if global_size:
            g_sum= grad_norm_sum[i]
            g_sum.mul_(beta).add_(grad_norm**2, value=1 - beta)
            denom = g_sum.sqrt().add_(eps)   
        elif coordinate_global_size:
            denom = exp_avg_sq_norm.add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        if not use_adam:
            eta = gamma * lr/(m + step)**(1/3)
        else:
            eta = gamma * lr/(m + step)** (1/2)

        param.addcdiv_(exp_avg, denom, value=-eta)
    
    return