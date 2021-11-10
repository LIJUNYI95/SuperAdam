import math
import torch
from torch.optim.optimizer import Optimizer
import pdb
from collections import defaultdict
from torch import Tensor
from typing import List


class SuperAdam(Optimizer):
    r"""Implements SuperAdam algorithm.
    """

    def __init__(self, params, lr=1e-3, c = 5, m = 100, gamma = 0.01, beta1 = 0.99, beta= 0.999, eps=1e-3, amsgrad=False,\
                      global_size = False, coordinate_global_size=False, b_b = False, use_adam=False):
                
        defaults = dict(lr=lr, beta1 = beta1, beta=beta, eps=eps, c=c, m=m, gamma=gamma, amsgrad=amsgrad, \
            global_size = global_size, coordinate_global_size=coordinate_global_size, b_b = b_b, use_adam=use_adam)
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
            state_sums = []
            max_exp_avg_sqs = []
            old_grads = []
            old_steps = []
            state_steps = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('Super-Adam does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    # Lazy state initialization
                    if len(state) == 0:
                        state['step'] = 0
                        state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        
                        if group['global_size']:
                            state['grad_norm_sum'] = torch.zeros(1).cuda()
                        
                        if group['b_b']:
                            state['old_grad'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                            state['old_step'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                        
                        if group['amsgrad']:
                            state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    
                    if group['global_size']:
                        grad_norm_sum.append(state['grad_norm_sum'])
                    
                    if group['b_b']:                   
                        old_grads.append(state['old_grad'])
                        old_steps.append(state['old_step'])

                    if group['amsgrad']:
                        max_exp_avg_sqs.append(state['max_exp_avg_sq'])

                    # update the steps for each param group update
                    state['step'] += 1
                    # record the step after step update
                    state_steps.append(state['step'])

            h_max, h_min, g_norm = superadam(params_with_grad,
                   grads,
                   exp_avgs,
                   exp_avg_sqs,
                   max_exp_avg_sqs,
                   grad_norm_sum,
                   old_grads,
                   old_steps,
                   state_steps,
                   group['amsgrad'],
                   group['global_size'],
                   group['coordinate_global_size'],
                   group['b_b'],
                   group['use_adam'],
                   group['beta'],
                   group['beta1'],
                   group['lr'],
                   group['c'],
                   group['m'],
                   group['gamma'],
                   group['eps'])
        return h_max, h_min, g_norm
    
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
                    # Lazy state initialization
                    if group['b_b']:
                        state['old_grad'].add_(-state['old_grad'] + p.grad)
                    assert(len(state) != 0)

                    exp_avgs.append(state['exp_avg'])
                    
            eta = group['lr']/(group['m'] + state['step'])**(1/3)
            alpha = max(group['c'] * (eta)**2, 0.7) #superadam
            # alpha = group['c'] * (eta)**2
            # alpha = 0.7
            for i, _ in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg.add_(-grad).mul_(1 - alpha)
        # print('alpha',alpha)
        return loss


def superadam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         max_exp_avg_sqs: List[Tensor],
         grad_norm_sum: List[Tensor],
         old_grads: List[Tensor],
         old_steps: List[Tensor],
         state_steps: List[int],
         amsgrad: bool,
         global_size: bool,
         coordinate_global_size: bool,
         b_b: bool,
         use_adam: bool,
         beta: float,
         beta1: float,
         lr: float,
         c: float,
         m: float,
         gamma: float,
         eps: float):
    r"""Functional API that performs Adam algorithm computation.
    See :class:`~torch.optim.Adam` for details.
    
    """
    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_sq.mul_(beta).addcmul_(grad, grad, value=1 - beta)

    if coordinate_global_size:
        exp_avg_sq_norm  = torch.norm(torch.cat([g.sqrt().reshape(-1) for g in exp_avg_sqs]))

    H_max  = torch.max(torch.cat([g.sqrt().reshape(-1) for g in exp_avg_sqs])).data.cpu().numpy() + eps
    H_min  = torch.min(torch.cat([g.sqrt().reshape(-1) for g in exp_avg_sqs])).data.cpu().numpy() + eps
    g_norm  = torch.norm(torch.cat([g.reshape(-1) for g in grads])).data.cpu().numpy()


    if global_size:
        grad_norm = torch.norm(torch.cat([g.reshape(-1) for g in grads]))

    bb_denom = torch.ones(1).cuda()
    if b_b:    
        if state_steps[0] > 1:
            old_step_norm = torch.norm(torch.cat([g.reshape(-1) for g in old_steps]))
            diff_inner_product = torch.zeros(1).cuda()
            for i, param in enumerate(params):
                grad = grads[i]
                old_grad = old_grads[i]
                old_step = old_steps[i]
                
                diff_inner_product.add_(torch.matmul((old_grad- grad).reshape(-1), old_step.reshape(-1)))
            bb_denom = diff_inner_product.abs().div(old_step_norm**2).add_(eps).reshape(-1)

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]
        
        bias_correction2 = 1 - beta ** step
        
        if global_size:
            g_sum= grad_norm_sum[i]
        if b_b:
            old_grad = old_grads[i]; old_step = old_steps[i]
        if amsgrad:
            max_exp_avg_sq = max_exp_avg_sqs[i]

        
        if not use_adam:
            exp_avg.add_(grad)
        else:
            eta = lr/(m + step)**(1/2)
            alpha = c * eta #superadam
            bias_correction1 = 1 - alpha ** step
            # alpha = 0.1
            exp_avg.mul_(1 - alpha).add_(grad, alpha= alpha)
        ### Different ways of choosing the H matrix
        if amsgrad:
            torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
            denom = (max_exp_avg_sqs[i].sqrt() / math.sqrt(bias_correction2)).add_(eps)        
        elif global_size:
            g_sum.mul_(beta).add_(grad_norm**2, value=1 - beta)
            denom = g_sum.sqrt().add_(eps)   
        elif coordinate_global_size:
            denom = exp_avg_sq_norm.add_(eps)
        elif b_b:
            denom = torch.tensor(max(bb_denom.data.cpu().numpy().item(), 10), dtype=torch.float).cuda()
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        # eta = max(gamma * lr/(m + step)**(1/3), 0.0002) #superadam
        if not use_adam:
            eta = gamma * lr/(m + step)**(1/3)
        else:
            eta = gamma * lr/(m + step)** (1/2)
            # eta = 0.001
        # eta = 0.001

        param.addcdiv_(exp_avg, denom, value=-eta)
        if b_b:
            old_step.add_(-old_step).addcdiv_(exp_avg, denom, value=-eta/bias_correction1)
    
    # print(H_max, H_min, g_norm)
    return H_max, H_min, g_norm