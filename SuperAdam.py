import torch
from torch.optim.optimizer import Optimizer
from torch import Tensor

import math
from collections import defaultdict
from typing import List


class SuperAdam(Optimizer):
    r"""Implements SuperAdam Optimizer.
        @article{huang2021super,
            title={SUPER-ADAM: Faster and Universal Framework of Adaptive Gradients},
            author={Huang, Feihu and Li, Junyi and Huang, Heng},
            journal={arXiv preprint arXiv:2106.08208},
            year={2021}}
    """

    def __init__(self, params, tau=False, gamma = 0.01, k=1e-3, m = 100, c = 5,\
         beta= 0.999, eps=1e-3, glob_H = False, coord_glob_H=False):
        """
        General arguments:
            @param params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
            @param tau (boolean):  if set True: use storm-like variance reduction update;
                                        otherwise: use adam-like momentum update (refer to line 11 in Algorithm 1)
            @param gamma (float) : scaling coefficient of the quadratic optimization problem (please refer to line 9 in Algorithm 1)
            @param k (float): scaling coefficient of mu
            @param m (float): offset coefficient of mu
            @param c (float): coefficient of the momentum: alpha = c * mu^2

        Arguments related to the Adaptive matrix H:
            @param beta (float): exponential average coefficient
            @param eps (float) : bias term added to H matrix to adjust the eigenvalues of H (smaller eps means the optimizer is more adaptive)
            By default: SuperAdam maintains exponential average of the gradient square: 
                    v_t = beta * v_(t-1) + (1 - beta) * g_(t-1)^2 and adopts H matrix as: H_t = Diag(sqrt(vt) + eps) 
            We also implement the following altenatives:
                @param glob_H (boolean): If set True: maintain exponential average of gradient norms: 
                    v_t = beta * v_(t-1) + (1 - beta) * ||g_(t-1)|| and adopt H matrix as: H_t = (vt + eps) * I 
                @param coord_glob_H (boolean): If set True: maintain exponential average of gradient squares: 
                    v_t = beta * v_(t-1) + (1 - beta) * g_(t-1)^2 and adopt H matrix as: H_t = (sqrt(||vt||) + eps) * I 
        """

        defaults = dict(tau = tau, k=k, beta=beta, eps=eps, c=c, m=m, gamma=gamma,\
            glob_H = glob_H, coord_glob_H=coord_glob_H)
        super(SuperAdam, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SuperAdam, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self):
        """Performs a single optimization step.
        """
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
                        
                        if group['glob_H']:
                            state['grad_norm_sum'] = torch.zeros(1).cuda()

                    exp_avgs.append(state['exp_avg'])
                    exp_avg_sqs.append(state['exp_avg_sq'])
                    
                    if group['glob_H']:
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
                   group['glob_H'],
                   group['coord_glob_H'],
                   group['tau'],
                   group['beta'],
                   group['k'],
                   group['c'],
                   group['m'],
                   group['gamma'],
                   group['eps'])
        return
    
    @torch.no_grad()
    def update_momentum(self, closure=None):
        """Update momentum term (Only for tau is set True, which requires to evaluate at two different points)
        """

        for group in self.param_groups:
            params_with_grad = []
            grads = []
            exp_avgs = []

            for p in group['params']:
                if p.grad is not None:
                    params_with_grad.append(p)
                    if p.grad.is_sparse:
                        raise RuntimeError('SuperAdam does not support sparse gradients')
                    grads.append(p.grad)

                    state = self.state[p]
                    assert(len(state) != 0)

                    exp_avgs.append(state['exp_avg'])
                    
            eta = group['k']/(group['m'] + state['step'])**(1/3)
            alpha = min(group['c'] * (eta)**2, 0.99)
            for i, _ in enumerate(params_with_grad):
                grad = grads[i]
                exp_avg = exp_avgs[i]
                exp_avg.add_(-grad).mul_(1 - alpha)
        
        return


def superadam(params: List[Tensor],
         grads: List[Tensor],
         exp_avgs: List[Tensor],
         exp_avg_sqs: List[Tensor],
         grad_norm_sum: List[Tensor],
         state_steps: List[int],
         glob_H: bool,
         coord_glob_H: bool,
         tau: bool,
         beta: float,
         k: float,
         c: float,
         m: float,
         gamma: float,
         eps: float):

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg_sq = exp_avg_sqs[i]
        exp_avg_sq.mul_(beta).addcmul_(grad, grad, value=1 - beta)

    if coord_glob_H:
        exp_avg_sq_norm  = torch.norm(torch.cat([g.sqrt().reshape(-1) for g in exp_avg_sqs]))

    if glob_H:
        grad_norm = torch.norm(torch.cat([g.reshape(-1) for g in params]))

    for i, param in enumerate(params):
        grad = grads[i]
        exp_avg = exp_avgs[i]
        exp_avg_sq = exp_avg_sqs[i]
        step = state_steps[i]

        if not tau:
            exp_avg.add_(grad)
        else:
            eta = k/(m + step)**(1/2)
            alpha = c * eta
            exp_avg.mul_(1 - alpha).add_(grad, alpha= alpha)

        ### Different altenatives of H matrix
        bias_correction2 = 1 - beta ** step     
        if glob_H:
            g_sum= grad_norm_sum[i]
            g_sum.mul_(beta).add_(grad_norm**2, value=1 - beta)
            denom = g_sum.sqrt().add_(eps)   
        elif coord_glob_H:
            denom = exp_avg_sq_norm.add_(eps)
        else:
            denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

        if not tau:
            eta = gamma * k/(m + step)**(1/3)
        else:
            eta = gamma * k/(m + step)** (1/2)

        param.addcdiv_(exp_avg, denom, value=-eta)
    
    return