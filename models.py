"""
code modified from https://github.com/Physics-aware-AI/DiffCoSim/
"""

import torch.nn as nn
import torch

class VariationalODE(nn.Module):

    def __init__(self,input_dim,  output_dim, ode_func, num_samples = 1, **kwargs):
        super().__init__()
        self.num_samples = num_samples
        self.ode = ode_func
        self.kwargs = kwargs

    def forward_ode(self, X0, ts, tol=1e-4, method="rk4", treat=False, cf3=False):
        preds = self.ode.integrate(X0, ts, tol, method, treat, cf3=cf3)
        return preds

    def forward_diff(self, Xts, is_clds, X0_, ts, cf3, treat):
        X_hat_ = self.ode.integrate_diff(Xts, is_clds, X0_, ts, cf3=cf3, treat=treat)
        return X_hat_

    def forward_ode_onestep(self, Xts, ts, tol, method):
        X_hat = self.ode.integrate_onestep(Xts, ts, tol, method)
        return X_hat
        
    def forward(self, X0, Xts, is_clds, ts, X0_, ts_, tol=1e-4, method="rk4", cf3=False, treat=False):
        # X0_: [bz, 2, obj, 2]
        X_hat = self.forward_ode_onestep(Xts, ts, tol, method)
        X_prime_hat = self.forward_ode(X0_, ts_, tol, method, cf3=cf3)
        if self.kwargs['body_kwargs_file'] == '_CP3_cf_3':
            X_d_prime_hat = 0
        else:
            X_d_prime_hat = self.forward_diff(Xts, is_clds, X0_, ts, cf3, treat)
        return X_hat, X_prime_hat, X_d_prime_hat

    def compute_losses(self, X, X_hat, X_prime, X_prime_hat, X_prime_hat_diff):
        rec_X_loss = (X_hat - X).abs()
        rec_X_prime_loss = (X_prime_hat - X_prime).abs()
        if self.kwargs['body_kwargs_file'] == '_CP3_cf_3':
            rec_X_prime_loss_ = torch.zeros(1)
        else:
            rec_X_prime_loss_ = (X_prime_hat_diff - X_prime).abs()

        mse = (X_prime- X_prime_hat).pow(2).sum((2,3,4)).sqrt().mean()
        loss = rec_X_loss.mean() + rec_X_prime_loss_.mean() + rec_X_prime_loss.mean()
        return loss, mse

