"""
code modified from https://github.com/Physics-aware-AI/DiffCoSim/
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from ode_models.ode_models import ConstrainedLagrangianDynamics, LagrangianDynamics
from utils import Linear, mlp, Reshape, CosSin, _stable_division, MLPSimple
from systems.rigid_body import EuclideanT, GeneralizedT, rigid_DPhi

from torchdiffeq import odeint


class CFLNN(nn.Module):
    def __init__(self, body_graph, impulse_solver, n_c, d,
        is_homo=False,
        hidden_size: int = 256,
        num_layers: int = 3,
        dtype: torch.dtype = torch.float32,
        reg=0.01,
        **kwargs
    ):
        super().__init__()
        self.body_graph = body_graph
        self.n = len(body_graph.nodes)
        self.d, self.n_c = d, n_c
        self.dtype = dtype
        self.nfe = 0
        self.kwargs = kwargs
        self.dynamics = ConstrainedLagrangianDynamics(self.potential, self.Minv_op, 
                                                      self.DPhi, (self.n, self.d))
        sizes = [self.n * self.d] + num_layers * [hidden_size] + [1]
        self.V_net_raw = nn.Sequential(
            mlp(sizes, nn.Softplus, orthogonal_init=True),
            Reshape(-1)
        )

        self.V_net_treat = nn.Sequential(
            mlp(sizes, nn.Softplus, orthogonal_init=True),
            Reshape(-1)
        )
        self.V_net = self.V_net_raw

        self.raw_params = nn.ParameterDict(
             {str(d): nn.Parameter(0.1 * torch.randn(len(ids)//(d+1), d+1, dtype=dtype)) # N, d+1
                 for d, ids in body_graph.d2ids.items()}
        )

        self.treat_params = nn.ParameterDict(
             {str(d): nn.Parameter(0.1 * torch.randn(len(ids)//(d+1), d+1, dtype=dtype)) # N, d+1
                 for d, ids in body_graph.d2ids.items()}
        )
        self.m_params = self.raw_params

        obsrv_std = 0.01
        self.obsrv_std = obsrv_std*torch.ones(self.raw_params[str(0)].shape, dtype=torch.float16)
        self.treatment_fun = MLPSimple(input_dim=1, output_dim=1, hidden_dim=20, depth=4,
                                       activations=[nn.ReLU() for _ in range(4)])

        assert len(self.raw_params) == 1 # limited support for now
        self.n_p = int(list(self.raw_params.keys())[0]) + 1
        self.n_o = self.n // self.n_p
        if is_homo:
            self.mu_params = nn.Parameter(torch.rand(1, dtype=dtype))
            self.cor_params = nn.Parameter(torch.randn(1, dtype=dtype))
        else:
            self.mu_params = nn.Parameter(torch.rand(n_c, dtype=dtype))
            self.cor_params = nn.Parameter(torch.randn(n_c, dtype=dtype))
        self.is_homo = is_homo

        if impulse_solver.__class__.__name__ == "ContactModelReg":
            if reg < 0:
                # override the reg with a learnable parameter
                self.reg = nn.Parameter(torch.randn(1, dtype=dtype))
                impulse_solver.reg = self.reg
            else:
                impulse_solver.reg = reg

        self.impulse_solver = impulse_solver
        self.diff_func = MLPSimple(input_dim=6, output_dim=2, hidden_dim=20, depth=4,
                                       activations=[nn.ReLU() for _ in range(4)])

    @property
    def Minv(self):
        n = self.n
        # d == n_p-1
        d = int(list(self.m_params.keys())[0])

        inv_moments = torch.exp(-self.m_params[str(d)]) # n_o, n_p
        inv_masses = inv_moments[:, :1] # n_o, 1
        if d == 0:
            return torch.diag_embed(inv_masses[:, 0])
        blocks = torch.diag_embed(torch.cat([0*inv_masses, inv_moments[:, 1:]], dim=-1)) # n_o, n_p, n_p
        blocks = blocks + torch.ones_like(blocks)
        blocks = blocks * inv_masses.unsqueeze(-1)
        return torch.block_diag(*blocks) # (n, n)

    def Minv_op(self, p):
        assert len(self.m_params) == 1 
        *dims, n, a = p.shape
        d = int(list(self.m_params.keys())[0])
        n_o = n // (d+1) # number of extended bodies
        p_reshaped = p.reshape(*dims, n_o, d+1, a)
        inv_moments = torch.exp(-self.m_params[str(d)])
        inv_masses = inv_moments[:, :1] # n_o, 1
        if d == 0:
            return (inv_masses.unsqueeze(-1)*p_reshaped).reshape(*p.shape)
        inv_mass_p = inv_masses.unsqueeze(-1) * p_reshaped.sum(-2, keepdims=True) # (n_o, 1, 1) * (..., n_o, 1, a) = (..., n_o, 1, a)
        padded_intertias_inv = torch.cat([0*inv_masses, inv_moments[:, 1:]], dim=-1) # (n_o, d+1)
        inv_intertia_p = padded_intertias_inv.unsqueeze(-1) * p_reshaped # (n_o, d+1, 1) * (..., n_o, d+1, a) = (..., n_o, d+1, a)
        return (inv_mass_p + inv_intertia_p).reshape(*p.shape)

    def forward(self, t, z):
        self.nfe += 1
        return self.dynamics(t, z)

    def DPhi(self, x, x_dot):
        return rigid_DPhi(self.body_graph, x, x_dot)

    def potential(self, x):
        assert x.ndim == 3
        return self.V_net(x.reshape(x.shape[0], -1))

    def forward_diff(self, t, z):
        z = self.diff_func(z) # bz, obj, dim, 2
        z = torch.cat([torch.zeros(z.shape).to(z), torch.zeros(z.shape).to(z), z], dim=-1)
        return z

    def mask(self, z):
        return torch.cat((z[:,:,:-1,:], torch.zeros(z.shape).sum(2, keepdim=True).to(z)),dim=2)

    def integrate_diff(self, zts, is_clds, z0_, ts, tol=1e-4, method="rk4", cf3=False, treat=False):
        # need to compute contact on zts_, not on zts
        if self.is_homo:
            mus = F.relu(self.mu_params * torch.ones(self.n_c).type_as(self.mu_params))
            cors = F.hardsigmoid(self.cor_params * torch.ones(self.n_c).type_as(self.cor_params))
        else:
            mus = F.relu(self.mu_params)
            cors = F.hardsigmoid(self.cor_params)

        bs = z0_.shape[0]
        ts = ts.to(z0_.device, z0_.dtype)
        zT_ = torch.zeros([bs, len(ts), 2 * self.n* self.d], device=z0_.device, dtype=z0_.dtype)
        zt_ = z0_.clone()
        zT_[:, 0] = zt_.reshape(bs, -1)
        for i in range(len(ts) - 1):
            if is_clds[:,i+1].sum() == 0:  # no collision in time i+1 for all bz, the index with is_clds already add impulse
                zt = zts[:, i]  # bz, 2, obj, dim
                zt_n = zts[:, i + 1]  # t+1 in X

                if treat:
                    self.m_params = self.treat_params
                    self.V_net = self.V_net_treat
                if cf3:
                    zt_ = self.mask(zt_)
                zt_ = zt_.reshape(bs, -1)
                z_diff_ = odeint(self.dynamics, zt_, ts[i:i + 2], rtol=tol, method=method)[1].reshape(bs, 2, self.n, self.d)
                if cf3:
                    z_diff_ = self.mask(z_diff_)

                self.m_params = self.raw_params
                self.V_net = self.V_net_raw
                zt = zt.reshape(bs, -1)
                z_diff = odeint(self.dynamics, zt, ts[i:i + 2], rtol=tol, method=method)[1].reshape(bs, 2, self.n, self.d)
                z_diff = z_diff_ - z_diff

                zt_n_ = zt_n + z_diff
                zt_n_ = zt_n_.reshape(bs, -1)
                if treat:
                    self.m_params = self.treat_params
                    self.V_net = self.V_net_treat
                zt_n_, _ = self.impulse_solver.add_impulse(zt_n_, mus, cors, self.Minv, cf3)
                zt_ = zt_n_
                zT_[:, i + 1] = zt_
                zt_ = zt_.reshape(bs, 2, self.n, self.d)

                continue

            zt_cls_ = zt_[is_clds[:,i+1]]

            zt_cls_ = zt_cls_.reshape(zt_cls_.shape[0], -1)

            # x' on treatment
            if treat:
                self.m_params = self.treat_params
                self.V_net = self.V_net_treat

            zt_cls_n_ = odeint(self, zt_cls_, ts[i:i + 2], rtol=tol, method=method)[1]
            zt_cls_n_, _ = self.impulse_solver.add_impulse(zt_cls_n_, mus, cors, self.Minv, cf3)

            zt_n_cls_ = zt_[~is_clds[:, i+1]]
            if zt_n_cls_.shape[0] == 0:  # all bz has collision
                zt_ = zt_cls_n_
                zT_[:, i + 1] = zt_
                zt_ = zt_.reshape(bs, 2, self.n, self.d)
                continue
            else:
                zt = zts[~is_clds[:, i+1], i]
                zt_n = zts[~is_clds[:, i+1], i + 1]  # t+1 in X

                if treat:
                    self.m_params = self.treat_params
                    self.V_net = self.V_net_treat
                if cf3:
                    zt_n_cls_ = self.mask(zt_n_cls_)
                zt_n_cls_ = zt_n_cls_.reshape(zt_n_cls_.shape[0], -1)
                z_diff_ = odeint(self.dynamics, zt_n_cls_, ts[i:i + 2], rtol=tol, method=method)[1].reshape(zt_n_cls_.shape[0], 2, self.n,
                                                                                                  self.d)
                if cf3:
                    z_diff_ = self.mask(z_diff_)

                self.m_params = self.raw_params
                self.V_net = self.V_net_raw
                zt = zt.reshape(zt.shape[0], -1)
                z_diff = odeint(self.dynamics, zt, ts[i:i + 2], rtol=tol, method=method)[1].reshape(zt_n_cls_.shape[0], 2, self.n,
                                                                                                    self.d)
                z_diff = z_diff_ - z_diff

                zt_n_cls_n_ = zt_n + z_diff
                zt_n_cls_n_ = zt_n_cls_n_.reshape(zt_n_cls_.shape[0], -1)

                if treat:
                    self.m_params = self.treat_params
                    self.V_net = self.V_net_treat

                zt_n_cls_n_, _ = self.impulse_solver.add_impulse(zt_n_cls_n_, mus, cors, self.Minv, cf3)

                zt_ = zt_.reshape(zt_.shape[0], -1)
                zt_[is_clds[:, i+1]] = zt_cls_n_
                zt_[~is_clds[:, i+1]] = zt_n_cls_n_

                zT_[:, i + 1] = zt_
                zt_ = zt_.reshape(bs, 2, self.n, self.d)

        return zT_.reshape(bs, len(ts), 2, self.n, self.d)

    def integrate(self, z0, ts, tol=1e-4, method="rk4", treat=False, cf3=False):
        """
        input:
            z0: bs, 2, n, d
            ts: length T
        returns:
            a tensor of size bs, T, 2, n, d
        """
        if treat:
            self.m_params = self.treat_params
            self.V_net = self.V_net_treat
        else:
            self.m_params = self.raw_params
            self.V_net = self.V_net_raw

        assert (z0.ndim == 4) and (ts.ndim == 1)
        assert (z0.shape[-1] == self.d) and z0.shape[-2] == self.n
        bs = z0.shape[0]
        if self.is_homo:
            mus = F.relu(self.mu_params * torch.ones(self.n_c).type_as(self.mu_params))
            cors = F.hardsigmoid(self.cor_params* torch.ones(self.n_c).type_as(self.cor_params))
        else:
            mus = F.relu(self.mu_params)
            cors = F.hardsigmoid(self.cor_params)
        ts = ts.to(z0.device, z0.dtype)
        zt = z0.reshape(bs, -1)
        zT = torch.zeros([bs, len(ts), zt.shape[1]], device=z0.device, dtype=z0.dtype)
        zT[:, 0] = zt
        for i in range(len(ts)-1):
            zt_n = odeint(self, zt, ts[i:i+2], rtol=tol, method=method)[1]
            zt_n, _ = self.impulse_solver.add_impulse(zt_n, mus, cors, self.Minv, cf3)
            zt = zt_n
            zT[:, i+1] = zt
        return zT.reshape(bs, len(ts), 2, self.n, self.d)

    def integrate_onestep(self, zts, ts, tol=1e-4, method="rk4"):
        """
        input:
            zts: bs, T, 2, n, d
            ts: length T
        returns:
            a tensor of size bs, T, 2, n, d        """

        # x0 no treatment
        self.m_params = self.raw_params
        self.V_net = self.V_net_raw

        bs = zts.shape[0]
        if self.is_homo:
            mus = F.relu(self.mu_params * torch.ones(self.n_c).type_as(self.mu_params))
            cors = F.hardsigmoid(self.cor_params* torch.ones(self.n_c).type_as(self.cor_params))
        else:
            mus = F.relu(self.mu_params)
            cors = F.hardsigmoid(self.cor_params)
        ts = ts.to(zts.device, zts.dtype)

        zt = zts[:, 0].reshape(bs, -1)
        zT = torch.zeros([bs, len(ts), zt.shape[1]], device=zts.device, dtype=zts.dtype)
        zT[:, 0] = zt
        for i in range(len(ts)-1):
            zt_n = odeint(self, zt, ts[i:i+2], rtol=tol, method=method)[1]
            zt_n, _ = self.impulse_solver.add_impulse(zt_n, mus, cors, self.Minv, False)
            zT[:, i+1] = zt_n
            zt = zts[:,i+1].reshape(bs, -1)  # ground truth from X0 series

        return zT.reshape(bs, len(ts), 2, self.n, self.d)