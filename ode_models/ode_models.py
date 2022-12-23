"""
code modified from https://github.com/Physics-aware-AI/DiffCoSim/
"""
import torch
import torch.nn as nn
from torch import Tensor
from typing import Callable

def J(M):
    """ applies the J matrix to another matrix M.
        input: M (*,2nd,b), output: J@M (*,2nd,b)"""
    *star, D, b = M.shape
    JM = torch.cat([M[..., D // 2 :, :], -M[..., : D // 2, :]], dim=-2)
    return JM

def Proj(DPhi):
    if DPhi.shape[-1]==0: return lambda M:M # (no constraints)
    def _P(M):
        DPhiT = DPhi.transpose(-1, -2)
        X, _ = torch.solve(DPhiT @ M, DPhiT @ J(DPhi))
        return M - J(DPhi @ X)

    return _P

class LagrangianDynamics(nn.Module):
    def __init__(self, L: Callable[[Tensor, Tensor], Tensor]):
        super().__init__()
        self.L = L

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        assert (t.ndim == 0) and (z.ndim == 2)
        D = z.shape[-1] ; d = D// 2
        with torch.enable_grad():
            q = z[..., :d]
            v = z[..., d:]
            q = q + torch.zeros_like(q, requires_grad=True)
            v = v + torch.zeros_like(v, requires_grad=True)
            z = torch.cat([q, v], dim=-1)
            L = self.L(t, z).sum()
            dL_dz = torch.autograd.grad(L, z, create_graph=True)[0]
            dL_dq = dL_dz[..., :d]
            dL_dv = dL_dz[..., d:]
            
            d2L_dvdq_v = torch.autograd.grad((dL_dq * v.detach()).sum(), v, create_graph=True)[0] # allow_unused=True
            eye = torch.eye(d, device=z.device, dtype=z.dtype)
            d2L_d2v = torch.stack(
                [torch.autograd.grad((dL_dv * eye[i]).sum(),v, create_graph=True)[0]
                    for i in range(d)], dim=-1
            )
            RHS = (dL_dq - d2L_dvdq_v).unsqueeze(-1)
            vdot = torch.solve(RHS, d2L_d2v)[0].squeeze(-1)
            dynamics = torch.cat([v, vdot], dim=-1)
            return dynamics

class ConstrainedLagrangianDynamics(nn.Module):
    """
    Assume cartesian coordinates of rigid body systems
    """
    def __init__(self,
        V,
        Minv: Callable[[Tensor], Tensor],
        DPhi: Callable[[Tensor, Tensor], Tensor],
        shape
    ):
        super().__init__()
        self.V = V
        self.Minv = Minv
        self.DPhi = DPhi
        self.shape = shape

    def forward(self, t: Tensor, z: Tensor) -> Tensor:
        assert (t.ndim == 0) and (z.ndim == 2)
        bs, n, d = z.shape[0], *self.shape
        with torch.enable_grad():
            r, v = z.reshape(bs, 2, n, d).unbind(dim=1) # (bs, n, d)
            r = torch.zeros_like(r, requires_grad=True) + r
            dV = torch.autograd.grad(self.V(r).sum(), r, create_graph=True)[0]
            Minv_f = - self.Minv(dV).reshape(bs, n*d)                                   # (bs, nd)
            DPhi = self.DPhi(r, v) # (bs, 2, n, d, 2, C)
            if DPhi.shape[-1] != 0:
                C = DPhi.shape[-1]
                dphi_dr = DPhi[:,0,:,:,0,:].reshape(bs, n*d, C)                         # (bs, nd, C)
                dphi_dr_T = dphi_dr.permute(0, 2, 1)                                    # (bs, C, nd)
                dphi_dot_dr_T = DPhi[:,0,:,:,1,:].reshape(bs, n*d, C).permute(0,2,1)    # (bs, C, nd)
                Minv_dphi_dr = self.Minv(dphi_dr.reshape(bs,n,-1)).reshape(bs, n*d, C)  # (bs, nd, C)
                dphi_dr_T_Minv_dphi_dr = dphi_dr_T @ Minv_dphi_dr                       # (bs, C, C)
                dphi_dr_T_Minv_f = dphi_dr_T @ Minv_f.unsqueeze(-1)                     # (bs, C, 1)
                dphi_dot_dr_T_v = dphi_dot_dr_T @ v.reshape(bs, n*d, 1)                 # (bs, C, 1)
                RHS = dphi_dr_T_Minv_f + dphi_dot_dr_T_v                                # (bs, C, 1)
            #    lambda_ = torch.solve(RHS, dphi_dr_T_Minv_dphi_dr)[0]                      # (bs, C, 1)
                lambda_ = torch.linalg.solve(dphi_dr_T_Minv_dphi_dr, RHS)
                vdot = Minv_f - (Minv_dphi_dr @ lambda_).squeeze(-1)                    # (bs, nd)
            else:
                vdot = Minv_f
            dynamics = torch.cat([v.reshape(bs, n*d), vdot], dim=-1)
        return dynamics

