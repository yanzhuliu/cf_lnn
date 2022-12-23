"""
code modified from https://github.com/Physics-aware-AI/DiffCoSim/
"""

from systems.rigid_body import RigidBody, BodyGraph
import numpy as np
import torch
import networkx as nx
from ode_models.contact_model import ContactModel
from baselines.lcp.contact_model_lcp import ContactModelLCP


class ChainPendulumWithContact(RigidBody):
    dt = 0.01
    integration_time = 1.0

    def __init__(
        self, 
        kwargs_file_name="default",
        n_o=2, 
        g=9.81,
        ms=[0.1, 0.1], 
        ls=[0.7, 0.7], 
        radii=[0.1, 0.1], 
        mus=[0.0, 0.0, 0.0, 0.0], 
        cors=[1.0, 1.0, 1.0, 1.0], 
        bdry_lin_coef=[[0, 1, 1]],
        is_homo=True,
        is_reg_data=False,
        is_reg_model=False,
        is_lcp_data=False,
        is_lcp_model=False,
        dtype=torch.float64
    ):
        assert n_o == len(ms) == len(ls) == len(radii)
        assert is_homo
        assert not (is_reg_model and is_lcp_model)
        self.body_graph = BodyGraph()
        self.kwargs_file_name = kwargs_file_name
        self.ms = torch.tensor(ms, dtype=dtype)
        self.ls = torch.tensor(ls, dtype=dtype)
        self.radii = torch.tensor(radii, dtype=dtype)
        self.body_graph.add_extended_body(0, ms[0], d=0, tether=(torch.zeros(2), ls[0]))
        for i in range(1, n_o):
            self.body_graph.add_extended_body(i, ms[i], d=0)
            self.body_graph.add_edge(i-1, i, l=ls[i])
        self.g = g
        self.n_o, self.n_p, self.d = n_o, 1, 2
        self.n = self.n_o * self.n_p

        self.bdry_lin_coef = torch.tensor(bdry_lin_coef, dtype=dtype)
        self.n_c = n_o * self.bdry_lin_coef.shape[0]
        assert len(mus) == len(cors) == 1
        self.mus = torch.tensor(mus*self.n_c, dtype=torch.float64)
        self.cors = torch.tensor(cors*self.n_c, dtype=torch.float64)
        self.is_homo = is_homo
        self.is_reg_model = is_reg_model
        self.is_reg_data = is_reg_data
        self.is_lcp_model = is_lcp_model
        self.is_lcp_data = is_lcp_data

        if is_lcp_model:
            self.impulse_solver = ContactModelLCP(
                dt = self.dt,
                n_o = self.n_o,
                n_p = self.n_p,
                d = self.d,
                ls = self.ls,
                bdry_lin_coef = self.bdry_lin_coef,
                check_collision = self.check_collision,
                cld_2did_to_1did = self.cld_2did_to_1did,
                DPhi = self.DPhi
            )     
        elif is_reg_model:
            self.impulse_solver = ContactModelReg(
                dt = self.dt,
                n_o = self.n_o,
                n_p = self.n_p,
                d = self.d,
                ls = self.ls,
                bdry_lin_coef = self.bdry_lin_coef,
                check_collision = self.check_collision,
                cld_2did_to_1did = self.cld_2did_to_1did,
                DPhi = self.DPhi
            )   
        else:
            self.impulse_solver = ContactModel(
                dt = self.dt,
                n_o = self.n_o,
                n_p = self.n_p,
                d = self.d,
                ls = self.ls,
                bdry_lin_coef = self.bdry_lin_coef,
                check_collision = self.check_collision,
                cld_2did_to_1did = self.cld_2did_to_1did,
                DPhi = self.DPhi
            )   

    def __str__(self):
        if self.is_reg_data:
            return f"{self.__class__.__name__}_{self.kwargs_file_name}_reg"
        elif self.is_lcp_data:
            return f"{self.__class__.__name__}_{self.kwargs_file_name}_lcp"
        else:
            return f"{self.__class__.__name__}_{self.kwargs_file_name}"

    def potential(self, x):
        M = self.M.to(dtype=x.dtype)
        return self.g * (M @ x)[..., 1].sum(1) 

    def sample_initial_conditions(self, N, dtype=torch.float64):
        n = len(self.body_graph.nodes)
        xv0_list = []
        ptr = 0
        while ptr < N:
            q0 = torch.rand([N, n], dtype=dtype) * 3.14
            q_dot0 = torch.randn([N, n], dtype=dtype)
            q_q_dot0 = torch.stack([q0, q_dot0], dim=1)
            xv0 = self.angle_to_global_cartesian(q_q_dot0) # (N, 2, n, d)
            is_collide, *_ = self.check_collision(xv0[:, 0])
            xv0_list.append(xv0[torch.logical_not(is_collide)])
            ptr += sum(torch.logical_not(is_collide))
        return torch.cat(xv0_list, dim=0)[0:N]

    def check_collision(self, x):
        bs, n, _ = x.shape
        is_cld, is_cld_bdry, dist_bdry = self.check_boundary_collision(x)
        is_cld_ij = torch.zeros(bs, n, n, dtype=torch.bool, device=x.device)
        dist_ij = torch.zeros(bs, n, n, dtype=x.dtype, device=x.device)
        is_cld_limit = torch.zeros(bs, 0, 2, dtype=torch.bool, device=x.device)
        dist_limit = torch.zeros(bs, 0, 2).type_as(x)
        return is_cld, is_cld_ij, is_cld_bdry, is_cld_limit, dist_ij, dist_bdry, dist_limit

    def check_boundary_collision(self, x):
        coef = self.bdry_lin_coef / (self.bdry_lin_coef[:, 0:1] ** 2 + 
                    self.bdry_lin_coef[:, 1:2] ** 2).sqrt() # n_bdry, 3
        x_one = torch.cat(
            [x, torch.ones(*x.shape[:-1], 1, dtype=x.dtype, device=x.device)],
            dim=-1
        ).unsqueeze(-2) # bs, n, 1, 3
        dist = (x_one * coef.type_as(x)).sum(-1) # bs, n, n_bdry
        dist_bdry = dist - self.radii[:, None].type_as(x)
        is_collide_bdry = dist_bdry < 0 # bs, n, n_bdry
        is_collide = is_collide_bdry.sum([1, 2]) > 0
        return is_collide, is_collide_bdry, dist_bdry

    def cld_2did_to_1did(self, cld_ij_ids, cld_bdry_ids, cld_limit_ids):
        n = self.n
        link_i, bdry_i = cld_bdry_ids.unbind(dim=1)
        bdry_1d_ids = link_i * self.bdry_lin_coef.shape[0] + bdry_i
        return bdry_1d_ids

    def angle_to_global_cartesian(self, q_q_dot):
        *N2, n = q_q_dot.shape
        d = 2
        r_r_dot = torch.zeros(*N2, n, d, device=q_q_dot.device, dtype=q_q_dot.dtype)
        joint_r_r_dot = torch.zeros(*N2, d, device=q_q_dot.device, dtype=q_q_dot.dtype)

        l0 = self.body_graph.nodes[0]["tether"][1]
        joint_r_r_dot[..., 0, :] = self.body_graph.nodes[0]["tether"][0]
        joint_r_r_dot += self.angle_to_local_cartesian(l0, q_q_dot[..., 0])
        r_r_dot[..., 0, :] = joint_r_r_dot
        for (i_prev, i), l in nx.get_edge_attributes(self.body_graph, "l").items():
            joint_r_r_dot += self.angle_to_local_cartesian(l, q_q_dot[..., i])
            r_r_dot[..., i, :] += joint_r_r_dot
        return r_r_dot
    
    def angle_to_local_cartesian(self, l, q_q_dot):
        # input *bsT, 2, output *bsT, 2, 2
        *bsT, _ = q_q_dot.shape
        r_r_dot = torch.zeros(*bsT, 2, 2, device=q_q_dot.device, dtype=q_q_dot.dtype)
        r_r_dot[..., 0, 0] = l * q_q_dot[..., 0].sin() # x
        r_r_dot[..., 0, 1] = - l * q_q_dot[..., 0].cos() # y
        r_r_dot[..., 1, 0] = l * q_q_dot[..., 0].cos() * q_q_dot[..., 1] # x_dot
        r_r_dot[..., 1, 1] = l * q_q_dot[..., 0].sin() * q_q_dot[..., 1] # y_dot
        return r_r_dot

    def global_cartesian_to_angle(self, r_r_dot):
        *NT2, n, d = r_r_dot.shape
        q_q_dot = torch.zeros(*NT2, n, device=r_r_dot.device, dtype=r_r_dot.dtype)
        joint_r_r_dot = torch.zeros(*NT2, d, device=r_r_dot.device, dtype=r_r_dot.dtype)
        joint_r_r_dot[..., 0, :] = self.body_graph.nodes[0]["tether"][0] # position
        rel_r_r_dot = r_r_dot[..., 0, :] - joint_r_r_dot # *NT2, d
        q_q_dot[..., 0] += self.local_cartesian_to_angle(rel_r_r_dot)
        joint_r_r_dot += rel_r_r_dot
        for (i_prev, i), l in nx.get_edge_attributes(self.body_graph, "l").items():
            rel_r_r_dot = r_r_dot[..., i, :] - joint_r_r_dot
            q_q_dot[..., i] += self.local_cartesian_to_angle(rel_r_r_dot)
            joint_r_r_dot += rel_r_r_dot
        return q_q_dot
        
    def local_cartesian_to_angle(self, rel_r_r_dot):
        assert rel_r_r_dot.ndim >= 4
        x, y = rel_r_r_dot[..., 0, :].chunk(2, dim=-1) # *NT1, *NT1
        vx, vy = rel_r_r_dot[..., 1, :].chunk(2, dim=-1)
        q = torch.atan2(x, -y)
        q_dot = torch.where(q < 1e-2, vx / (-y), vy / x)
        q_unwrapped = torch.from_numpy(np.unwrap(q.detach().cpu().numpy(), axis=-2)).to(x.device, x.dtype)
        return torch.cat([q_unwrapped, q_dot], dim=-1) # *NT2

    def angle_to_cos_sin(self, q_q_dot):
        # input shape (*NT, 2, n)
        q, q_dot = q_q_dot.chunk(2, dim=-2)
        return torch.stack([q.cos(), q.sin(), q_dot], dim=-2)