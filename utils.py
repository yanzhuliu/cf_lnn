"""
code modified from https://github.com/Physics-aware-AI/DiffCoSim/
"""

import torch
import torch.nn as nn
from torch.utils import data as data
from scipy.spatial.transform import Rotation
from torch import Tensor


def dummy_dataloader():
    # dummy dataloader for Lightning Module
    dummy = data.DataLoader(
        data.TensorDataset(
            torch.Tensor(1, 1),
            torch.Tensor(1, 1)
        ),
        batch_size=1,
        shuffle=False
    )
    return dummy


def Linear(chin, chout, zero_bias=False, orthogonal_init=False):
    linear = nn.Linear(chin, chout)
    if zero_bias:
        torch.nn.init.zeros_(linear.bias)
    if orthogonal_init:
        torch.nn.init.orthogonal_(linear.weight)
    return linear


def mlp(sizes, activation, output_activation=nn.Identity, orthogonal_init=True):
    layers = []
    for i in range(len(sizes)-1):
        act = activation if i < len(sizes)-2 else output_activation
        layers += [Linear(sizes[i], sizes[i+1], orthogonal_init=orthogonal_init), act()]
    return nn.Sequential(*layers)


class MLPSimple(nn.Module):
    def __init__(self,input_dim,output_dim, hidden_dim, depth, activations = None, dropout_p = None):
        super().__init__()
        self.input_layer = nn.Sequential(nn.Linear(input_dim,hidden_dim),nn.ReLU())
        self.output_layer = nn.Sequential(nn.Linear(hidden_dim,output_dim))
        if activations is None:
            activations = [nn.ReLU() for _ in range(depth)]
        if dropout_p is None:
            dropout_p = [0. for _ in range(depth)]
        assert len(activations) == depth
        self.layers = nn.ModuleList([nn.Sequential(nn.Linear(hidden_dim,hidden_dim),nn.Dropout(dropout_p[i]),activations[i]) for i in range(depth)])
    def forward(self,x):
        x = self.input_layer(x)
        for mod in self.layers:
            x = mod(x)
        x = self.output_layer(x)
        return x


class Reshape(nn.Module):
    def __init__(self, *args):
        super().__init__()
        self.shape = args

    def forward(self, x):
        return x.view(self.shape)


class CosSin(nn.Module):
    def __init__(self, q_ndim, angular_dims, only_q=True):
        super().__init__()
        self.q_ndim = q_ndim
        self.angular_dims = tuple(angular_dims)
        self.non_angular_dims = tuple(set(range(q_ndim)) - set(angular_dims))
        self.only_q = only_q

    def forward(self, q_or_qother):
        if self.only_q:
            q = q_or_qother
        else:
            q, other = q_or_qother.chunk(2, dim=-1)
        assert q.shape[-1] == self.q_ndim
        q_angular = q[..., self.angular_dims]
        q_not_angular = q[..., self.non_angular_dims]
        cos_ang_q, sin_ang_q = q_angular.cos(), q_angular.sin()
        q = torch.cat([cos_ang_q, sin_ang_q, q_not_angular], dim=-1)

        if self.only_q:
            res = q
        else:
            res = torch.cat([q, other], dim=-1)
        return res


def cross_matrix(k):
    """Application of hodge star on R3, mapping Λ^1 R3 -> Λ^2 R3"""
    K = torch.zeros(*k.shape[:-1],3,3,device=k.device,dtype=k.dtype)
    K[...,0,1] = -k[...,2]
    K[...,0,2] = k[...,1]
    K[...,1,0] = k[...,2]
    K[...,1,2] = -k[...,0]
    K[...,2,0] = -k[...,1]
    K[...,2,1] = k[...,0]
    return K


def uncross_matrix(K):
    k = torch.zeros(*K.shape[:-1],device=K.device,dtype=K.dtype)
    k[...,0] = (K[...,2,1] - K[...,1,2])/2
    k[...,1] = (K[...,0,2] - K[...,2,0])/2
    k[...,2] = (K[...,1,0] - K[...,0,1])/2
    return k


def eulerdot_to_omega_matrix(euler):
    """(*bsT, 3) -> (*bsT, 3, 3) matrix"""
    *bsT,_ = euler.shape
    M = torch.zeros(*bsT,3,3,device=euler.device,dtype=euler.dtype)
    phi,theta,psi = euler.unbind(-1)
    M[...,0,0] = theta.sin()*psi.sin()
    M[...,0,1] = psi.cos()
    M[...,1,0] = theta.sin()*psi.cos()
    M[...,1,1] = -psi.sin()
    M[...,2,0] = theta.cos()
    M[...,2,2] = 1
    return M


def euler_to_frame(euler_and_dot):
    """ input: (*bsT, 2, 3)
        output: (*bsT, 2, 3, 3) """
    *bsT, _, _ = euler_and_dot.shape
    euler, eulerdot = euler_and_dot.unbind(dim=-2) # (*bsT, 3)
    omega = (eulerdot_to_omega_matrix(euler) @ eulerdot.unsqueeze(-1)).squeeze(-1) # (*bsT, 3)
    RT_Rdot = cross_matrix(omega) 
    # Rdot_RT = cross_matrix(omega) # (*bsT, 3, 3)
    R = Rotation.from_euler("ZXZ", euler.reshape(-1, 3).detach().cpu().numpy()).as_matrix()
    R = torch.from_numpy(R).reshape(*bsT, 3, 3).to(euler.device, euler.dtype)
    Rdot = R @ RT_Rdot 
    # Rdot = Rdot_RT @ R
    return torch.stack([R, Rdot], dim=-3).transpose(-2, -1) # (bs, 2, d, n) -> (bs, 2, n, d)


def frame_to_euler(frame):
    """ input: (*bsT, 2, 3, 3) output: (*bsT, 2, 3) """
    *bsT, _, _, _ = frame.shape
    R, Rdot = frame.transpose(-2, -1).unbind(-3) # (*bsT, 3, 3)
    omega = uncross_matrix(R.transpose(-2, -1) @ Rdot) 
    # omega = uncross_matrix(Rdot @ R.transpose(-2, -1)) # (*bsT, 3)
    angles = Rotation.from_matrix(R.reshape(-1, 3, 3).detach().cpu().numpy()).as_euler("ZXZ") 
    angles = torch.from_numpy(angles).reshape(*bsT, 3).to(R.device, R.dtype) # (*bsT, 3)
    eulerdot = torch.solve(omega.unsqueeze(-1), eulerdot_to_omega_matrix(angles))[0].squeeze(-1) # (*bsT, 3)
    return torch.stack([angles, eulerdot], dim=-2) # (*bsT, 2, 3)


def com_euler_to_bodyX(com_euler):
    """ input (*bsT, 2, 6), output (*bsT, 2, 4, 3) """
    com = com_euler[..., :3] # (*bsT, 2, 3)
    frame = euler_to_frame(com_euler[..., 3:]) # (*bsT, 2, 3, 3)
    # in C frame, com would be zero
    shifted_frame = frame + com[..., None, :]
    return torch.cat([com[..., None, :], shifted_frame], dim=-2)


def bodyX_to_com_euler(X):
    """ input: (*bsT, 2, 4, 3) output: (*bsT, 2, 6) """
    com = X[..., 0, :] # (*bsT, 2, 3)
    euler = frame_to_euler(X[..., 1:, :] - com[..., None, :]) # (*bsT, 2, 3, 3) -> (*bsT, 2, 3)
    return torch.cat([com, euler], dim=-1)


# from cf-ode
def gaussian_nll_loss(input, target, var, *, full=False, eps=1e-6, reduction='mean'):
    r"""Gaussian negative log likelihood loss.
    See :class:`~torch.nn.GaussianNLLLoss` for details.
    Args:
        input: expectation of the Gaussian distribution.
        target: sample from the Gaussian distribution.
        var: tensor of positive variance(s), one for each of the expectations
            in the input (heteroscedastic), or a single one (homoscedastic).
        full: ``True``/``False`` (bool), include the constant term in the loss
            calculation. Default: ``False``.
        eps: value added to var, for stability. Default: 1e-6.
        reduction: specifies the reduction to apply to the output:
            `'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction will be applied,
            ``'mean'``: the output is the average of all batch member losses,
            ``'sum'``: the output is the sum of all batch member losses.
            Default: ``'mean'``.
    """
    if not torch.jit.is_scripting():
        tens_ops = (input, target, var)
        if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
            return handle_torch_function(
                gaussian_nll_loss, tens_ops, input, target, var, full=full, eps=eps, reduction=reduction)

    # Inputs and targets much have same shape
    #input = input.view(input.size(0), -1)
    #target = target.view(target.size(0), -1)
    if input.size() != target.size():
        raise ValueError("input and target must have same size")

    # Second dim of var must match that of input or be equal to 1
    #var = var.view(input.size(0), -1)
    if var.size() != input.size():
        raise ValueError("var is of incorrect size")

    # Check validity of reduction mode
    if reduction != 'none' and reduction != 'mean' and reduction != 'sum':
        raise ValueError(reduction + " is not valid")

    # Entries of var must be non-negative
    if torch.any(var < 0):
        raise ValueError("var has negative entry/entries")

    # Clamp for stability
    var = var.clone()
    with torch.no_grad():
        var.clamp_(min=eps)

    # Calculate loss (without constant)
    #loss = 0.5 * (torch.log(var) + (input - target)**2 / var).view(input.size(0), -1).sum(dim=1)
    loss = 0.5 * (torch.log(var) + (input - target)**2 / var)  # by lyz *-1

    # Add constant to loss term if required
    if full:
        D = input.size(1)
        loss = loss + 0.5 * D * math.log(2 * math.pi)

    # Apply reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss

class GaussianNLLLoss(torch.nn.modules.loss._Loss):
    r"""Gaussian negative log likelihood loss.

    The targets are treated as samples from Gaussian distributions with
    expectations and variances predicted by the neural network. For a
    ``target`` tensor modelled as having Gaussian distribution with a tensor
    of expectations ``input`` and a tensor of positive variances ``var`` the loss is:

    .. math::
        \text{loss} = \frac{1}{2}\left(\log\left(\text{max}\left(\text{var},
        \ \text{eps}\right)\right) + \frac{\left(\text{input} - \text{target}\right)^2}
        {\text{max}\left(\text{var}, \ \text{eps}\right)}\right) + \text{const.}

    where :attr:`eps` is used for stability. By default, the constant term of
    the loss function is omitted unless :attr:`full` is ``True``. If ``var`` is not the same
    size as ``input`` (due to a homoscedastic assumption), it must either have a final dimension
    of 1 or have one fewer dimension (with all other sizes being the same) for correct broadcasting.

    Args:
        full (bool, optional): include the constant term in the loss
            calculation. Default: ``False``.
        eps (float, optional): value used to clamp ``var`` (see note below), for
            stability. Default: 1e-6.
        reduction (string, optional): specifies the reduction to apply to the
            utput:``'none'`` | ``'mean'`` | ``'sum'``. ``'none'``: no reduction
            will be applied, ``'mean'``: the output is the average of all batch
            member losses, ``'sum'``: the output is the sum of all batch member
            losses. Default: ``'mean'``.

    Shape:
        - Input: :math:`(N, *)` where :math:`*` means any number of additional
          dimensions
        - Target: :math:`(N, *)`, same shape as the input, or same shape as the input
          but with one dimension equal to 1 (to allow for broadcasting)
        - Var: :math:`(N, *)`, same shape as the input, or same shape as the input but
          with one dimension equal to 1, or same shape as the input but with one fewer
          dimension (to allow for broadcasting)
        - Output: scalar if :attr:`reduction` is ``'mean'`` (default) or
          ``'sum'``. If :attr:`reduction` is ``'none'``, then :math:`(N, *)`, same
          shape as the input

    Examples::
        >>> loss = nn.GaussianNLLLoss()
        >>> input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> var = torch.ones(5, 2, requires_grad=True) #heteroscedastic
        >>> output = loss(input, target, var)
        >>> output.backward()

        >>> loss = nn.GaussianNLLLoss()
        >>> input = torch.randn(5, 2, requires_grad=True)
        >>> target = torch.randn(5, 2)
        >>> var = torch.ones(5, 1, requires_grad=True) #homoscedastic
        >>> output = loss(input, target, var)
        >>> output.backward()

    Note:
        The clamping of ``var`` is ignored with respect to autograd, and so the
        gradients are unaffected by it.

    Reference:
        Nix, D. A. and Weigend, A. S., "Estimating the mean and variance of the
        target probability distribution", Proceedings of 1994 IEEE International
        Conference on Neural Networks (ICNN'94), Orlando, FL, USA, 1994, pp. 55-60
        vol.1, doi: 10.1109/ICNN.1994.374138.
    """
    __constants__ = ['full', 'eps', 'reduction']
    full: bool
    eps: float

    def __init__(self, *, full: bool = False, eps: float = 1e-6, reduction: str = 'mean') -> None:
        super(GaussianNLLLoss, self).__init__(None, None, reduction)
        self.full = full
        self.eps = eps

    def forward(self, input: Tensor, target: Tensor, var: Tensor) -> Tensor:
        return gaussian_nll_loss(input, target, var, full=self.full, eps=self.eps, reduction=self.reduction)

def sample_standard_gaussian(mu, sigma):
    d = torch.distributions.normal.Normal(torch.Tensor([0.]).to(mu), torch.Tensor([1.]).to(sigma))
    r = d.sample(mu.size()).squeeze(-1)
    return r * sigma.float() + mu.float()

def str2bool(value, raise_exc=False):
    _true_set = {'yes', 'true', 't', 'y', '1'}
    _false_set = {'no', 'false', 'f', 'n', '0'}

    if isinstance(value, str) or sys.version_info[0] < 3 and isinstance(value, basestring):
        value = value.lower()
        if value in _true_set:
            return True
        if value in _false_set:
            return False

    if raise_exc:
        raise ValueError('Expected "%s"' % '", "'.join(_true_set | _false_set))
    return None


def _stable_division(a, b, epsilon=1e-7):
    b = torch.where(b.abs().detach() > epsilon, b, torch.full_like(b, fill_value=epsilon) * ((b > 0).float() * 2 - 1))
    return a / b


class LinearScheduler(object):
    def __init__(self, iters, maxval=1.0, start=0):
        self._iters = max(1, iters)
        self._val = 0  # maxval / self._iters
        self._maxval = maxval
        self._start = start
        self.current_iter = 0

    def step(self):
        if self.current_iter > self._iters:
            self._val = min(self._maxval, self._val + self._maxval / self._iters)
        self.current_iter += 1

    @property
    def val(self):
        return self._val