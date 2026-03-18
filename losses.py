# losses.py
# 损失函数模块：包含内区、边界层、边界条件、初始条件等各阶残差项

import torch
import utils
from config import Config


# ==================== 内区损失 (O(1) 阶) ====================

def loss_pde_inner(t, x, y, u_in, w_in, p_in, eta, cfg):
    """
    内区 O(1) 阶 PDE 残差
    包括：连续性方程、x-动量方程、z-动量方程（压力与y无关）
    """
    # 计算 η 的导数
    eta_t, eta_x, eta_xx, eta_xxx, eta_xxxx = utils.compute_eta_derivatives(eta, t, x)

    # 计算 u_in 的物理导数
    u_t, u_x, u_z, u_zz = utils.compute_inner_derivatives(u_in, t, x, y, eta, eta_x, eta_t)

    # 计算 w_in 的物理导数（需要先对 w_in 求导）
    # 为简化，直接使用 utils 中的类似函数（可复用，但需注意变量名）
    w_t, w_x, w_z, w_zz = utils.compute_inner_derivatives(w_in, t, x, y, eta, eta_x, eta_t)

    # 连续性方程: ∂u/∂x + ∂w/∂z = 0
    cont_res = u_x + w_z
    cont_loss = torch.mean(cont_res ** 2)

    # x-动量方程: -∂P0/∂x + u_zz = 0, 其中 P0 = -Γ η_xx
    P0 = -cfg.GAMMA * eta_xx
    P0_x = utils.grad_wrapper(P0, x, retain_graph=True)
    xmom_res = -P0_x + u_zz
    xmom_loss = torch.mean(xmom_res ** 2)

    # z-动量方程: ∂/∂y (p + Π) = 0  ⇒  p + Π 应与 y 无关
    Pi = -cfg.S / eta ** 3  # Π = -S/η^3
    p_plus_pi = p_in + Pi  # 注意 p_in 是内区压力，Pi 是分子力项
    p_plus_pi_y = utils.grad_wrapper(p_plus_pi, y, retain_graph=True)
    zmom_loss = torch.mean(p_plus_pi_y ** 2)


    return cont_loss + xmom_loss + zmom_loss


# ==================== 底边界层损失 (O(ε^{-1}) 阶) ====================

def loss_bottom_xmom_0(t, x, y_b, p_tilde, eta, cfg):
    """
    方程 (4.2): -∂P0/∂x + (y_b/η) η_x ∂_yb p_tilde = 0
    """
    eta_t, eta_x, eta_xx, eta_xxx, eta_xxxx = utils.compute_eta_derivatives(eta, t, x)
    P0 = -cfg.GAMMA * eta_xx
    P0_x = utils.grad_wrapper(P0, x, retain_graph=True)
    p_tilde_yb = utils.grad_wrapper(p_tilde, y_b, retain_graph=True)
    res = -P0_x + (y_b / eta) * eta_x * p_tilde_yb
    return torch.mean(res ** 2)


def loss_bottom_cont(t, x, y_b, w_tilde, eta, cfg):
    """
    底边界层连续性方程 O(ε^{-1}): ∂_yb w_tilde = 0
    """
    w_tilde_yb = utils.grad_wrapper(w_tilde, y_b, retain_graph=True)
    return torch.mean(w_tilde_yb ** 2)


def loss_bottom_match(t, x, y_b, u_tilde, w_tilde, p_tilde):
    """
    匹配条件：当 y_b 较大时，边界层项应趋于零
    此处作为软约束，直接惩罚边界层输出的幅值（已在网络输出中乘以衰减因子，可选再加此项）
    """
    return torch.mean(u_tilde ** 2 + w_tilde ** 2 + p_tilde ** 2)


# ==================== 上边界层损失 (O(ε^{-1}) 阶) ====================

def loss_top_xmom_0(t, x, y_t, p_hat, eta, cfg):
    """
    方程 (4.5): -∂P0/∂x - (y_t/η) η_x ∂_yt p_hat = 0
    """
    eta_t, eta_x, eta_xx, eta_xxx, eta_xxxx = utils.compute_eta_derivatives(eta, t, x)
    P0 = -cfg.GAMMA * eta_xx
    P0_x = utils.grad_wrapper(P0, x, retain_graph=True)
    p_hat_yt = utils.grad_wrapper(p_hat, y_t, retain_graph=True)
    res = -P0_x - (y_t / eta) * eta_x * p_hat_yt
    return torch.mean(res ** 2)


def loss_top_cont(t, x, y_t, w_hat, eta, cfg):
    """
    上边界层连续性方程 O(ε^{-1}): ∂_yt w_hat = 0
    """
    w_hat_yt = utils.grad_wrapper(w_hat, y_t, retain_graph=True)
    return torch.mean(w_hat_yt ** 2)


def loss_top_match(t, x, y_t, u_hat, w_hat, p_hat):
    """
    上边界层匹配条件
    """
    return torch.mean(u_hat ** 2 + w_hat ** 2 + p_hat ** 2)


# ==================== 边界条件损失 ====================

def loss_bc_wall(t, x, y, u_in, w_in):
    """
    底壁无滑移条件：u=0, w=0 在 y=0
    注意：此函数应在底壁点调用，需确保输入 y=0
    """
    loss_u = torch.mean(u_in ** 2)
    loss_w = torch.mean(w_in ** 2)
    return loss_u + loss_w


def loss_bc_interface_kinematic(t, x, y, u_in, w_in, eta):
    """
    界面运动学条件：w = η_t + u η_x 在 y=1
    """
    eta_t, eta_x, eta_xx, eta_xxx, eta_xxxx = utils.compute_eta_derivatives(eta, t, x)
    # 计算 u_in 在 y=1 处的值（此处 y 已固定为1）
    # 但 u_in 是对应于输入 y 的输出，所以直接使用传入的 u_in 即可（已按 y=1 计算）
    res = w_in - (eta_t + u_in * eta_x)
    return torch.mean(res ** 2)


def loss_bc_interface_pressure(t, x, y, p_in, eta, cfg):
    """
    界面压力条件：p = -Γ η_xx + S/η^3 在 y=1
    """
    eta_t, eta_x, eta_xx, eta_xxx, eta_xxxx = utils.compute_eta_derivatives(eta, t, x)
    p_target = -cfg.GAMMA * eta_xx + cfg.S / eta ** 3
    res = p_in - p_target
    return torch.mean(res ** 2)


def loss_bc_interface_shear(t, x, y, u_in, eta):
    """
    界面切应力条件：∂u/∂z = 0 在 y=1
    由于 ∂u/∂z = (1/η) ∂u/∂y，等价于 ∂u/∂y = 0
    """
    u_y = utils.grad_wrapper(u_in, y, retain_graph=True)
    return torch.mean(u_y ** 2)


# ==================== 初始条件损失 ====================

def loss_ic_eta(t, x, eta, eta_init_func):
    """
    初始界面高度条件：η(t=0,x) = η_init(x)
    eta_init_func 是用户提供的初始函数，需返回与 x 同形状的张量
    """
    eta_true = eta_init_func(x)
    return torch.mean((eta - eta_true) ** 2)


def loss_ic_velocity(t, x, y, u_in, w_in):
    """
    初始速度条件：u=0, w=0
    """
    loss_u = torch.mean(u_in ** 2)
    loss_w = torch.mean(w_in ** 2)
    return loss_u + loss_w


def loss_ic_pressure(t, x, y, p_in, p0=0.0):
    """
    初始压力条件（通常取 p = p_g = 0 无量纲）
    """
    return torch.mean((p_in - p0) ** 2)