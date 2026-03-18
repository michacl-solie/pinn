# utils.py
# 工具函数：自动微分包装、坐标变换下的导数计算

import torch
from torch.autograd import grad

def grad_wrapper(y, x, create_graph=True, retain_graph=True):
    """
    自动微分包装函数，计算 dy/dx
    :param y: 依赖 x 的张量
    :param x: 自变量张量
    :param create_graph: 是否创建计算图用于高阶导数
    :param retain_graph: 是否保留计算图以便多次 backward
    :return: dy/dx
    """
    return grad(y, x, grad_outputs=torch.ones_like(y),
                create_graph=create_graph, retain_graph=retain_graph)[0]


def compute_eta_derivatives(eta, t, x):
    """
    计算界面高度 η 的导数
    :param eta: 界面高度张量 (N,1)
    :param t: 时间坐标 (N,1)
    :param x: 水平坐标 (N,1)
    :return: (η_t, η_x, η_xx, η_xxx, η_xxxx)
    """
    eta_t = grad_wrapper(eta, t, retain_graph=True)
    eta_x = grad_wrapper(eta, x, retain_graph=True)
    eta_xx = grad_wrapper(eta_x, x, retain_graph=True)
    eta_xxx = grad_wrapper(eta_xx, x, retain_graph=True)
    eta_xxxx = grad_wrapper(eta_xxx, x, retain_graph=True)
    return eta_t, eta_x, eta_xx, eta_xxx, eta_xxxx


def compute_inner_derivatives(u, t, x, y, eta, eta_x, eta_t):
    """
    计算内区变量 u 的物理导数，考虑坐标变换 y = z/η
    :param u: 内区变量张量 (N,1)
    :param t: 时间坐标 (N,1)
    :param x: 水平坐标 (N,1)
    :param y: 归一化垂直坐标 (N,1)
    :param eta: 界面高度 (N,1)
    :param eta_x: η 对 x 的偏导
    :param eta_t: η 对 t 的偏导
    :return: (u_t, u_x, u_z, u_zz)
    """
    # 计算 u 对 (t, x, y) 的偏导（保持 y 固定）
    u_t_y = grad_wrapper(u, t, retain_graph=True)
    u_x_y = grad_wrapper(u, x, retain_graph=True)
    u_y   = grad_wrapper(u, y, retain_graph=True)
    u_yy  = grad_wrapper(u_y, y, retain_graph=True)

    # 物理空间导数（链式法则）
    u_t = u_t_y - (y / eta) * eta_t * u_y
    u_x = u_x_y - (y / eta) * eta_x * u_y
    u_z = u_y / eta
    u_zz = u_yy / eta**2
    return u_t, u_x, u_z, u_zz


def compute_bottom_derivatives(u_tilde, t, x, y_b, eta, eps):
    """
    计算底边界层变量 u_tilde 的物理导数
    拉伸坐标关系: y_b = y / ε, 且 ∂/∂z = 1/(εη) ∂/∂y_b
    :param u_tilde: 底边界层变量 (N,1)
    :param t: 时间坐标 (N,1)
    :param x: 水平坐标 (N,1)
    :param y_b: 底边界层拉伸坐标 (N,1)
    :param eta: 界面高度 (N,1)
    :param eps: 小参数 ε
    :return: (u_t, u_x, u_z, u_zz)
    """
    u_t = grad_wrapper(u_tilde, t, retain_graph=True)
    u_x = grad_wrapper(u_tilde, x, retain_graph=True)
    u_yb = grad_wrapper(u_tilde, y_b, retain_graph=True)
    u_yb_yb = grad_wrapper(u_yb, y_b, retain_graph=True)

    u_z = u_yb / (eps * eta)
    u_zz = u_yb_yb / (eps**2 * eta**2)
    return u_t, u_x, u_z, u_zz


def compute_top_derivatives(u_hat, t, x, y_t, eta, eps):
    """
    计算上边界层变量 u_hat 的物理导数
    拉伸坐标关系: y_t = (1-y)/ε, 且 ∂/∂z = -1/(εη) ∂/∂y_t
    :param u_hat: 上边界层变量 (N,1)
    :param t: 时间坐标 (N,1)
    :param x: 水平坐标 (N,1)
    :param y_t: 上边界层拉伸坐标 (N,1)
    :param eta: 界面高度 (N,1)
    :param eps: 小参数 ε
    :return: (u_t, u_x, u_z, u_zz)
    """
    u_t = grad_wrapper(u_hat, t, retain_graph=True)
    u_x = grad_wrapper(u_hat, x, retain_graph=True)
    u_yt = grad_wrapper(u_hat, y_t, retain_graph=True)
    u_yt_yt = grad_wrapper(u_yt, y_t, retain_graph=True)

    u_z = -u_yt / (eps * eta)
    u_zz = u_yt_yt / (eps**2 * eta**2)
    return u_t, u_x, u_z, u_zz