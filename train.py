# train.py
# 训练器类（优化版：返回并打印所有损失分量）

import torch
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau
from data import DataSampler
from networks import InnerNet, BottomNet, TopNet, EtaNet
import losses

class Trainer:
    def __init__(self, cfg, device):
        self.cfg = cfg
        self.device = device
        self.sampler = DataSampler(cfg, device)

        # 初始化网络
        self.inner_net = InnerNet(cfg.INNER_LAYERS).to(device)
        self.bottom_net = BottomNet(cfg.BOTTOM_LAYERS, cfg.EPS).to(device)
        self.top_net = TopNet(cfg.TOP_LAYERS, cfg.EPS).to(device)
        self.eta_net = EtaNet(cfg.ETA_LAYERS).to(device)

        # 合并所有网络参数
        params = (list(self.inner_net.parameters()) +
                  list(self.bottom_net.parameters()) +
                  list(self.top_net.parameters()) +
                  list(self.eta_net.parameters()))
        self.optimizer = Adam(params, lr=cfg.LR)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', patience=500, factor=0.5)

        # 混合精度 scaler（使用新 API 消除警告）
        self.scaler = torch.amp.GradScaler('cuda', enabled=True)

        # 存储固定点集
        self.fixed_points = None

        # 损失历史
        self.loss_history = []

    def generate_fixed_points(self):
        """生成固定点集（每个 epoch 重新采样一次）"""
        # 内区
        t_int, x_int, y_int = self.sampler.sample_interior(self.cfg.N_INTERIOR)
        # 底边界层
        t_bot, x_bot, yb = self.sampler.sample_bottom_layer(self.cfg.N_BOTTOM)
        # 上边界层
        t_top, x_top, yt = self.sampler.sample_top_layer(self.cfg.N_TOP)
        # 底壁边界（内区点）
        t_bc_b, x_bc_b, y_bc_b = self.sampler.sample_boundary_bottom(self.cfg.N_BC // 2)
        # 界面边界（内区点）
        t_bc_t, x_bc_t, y_bc_t = self.sampler.sample_boundary_top(self.cfg.N_BC - self.cfg.N_BC // 2)
        # 初始条件（内区点）
        t_ic, x_ic, y_ic = self.sampler.sample_initial(self.cfg.N_IC)
        # 匹配区域
        t_mb, x_mb, yb_m = self.sampler.sample_match_bottom(self.cfg.N_MATCH)
        t_mt, x_mt, yt_m = self.sampler.sample_match_top(self.cfg.N_MATCH)
        # 底壁边界层点
        t_wb, x_wb, yb_w = self.sampler.sample_wall_bottom(self.cfg.N_BC // 2)

        self.fixed_points = {
            'int': (t_int, x_int, y_int),
            'bot': (t_bot, x_bot, yb),
            'top': (t_top, x_top, yt),
            'bc_b': (t_bc_b, x_bc_b, y_bc_b),
            'bc_t': (t_bc_t, x_bc_t, y_bc_t),
            'ic': (t_ic, x_ic, y_ic),
            'mb': (t_mb, x_mb, yb_m),
            'mt': (t_mt, x_mt, yt_m),
            'wb': (t_wb, x_wb, yb_w),
        }

    def train_step(self):
        """单步训练：使用固定点集 + 混合精度，返回总损失和各分量字典"""
        pts = self.fixed_points
        t_int, x_int, y_int = pts['int']
        t_bot, x_bot, yb = pts['bot']
        t_top, x_top, yt = pts['top']
        t_bc_b, x_bc_b, y_bc_b = pts['bc_b']
        t_bc_t, x_bc_t, y_bc_t = pts['bc_t']
        t_ic, x_ic, y_ic = pts['ic']
        t_mb, x_mb, yb_m = pts['mb']
        t_mt, x_mt, yt_m = pts['mt']
        t_wb, x_wb, yb_w = pts['wb']

        # 开启自动微分
        for tensor in [t_int, x_int, y_int, t_bot, x_bot, yb, t_top, x_top, yt,
                       t_bc_b, x_bc_b, y_bc_b, t_bc_t, x_bc_t, y_bc_t,
                       t_ic, x_ic, y_ic, t_mb, x_mb, yb_m, t_mt, x_mt, yt_m,
                       t_wb, x_wb, yb_w]:
            tensor.requires_grad_(True)

        # 混合精度前向（使用新 API）
        with torch.amp.autocast('cuda', enabled=True):
            # 计算所有点的 eta
            t_all = torch.cat([t_int, t_bot, t_top, t_bc_b, t_bc_t, t_ic, t_mb, t_mt, t_wb], dim=0)
            x_all = torch.cat([x_int, x_bot, x_top, x_bc_b, x_bc_t, x_ic, x_mb, x_mt, x_wb], dim=0)
            eta_all = self.eta_net(t_all, x_all)

            # 切分 eta
            n_int = t_int.shape[0]
            n_bot = t_bot.shape[0]
            n_top = t_top.shape[0]
            n_bc_b = t_bc_b.shape[0]
            n_bc_t = t_bc_t.shape[0]
            n_ic = t_ic.shape[0]
            n_mb = t_mb.shape[0]
            n_mt = t_mt.shape[0]
            n_wb = t_wb.shape[0]

            idx = 0
            eta_int = eta_all[idx:idx+n_int]; idx += n_int
            eta_bot = eta_all[idx:idx+n_bot]; idx += n_bot
            eta_top = eta_all[idx:idx+n_top]; idx += n_top
            eta_bc_b = eta_all[idx:idx+n_bc_b]; idx += n_bc_b
            eta_bc_t = eta_all[idx:idx+n_bc_t]; idx += n_bc_t
            eta_ic = eta_all[idx:idx+n_ic]; idx += n_ic
            eta_mb = eta_all[idx:idx+n_mb]; idx += n_mb
            eta_mt = eta_all[idx:idx+n_mt]; idx += n_mt
            eta_wb = eta_all[idx:idx+n_wb]; idx += n_wb

            # 内区网络前向
            u_int, w_int, p_int = self.inner_net(t_int, x_int, y_int)
            u_bc_b, w_bc_b, p_bc_b = self.inner_net(t_bc_b, x_bc_b, y_bc_b)
            u_bc_t, w_bc_t, p_bc_t = self.inner_net(t_bc_t, x_bc_t, y_bc_t)
            u_ic, w_ic, p_ic = self.inner_net(t_ic, x_ic, y_ic)

            # 边界层网络前向
            u_tilde_bot, w_tilde_bot, p_tilde_bot = self.bottom_net(t_bot, x_bot, yb)
            u_tilde_mb, w_tilde_mb, p_tilde_mb = self.bottom_net(t_mb, x_mb, yb_m)
            u_tilde_wb, w_tilde_wb, p_tilde_wb = self.bottom_net(t_wb, x_wb, yb_w)

            u_hat_top, w_hat_top, p_hat_top = self.top_net(t_top, x_top, yt)
            u_hat_mt, w_hat_mt, p_hat_mt = self.top_net(t_mt, x_mt, yt_m)

            # 计算损失（调用 losses.py）
            loss_pde_inner = losses.loss_pde_inner(t_int, x_int, y_int, u_int, w_int, p_int, eta_int, self.cfg)
            loss_bottom_x0 = losses.loss_bottom_xmom_0(t_bot, x_bot, yb, p_tilde_bot, eta_bot, self.cfg)
            loss_bottom_cont = losses.loss_bottom_cont(t_bot, x_bot, yb, w_tilde_bot, eta_bot, self.cfg)
            loss_top_x0 = losses.loss_top_xmom_0(t_top, x_top, yt, p_hat_top, eta_top, self.cfg)
            loss_top_cont = losses.loss_top_cont(t_top, x_top, yt, w_hat_top, eta_top, self.cfg)
            loss_bottom_match = losses.loss_bottom_match(t_mb, x_mb, yb_m, u_tilde_mb, w_tilde_mb, p_tilde_mb)
            loss_top_match = losses.loss_top_match(t_mt, x_mt, yt_m, u_hat_mt, w_hat_mt, p_hat_mt)
            loss_bc_wall = losses.loss_bc_wall(t_bc_b, x_bc_b, y_bc_b, u_bc_b, w_bc_b)
            loss_bc_interface_kin = losses.loss_bc_interface_kinematic(t_bc_t, x_bc_t, y_bc_t, u_bc_t, w_bc_t, eta_bc_t)
            loss_bc_interface_press = losses.loss_bc_interface_pressure(t_bc_t, x_bc_t, y_bc_t, p_bc_t, eta_bc_t, self.cfg)
            loss_bc_interface_shear = losses.loss_bc_interface_shear(t_bc_t, x_bc_t, y_bc_t, u_bc_t, eta_bc_t)
            loss_ic_eta = losses.loss_ic_eta(t_ic, x_ic, eta_ic, self.initial_eta_func)
            loss_ic_vel = losses.loss_ic_velocity(t_ic, x_ic, y_ic, u_ic, w_ic)
            loss_ic_p = losses.loss_ic_pressure(t_ic, x_ic, y_ic, p_ic)

            total_loss = (
                self.cfg.LAMBDA['pde_inner'] * loss_pde_inner +
                self.cfg.LAMBDA['bottom'] * (loss_bottom_x0 + loss_bottom_cont) +
                self.cfg.LAMBDA['top'] * (loss_top_x0 + loss_top_cont) +
                self.cfg.LAMBDA['match'] * (loss_bottom_match + loss_top_match) +
                self.cfg.LAMBDA['bc'] * (loss_bc_wall + loss_bc_interface_kin + loss_bc_interface_press + loss_bc_interface_shear) +
                self.cfg.LAMBDA['ic'] * (loss_ic_eta + loss_ic_vel + loss_ic_p)
            )

        # 混合精度反向传播
        self.optimizer.zero_grad()
        self.scaler.scale(total_loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # 收集所有损失分量
        loss_dict = {
            'pde_inner': loss_pde_inner.item(),
            'bottom_x0': loss_bottom_x0.item(),
            'bottom_cont': loss_bottom_cont.item(),
            'top_x0': loss_top_x0.item(),
            'top_cont': loss_top_cont.item(),
            'bottom_match': loss_bottom_match.item(),
            'top_match': loss_top_match.item(),
            'bc_wall': loss_bc_wall.item(),
            'bc_kin': loss_bc_interface_kin.item(),
            'bc_press': loss_bc_interface_press.item(),
            'bc_shear': loss_bc_interface_shear.item(),
            'ic_eta': loss_ic_eta.item(),
            'ic_vel': loss_ic_vel.item(),
            'ic_p': loss_ic_p.item(),
        }

        return total_loss.item(), loss_dict

    def train(self, initial_eta_func=None):
        """完整训练循环，每个 epoch 重新采样"""
        self.initial_eta_func = initial_eta_func if initial_eta_func is not None else self.default_initial_eta
        for epoch in range(self.cfg.EPOCHS):
            self.generate_fixed_points()
            loss, loss_dict = self.train_step()
            self.loss_history.append(loss)
            if epoch % 100 == 0:
                # 打印总损失和关键分量
                print(f"Epoch {epoch}, Loss: {loss:.3e}, "
                      f"PDE: {loss_dict['pde_inner']:.2e}, "
                      f"BC_wall: {loss_dict['bc_wall']:.2e}, "
                      f"BC_kin: {loss_dict['bc_kin']:.2e}, "
                      f"BC_press: {loss_dict['bc_press']:.2e}, "
                      f"IC_eta: {loss_dict['ic_eta']:.2e}, "
                      f"IC_vel: {loss_dict['ic_vel']:.2e}")
                self.scheduler.step(loss)
        print("Training completed.")

    def default_initial_eta(self, x):
        """默认初始界面高度函数：一个小扰动"""

        return 0.2 * torch.cos(torch.pi * x / (2 * self.cfg.L)) + 0.01

    def save_models(self, path):
        torch.save({
            'inner': self.inner_net.state_dict(),
            'bottom': self.bottom_net.state_dict(),
            'top': self.top_net.state_dict(),
            'eta': self.eta_net.state_dict()
        }, path)

    def load_models(self, path):
        checkpoint = torch.load(path, map_location=self.device)
        self.inner_net.load_state_dict(checkpoint['inner'])
        self.bottom_net.load_state_dict(checkpoint['bottom'])
        self.top_net.load_state_dict(checkpoint['top'])
        self.eta_net.load_state_dict(checkpoint['eta'])