# main.py
# 主程序入口（优化版：增强可视化，添加调试输出）

import torch
import numpy as np
import random
import matplotlib.pyplot as plt
from config import Config
from train import Trainer

def set_seed(seed=42):
    """设置随机种子以保证可重复性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

def plot_loss_curve(loss_history, save_path='loss_curve.png'):
    """绘制训练损失曲线（对数坐标）"""
    plt.figure(figsize=(8, 5))
    plt.semilogy(loss_history)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('Training Loss History')
    plt.grid(True, which="both", ls="--", alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Loss curve saved to {save_path}")

def plot_interface_evolution(trainer, cfg, save_path='interface_evolution.png'):
    """绘制不同时刻的界面高度 η(x)"""
    x_plot = torch.linspace(-cfg.L, cfg.L, 100, device=trainer.device).reshape(-1, 1)
    times = torch.linspace(0, cfg.T_MAX, 5, device=trainer.device)

    plt.figure(figsize=(8, 5))
    for t_val in times:
        t_plot = torch.ones_like(x_plot) * t_val
        with torch.no_grad():
            eta = trainer.eta_net(t_plot, x_plot).cpu().numpy()
        plt.plot(x_plot.cpu().numpy(), eta, label=f't={t_val.item():.2f}')

    plt.xlabel('x')
    plt.ylabel('η')
    plt.legend()
    plt.title('Interface Evolution')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Interface evolution saved to {save_path}")

def plot_velocity_field(trainer, cfg, t_fixed=0.5, save_path='velocity_field.png'):
    """绘制固定时刻的速度场 u(x,y) 云图"""
    nx, nz = 100, 50
    x = np.linspace(-cfg.L, cfg.L, nx)
    z_norm = np.linspace(0, 1, nz)
    X, Z_norm = np.meshgrid(x, z_norm, indexing='ij')

    X_flat = torch.tensor(X.reshape(-1, 1), dtype=torch.float32, device=trainer.device)
    Z_flat = torch.tensor(Z_norm.reshape(-1, 1), dtype=torch.float32, device=trainer.device)
    t_flat = torch.ones_like(X_flat) * t_fixed

    with torch.no_grad():
        eta = trainer.eta_net(t_flat, X_flat)
        y = Z_flat  # 归一化坐标
        u_in, w_in, _ = trainer.inner_net(t_flat, X_flat, y)
        u = u_in.cpu().numpy().reshape(nx, nz)  # (nx, nz)

    # 打印速度范围以便调试
    print(f"Velocity u range: [{u.min():.3e}, {u.max():.3e}]")

    plt.figure(figsize=(10, 6))
    # 使用对称色阶，自动调整范围
    vmax = max(abs(u.min()), abs(u.max()))
    if vmax < 1e-12:
        vmax = 1e-3  # 防止全零时出错
    cf = plt.contourf(X, Z_norm, u, levels=50, cmap='RdBu_r', vmin=-vmax, vmax=vmax)
    plt.colorbar(cf, label='u')
    plt.xlabel('x')
    plt.ylabel('y (normalized)')
    plt.title(f'Velocity u at t={t_fixed}')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Velocity field saved to {save_path}")

def plot_boundary_layer_profile(trainer, cfg, x_fixed=0.0, t_fixed=0.5, save_path='boundary_layer.png'):
    """绘制固定 x 位置的速度剖面 u(y)"""
    nz = 200
    y = torch.linspace(0, 1, nz, device=trainer.device).reshape(-1, 1)
    x = torch.ones_like(y) * x_fixed
    t = torch.ones_like(y) * t_fixed

    with torch.no_grad():
        eta = trainer.eta_net(t, x)
        u_in, _, _ = trainer.inner_net(t, x, y)
        u = u_in.cpu().numpy().flatten()
        y_np = y.cpu().numpy().flatten()

    print(f"Profile u range: [{u.min():.3e}, {u.max():.3e}]")

    plt.figure(figsize=(6, 8))
    plt.plot(u, y_np, 'b-', linewidth=2)
    plt.xlabel('u')
    plt.ylabel('y')
    plt.title(f'Velocity profile at x={x_fixed}, t={t_fixed}')
    plt.grid(True, alpha=0.3)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.show()
    print(f"Boundary layer profile saved to {save_path}")

def main():
    set_seed(42)
    cfg = Config()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 创建训练器并训练
    trainer = Trainer(cfg, device)
    trainer.train()  # 使用默认初始条件

    # 保存模型
    trainer.save_models("trained_models.pth")
    print("Models saved to trained_models.pth")

    # 绘制损失曲线
    plot_loss_curve(trainer.loss_history, save_path='loss_curve.png')

    # 绘制结果图
    plot_interface_evolution(trainer, cfg, save_path='interface_evolution.png')
    plot_velocity_field(trainer, cfg, t_fixed=0.5, save_path='velocity_field.png')
    plot_boundary_layer_profile(trainer, cfg, x_fixed=0.0, t_fixed=0.5, save_path='boundary_layer.png')

if __name__ == "__main__":
    main()