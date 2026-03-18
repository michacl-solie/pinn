# data.py
# 数据采样器：生成内区、边界层、边界、初始条件及匹配区域的训练点

import torch
from config import Config

class DataSampler:
    def __init__(self, cfg: Config, device):
        """
        初始化采样器
        :param cfg: 配置对象
        :param device: 计算设备（'cuda' 或 'cpu'）
        """
        self.cfg = cfg
        self.device = device

    def sample_interior(self, n):
        """
        内区采样：返回 (t, x, y)
        y ∈ (0,1) 均匀分布
        """
        t = torch.rand(n, 1, device=self.device) * self.cfg.T_MAX
        x = torch.rand(n, 1, device=self.device) * 2 * self.cfg.L - self.cfg.L
        y = torch.rand(n, 1, device=self.device)  # (0,1)
        return t, x, y

    def sample_bottom_layer(self, n):
        """
        底边界层采样：返回 (t, x, y_b)
        y_b ∈ [0, YB_MAX] 均匀分布
        """
        t = torch.rand(n, 1, device=self.device) * self.cfg.T_MAX
        x = torch.rand(n, 1, device=self.device) * 2 * self.cfg.L - self.cfg.L
        y_b = torch.rand(n, 1, device=self.device) * self.cfg.YB_MAX
        return t, x, y_b

    def sample_top_layer(self, n):
        """
        上边界层采样：返回 (t, x, y_t)
        y_t ∈ [0, YT_MAX] 均匀分布
        """
        t = torch.rand(n, 1, device=self.device) * self.cfg.T_MAX
        x = torch.rand(n, 1, device=self.device) * 2 * self.cfg.L - self.cfg.L
        y_t = torch.rand(n, 1, device=self.device) * self.cfg.YT_MAX
        return t, x, y_t

    def sample_boundary_bottom(self, n):
        """
        底壁边界（内区点）：返回 (t, x, y=0)
        """
        t = torch.rand(n, 1, device=self.device) * self.cfg.T_MAX
        x = torch.rand(n, 1, device=self.device) * 2 * self.cfg.L - self.cfg.L
        y = torch.zeros(n, 1, device=self.device)
        return t, x, y

    def sample_boundary_top(self, n):
        """
        界面边界（内区点）：返回 (t, x, y=1)
        """
        t = torch.rand(n, 1, device=self.device) * self.cfg.T_MAX
        x = torch.rand(n, 1, device=self.device) * 2 * self.cfg.L - self.cfg.L
        y = torch.ones(n, 1, device=self.device)
        return t, x, y

    def sample_initial(self, n):
        """
        初始时刻采样：返回 (t=0, x, y)
        """
        t = torch.zeros(n, 1, device=self.device)
        x = torch.rand(n, 1, device=self.device) * 2 * self.cfg.L - self.cfg.L
        y = torch.rand(n, 1, device=self.device)
        return t, x, y

    def sample_match_bottom(self, n):
        """
        底边界层匹配区域：返回 (t, x, y_b=YB_MAX)
        """
        t = torch.rand(n, 1, device=self.device) * self.cfg.T_MAX
        x = torch.rand(n, 1, device=self.device) * 2 * self.cfg.L - self.cfg.L
        y_b = torch.ones(n, 1, device=self.device) * self.cfg.YB_MAX
        return t, x, y_b

    def sample_match_top(self, n):
        """
        上边界层匹配区域：返回 (t, x, y_t=YT_MAX)
        """
        t = torch.rand(n, 1, device=self.device) * self.cfg.T_MAX
        x = torch.rand(n, 1, device=self.device) * 2 * self.cfg.L - self.cfg.L
        y_t = torch.ones(n, 1, device=self.device) * self.cfg.YT_MAX
        return t, x, y_t

    def sample_wall_bottom(self, n):
        """
        底壁处边界层点：返回 (t, x, y_b=0)
        """
        t = torch.rand(n, 1, device=self.device) * self.cfg.T_MAX
        x = torch.rand(n, 1, device=self.device) * 2 * self.cfg.L - self.cfg.L
        y_b = torch.zeros(n, 1, device=self.device)
        return t, x, y_b