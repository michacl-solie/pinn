# networks.py
# 神经网络定义：内区网络、底边界层网络、上边界层网络、界面高度网络（强制η>0）

import torch
import torch.nn as nn
import torch.nn.functional as F

class InnerNet(nn.Module):
    """
    内区网络：输入 (t, x, y)，输出 (u_in, w_in, p_in)
    """
    def __init__(self, layers):
        """
        :param layers: 列表，如 [3, 64, 64, 64, 64, 3] 表示输入3维，四个隐藏层各64神经元，输出3维
        """
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList()
        for i in range(len(layers)-2):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        self._init_weights()

    def _init_weights(self):
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, t, x, y):
        inputs = torch.cat([t, x, y], dim=1)
        for linear in self.linears:
            inputs = self.activation(linear(inputs))
        out = self.output_layer(inputs)
        u = out[:, 0:1]
        w = out[:, 1:2]
        p = out[:, 2:3]
        return u, w, p


class BottomNet(nn.Module):
    """
    底边界层网络：输入 (t, x, y_b)，输出 (u_tilde, w_tilde, p_tilde)
    输出乘以 exp(-y_b/eps) 以确保匹配条件（当 y_b → ∞ 时衰减为零）
    """
    def __init__(self, layers, eps):
        """
        :param layers: 列表，如 [3, 32, 32, 32, 32, 3]
        :param eps: 小参数 ε
        """
        super().__init__()
        self.eps = eps
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList()
        for i in range(len(layers)-2):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        self._init_weights()

    def _init_weights(self):
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, t, x, y_b):
        inputs = torch.cat([t, x, y_b], dim=1)
        for linear in self.linears:
            inputs = self.activation(linear(inputs))
        out = self.output_layer(inputs)
        decay = torch.exp(-y_b / self.eps)   # 指数衰减因子
        u = out[:, 0:1] * decay
        w = out[:, 1:2] * decay
        p = out[:, 2:3] * decay
        return u, w, p


class TopNet(nn.Module):
    """
    上边界层网络：输入 (t, x, y_t)，输出 (u_hat, w_hat, p_hat)
    输出乘以 exp(-y_t/eps) 以确保匹配条件
    """
    def __init__(self, layers, eps):
        super().__init__()
        self.eps = eps
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList()
        for i in range(len(layers)-2):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        self._init_weights()

    def _init_weights(self):
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, t, x, y_t):
        inputs = torch.cat([t, x, y_t], dim=1)
        for linear in self.linears:
            inputs = self.activation(linear(inputs))
        out = self.output_layer(inputs)
        decay = torch.exp(-y_t / self.eps)
        u = out[:, 0:1] * decay
        w = out[:, 1:2] * decay
        p = out[:, 2:3] * decay
        return u, w, p


class EtaNet(nn.Module):
    """
    界面高度网络：输入 (t, x)，输出 η
    使用 Softplus 激活确保 η > 0
    """
    def __init__(self, layers):
        """
        :param layers: 列表，如 [2, 64, 64, 64, 64, 1]
        """
        super().__init__()
        self.activation = nn.Tanh()
        self.linears = nn.ModuleList()
        for i in range(len(layers)-2):
            self.linears.append(nn.Linear(layers[i], layers[i+1]))
        self.output_layer = nn.Linear(layers[-2], layers[-1])
        self._init_weights()

    def _init_weights(self):
        for m in self.linears:
            nn.init.xavier_normal_(m.weight)
            nn.init.zeros_(m.bias)
        nn.init.xavier_normal_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def forward(self, t, x):
        inputs = torch.cat([t, x], dim=1)
        for linear in self.linears:
            inputs = self.activation(linear(inputs))
        eta = F.softplus(self.output_layer(inputs))  # 强制 η > 0
        return eta