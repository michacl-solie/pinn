import torch
print(torch.__version__)        # 查看 PyTorch 版本
print(torch.cuda.is_available()) # 必须输出 True
print(torch.version.cuda)        # 应该输出 '13.0'