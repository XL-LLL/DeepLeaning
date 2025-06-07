import torch

# 测试GPU环境是否可使用
print(torch.__version__,torch.version.cuda,torch.cuda.is_available())  # pytorch版本 cuda版本 查看cuda是否可用


# 使用GPU or CPU
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.device_count(),torch.cuda.current_device())

# 查询张量所在设备
X = torch.tensor([1,2,3])
print(X.device) # 默认在CPU内存上
# 存储在GPU上
X = torch.ones(2,3,device=device)
print(X.device)
