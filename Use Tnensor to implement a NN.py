import torch

# numpy已经为我们提供了强大的计算能力，但是为了能让我们的数据
# 能够在GPU上进行加速，我们需要将计算的类型转化为Tensor
# 设置torch tensor类型，设置计算设备
dtype = torch.float
device = torch.device("cpu")

N = 64
D_in = 1000
H = 100
D_out = 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype)
w2 = torch.randn(H, D_out, device=device, dtype=dtype)

lr = 1e-6

for t in range(500):
    # 前馈计算
    h = x.mm(w1)
    h_relu = h.clamp(min=0)
    y_pred = h_relu.mm(w2)

    # 计算损失
    loss = (y_pred - y).pow(2).sum().item()
    print(t, loss)

    # bp算法
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.t().mm(grad_y_pred)
    grad_h_relu = grad_y_pred.mm(w2.t())
    grad_h = grad_h_relu.clone()
    grad_h[h < 0] = 0
    grad_w1 = x.t().mm(grad_h)

    # 更新网络权值
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2
