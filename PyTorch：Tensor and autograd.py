import torch

dtype = torch.float
device = torch.device("cuda")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

# 将自动求导标记设为True，会有一个grad域保存自动求导得到的数据
w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

lr = 1e-6
for t in range(500):

    # 前馈计算，一步到位
    y_pred = x.mm(w1).clamp(min=0).mm(w2)

    # 损失计算
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # 使用自动求导,自动求导的结果会保存在相应数据的.grad域
    loss.backward()

    with torch.no_grad():
        # 更新参数
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad

        # 每次更新后应该将数据清空，防止自动求导数据失效
        w1.grad.zero_()
        w2.grad.zero_()
