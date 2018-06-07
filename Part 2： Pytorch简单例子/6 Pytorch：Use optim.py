import torch

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out)
)

loss_fn = torch.nn.MSELoss(size_average=False)

lr = 1e-4

# 几种常见的优化器, SGD, Momentum, RMSprop, Adam
"""
SGD 是最普通的优化器, 也可以说没有加速效果, 
而 Momentum 是 SGD 的改良版, 它加入了动量原则. 
后面的 RMSprop 又是 Momentum 的升级版. 
而 Adam 又是 RMSprop 的升级版. 
不过从这个结果中我们看到, Adam 的效果似乎比 RMSprop 要差一点. 
所以说并不是越先进的优化器, 结果越佳. 
我们在自己的试验中可以尝试不同的优化器, 找到那个最适合你数据/网络的优化器.
"""
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


for t in range(500):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    print(t, loss)

    optimizer.zero_grad()

    loss.backward()

    # 使用优化器来更新梯度
    optimizer.step()

