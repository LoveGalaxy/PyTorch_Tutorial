# 介绍了如何像 Keras 这样的库一样，对 nn 模型进行抽象

import torch

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in)
y = torch.randn(N, D_out)

# 抽象我们的 NN 模型，类似 Keras
# 生成了一个名字叫做 model 的函数，接受我们的输入向量，
model = torch.nn.Sequential(
    torch.nn.Linear(D_in, H),
    torch.nn.ReLU(),
    torch.nn.Linear(H, D_out),
)

# 生成损失函数
loss_fn = torch.nn.MSELoss(size_average=False)

lr = 1e-6

for t in range(500):
    y_pred = model(x)

    loss = loss_fn(y_pred, y)
    print(t, loss.item())

    model.zero_grad()

    loss.backward()

    with torch.no_grad():
        for param in model.parameters():
            param -= lr * param.grad


