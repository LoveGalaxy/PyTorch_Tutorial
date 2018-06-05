import torch

# 这个例子的主要目的是教会我们如何使用自己定义的激活函数来实现神经网络

class MyReLU(torch.autograd.Function):
    # 我们定义了自己的激活函数

    @staticmethod
    def forward(ctx, input):
        # 实现了激活函数的前馈计算
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # 实现了激活函数的反馈计算
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input


dtype = torch.float
device = torch.device("cpu")

N, D_in, H, D_out = 64, 1000, 100, 10

x = torch.randn(N, D_in, device=device, dtype=dtype)
y = torch.randn(N, D_out, device=device, dtype=dtype)

w1 = torch.randn(D_in, H, device=device, dtype=dtype, requires_grad=True)
w2 = torch.randn(H, D_out, device=device, dtype=dtype, requires_grad=True)

lr = 1e-6
for t in range(500):
    # 生成了我们自己的激活函数
    relu = MyReLU.apply

    # 前馈计算，是用我们自己的激活函数
    y_pred = relu(x.mm(w1)).mm(w2)

    # 计算损失
    loss = (y_pred - y).pow(2).sum()
    print(t, loss.item())

    # 计算梯度
    loss.backward()

    # 更新权值
    with torch.no_grad():
        w1 -= lr * w1.grad
        w2 -= lr * w2.grad

        # 清空梯度
        w1.grad.zero_()
        w2.grad.zero_()
