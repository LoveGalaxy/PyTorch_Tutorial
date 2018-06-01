import torch

# 把 requires_grad 设为 true，可以用来自动求导
x = torch.ones(2, 2, requires_grad=True)
print(x)

# 对 x 做一些运算
y = x + 2
print(y)
print(y.grad_fn)

z = y * y * 3
out = z.mean()
print(z, out)

# 更改 requires_grad 属性
a = torch.randn(2, 2)
a = ((a * 3) / (a - 1))
print(a.requires_grad)
a.requires_grad_(True)
print(a.requires_grad)
b = (a * a).sum()
print(b.grad_fn)

out.backward()
print(x.grad)

x = torch.randn(3, requires_grad=True)

y = x * 2
while y.data.norm() < 1000:  # .norm() 平方根
    y = y * 2
print(x)
print(y)

# 梯度的尺度
gradients = torch.tensor([0.1, 1.0, 0.0001], dtype=torch.float)
y.backward(gradients)
print(x.grad)

# 修改 .requires_grad 属性
print(x.requires_grad)
print((x ** 2).requires_grad)

with torch.no_grad():
    print((x ** 2).requires_grad)





