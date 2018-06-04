import numpy as np

# N 是 batch 大小
# D_in, D_out 是神经网络输入和输出的维度
# H 是神经网络隐藏层的维度
N = 64
D_in = 1000
H = 100
D_out = 10

# 生成随机数据
x = np.random.randn(N, D_in)
y = np.random.randn(N, D_out)

# 初始化神经网络参数权值
w1 = np.random.rand(D_in, H)
w2 = np.random.rand(H, D_out)

# 设置学习速率
lr = 1e-6

for t in range(500):
    # 前馈计算
    h = x.dot(w1)
    h_relu = np.maximum(h, 0)
    y_pred = h_relu.dot(w2)

    # 计算误差（loss），采用的是平方和
    loss = np.square(y_pred - y).sum()
    print(t, loss)

    # 反馈计算
    grad_y_pred = 2.0 * (y_pred - y)
    grad_w2 = h_relu.T.dot(grad_y_pred)
    grad_h_relu = grad_y_pred.dot(w2.T)
    grad_h = grad_h_relu.copy()
    grad_h[h < 0] = 0
    grad_w1 = x.T.dot(grad_h)

    # ，更新权值
    w1 -= lr * grad_w1
    w2 -= lr * grad_w2













