import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    # 这个函数的主要目的是构建神经网络的结构
    def __init__(self):
        super(Net, self).__init__()
        # 输入 1 通道图片,图片大小为 32*32， 6通道输出，采用 5*5 卷积核，卷积后图片大小为 28*28
        self.conv1 = nn.Conv2d(1, 6, 5)
        # 输入 6 通道图片，图片大小为 14*14，16通道输出，采用 5*5 卷积核，卷积后图片大小为 10*10
        self.conv2 = nn.Conv2d(6, 16, 5)
        # 全连接层 图片大小为 5*5，通道数为16，转换为一维向量
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        # 输出种类共有10个类
        self.fc3 = nn.Linear(84, 10)

    # 前馈算法，即网络接到输入后，按照前馈算法的顺序，一步一步得到输出
    def forward(self, x):
        # 用 conv1 卷积层对图片进行卷积，接着用核为2的最大池化层进行池化（降采样）
        x = F.max_pool2d(F.relu(self.conv1(x)), 2)
        x = F.max_pool2d(F.relu(self.conv2(x)), 2)
        x = x.view(-1, self.num_flat_features(x))
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        # 输出层的激活函数爱用 softmax
        x = self.fc3(x)
        return x

    # 这个函数为了获取tensor张量的元素个数
    def num_flat_features(self, x):
        # 在卷积神经网络中，常常采用随机梯度下降更新网络中的元素
        # 随机梯度下降需要随机选取样本中的一批数据（batch）数据来进行计算
        # 以图片为例，输入网络的输入网络通常是一个4维张量
        # 第0维是 batch
        # 第1维是 图片通道channel，灰度图为1，rgb为3
        # 第2，3维是 图片的x,y坐标
        size = x.size()[1:]  # all dimensions except the batch dimension
        num_features = 1
        for s in size:
            num_features *= s
        return num_features


net = Net()
print(net)

# 获取参数
params = list(net.parameters())
print(len(params))
print(params[0].size())


input = torch.randn(1, 1, 32, 32)

out = net(input)
print(out)

net.zero_grad()  # 对所有参数的梯度缓存区进行归零操作
out.backward(torch.randn(1, 10))  # 使用随机的梯度进行反向传播


# 损失函数
output = net(input)
# 1到10的数组
target = torch.arange(1, 11)  # a dummy target, for example
target = target.view(1, -1)  # make it the same shape as output

criterion = nn.MSELoss()

# mean-squared error 均方误差
loss = criterion(output, target)
print(loss)

"""
前馈计算数据流
input -> conv2d -> relu -> maxpool2d -> conv2d -> relu -> maxpool2d
      -> view -> linear -> relu -> linear -> relu -> linear
      -> MSELoss
      -> loss
"""

# So, when we call loss.backward(), the whole graph is differentiated w.r.t.
# the loss, and all Tensors in the graph that has requres_grad=True will
# have their .grad Tensor accumulated with the gradient.

print(loss.grad_fn)  # MSELoss
print(loss.grad_fn.next_functions[0][0])  # Linear
print(loss.grad_fn.next_functions[0][0].next_functions[0][0])  # ReLU

# 清空参数的梯度缓存，这是每一次bp前应做的
net.zero_grad()     # zeroes the gradient buffers of all parameters

print('conv1.bias.grad before backward')
print(net.conv1.bias.grad)

loss.backward()

print('conv1.bias.grad after backward')
print(net.conv1.bias.grad)

# 更新参数
# weight = weight - learning_rate * gradient

learning_rate = 0.01
for f in net.parameters():
    f.data.sub_(f.grad.data * learning_rate)

# 事实上，更新参数的方法pytorch也为我们提供了
import torch.optim as optim

# create your optimizer
optimizer = optim.SGD(net.parameters(), lr=0.01)  # lr learning rate 学习速率

# in your training loop:
optimizer.zero_grad()   # zero the gradient buffers
output = net(input)
loss = criterion(output, target)
loss.backward()
# 这一句就是更新的语句
optimizer.step()    # Does the update









