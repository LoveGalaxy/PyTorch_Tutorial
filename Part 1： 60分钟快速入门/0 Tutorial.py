import torch

# python version is 3.6.4
# torch version is 0.4.0
# print(torch.__version__)

# create a 5*3 matrix, uninitialized
# 创建一个没有初始化的 5*3 的矩阵
x = torch.empty(5, 3)
print(x)

# create a 5*3 randomly initialized matrix
# 创建一个随机初始化的 5*3 矩阵
x = torch.rand(5, 3)
print(x)

# create a matrix filled zeros and of dtype long
# 创建一个由 0 组成的，数据类型为 long 的矩阵
x = torch.zeros(5, 3, dtype=torch.long)
print(x)

# construct a tensor directly from data
# 直接从数据中创建 tensor 对象
x = torch.tensor([5.5, 3])
print(x)

# create a tensor basing on existing tensor
# these methods will reuse properties of the input tensor
# 首先创建一个全为1的矩阵
# 然后创建一个和原矩阵一样大,由随机数生成的矩阵
x = x.new_ones(5, 3, dtype=torch.double)
print(x)
x = torch.randn_like(x, dtype=torch.float)
print(x)

# get its size
(i, j) = x.size()  # torch.Size is in fact a tuple, so it supports all tuple operations.
print(i, j)
print(x.shape)  # .shape == .size()

# There are multiple syntaxes for operations
# In the following example, we will take a look at the addition operation
# 接下来，以加法为例介绍tensor类的运算操作


# Addition: syntax 1
# 加法： 第一种形式
y = torch.rand(5, 3)
print(x + y)

# Addition: syntax 2
# 加法： 第二种形式
print(torch.add(x, y))

# Addition: providing an output tensor as argument
# 加法： 参数返回结果
result = torch.empty(5, 3)
torch.add(x, y, out=result)
print(result)

# Addtion in-place
# 加法：原地加法
# Any operation that mutates a tensor in-place is post-fixed with an _.
# For example: x.copy_(y), x.t_(), will change x
y.add_(x)
print(y)

# You can use standard NumPy-like indexing with all bells and whistles!
# 可以像 numpy 一样索引
print(x[:, 1])

# Resizing: If you want to resize/reshape tensor, you can use torch.view
# 可以是用torch.view方法，重建tensor对象的维度
x = torch.randn(4, 4, 3)
y = x.view(3, 16)
z = x.view(-1, 8)  # the size -1 is inferred from other dimensions
print(x.size(), y.size(), z.size())
# 这个方法是顺序整理
# 例如上面代码中x生成了一个 4 维 3*3 的矩阵
# 转换为 3 维，长度为 16 的向量时，都是顺序排列
x = torch.randn(4, 4, 3)
y = x.view(3, 4, 4)
# 再考虑一下这个代码，我们把 3*4*4 的张量看做一个长方体
# 事实上，通过 view 的变换的到得长方体并不是旋转过后的长方体
# 而是按照顺序重排列之后的长方体

# If you have a one element tensor, use .item() to get the value as a Python number
# 如果你的 tensor 张量只有一个元素，使用 .item() 获取里面的值
x = torch.randn(1)
print(x)
print(x.item())

# The Torch Tensor and NumPy array will share their underlying memory locations,
# and changing one will change the other.
# torch tensor 可以和 numpy array 直接转换，但他们会共享内存，
# 这意味着如果你改变了其中之一，另一个也会随之改变
a = torch.ones(5)
print(a)
b = a.numpy()
print(b)
# 改变了a中元素值，b中的值也随之改变
a.add_(1)
print(a)
print(b)

# changing the np array changed the Torch Tensor automatically
# numpy 转 tensor， 接着改变 numpy array， torch tensor 随之改变
# 注意：不在文件头部 import 模块是不符合 PEP 8 规范的
import numpy as np
a = np.ones(5)
b = torch.from_numpy(a)
np.add(a, 1, out=a)
print(a)
print(b)

# All the Tensors on the CPU except a CharTensor support converting to NumPy and back.
# 除了 CharTensor 外，别的 tensor 都支持与 numpy array 进行转换

# Tensors can be moved onto any device using the .to method.
# 用 .to 方法更换计算设备

# let us run this cell only if CUDA is available
# We will use ``torch.device`` objects to move tensors in and out of GPU
if torch.cuda.is_available():
    device = torch.device("cuda")          # a CUDA device object
    y = torch.ones_like(x, device=device)  # directly create a tensor on GPU
    x = x.to(device)                       # or just use strings ``.to("cuda")``
    z = x + y
    print(z)
    print(z.to("cpu", torch.double))       # ``.to`` can also change dtype together!

# 有的电脑有CUDA，有的没有，为了使代码有更好的移植性，我们可以额这么来写
device = "cuda" if torch.cuda.is_available() else "cpu"
x = torch.rand(5, 3, device=device)  # rand 0, 1均匀分布
y = torch.randn(5, 3, device=device)  # randn 均值0，方差1的正态分布
z = x + y
print(z)

