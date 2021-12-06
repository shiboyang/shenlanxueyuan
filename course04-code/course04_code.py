# 第四次作业，通过在优化器定义的部分，每次只使一个优化器起作用，看一下模型的训练结果。
# 关于pytorch框架的讲解会在下一课中展开。


import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

# 如果自己的环境配置了cude，清将下面的标志改成True
USE_CUDA = False

batch_size = 64

# MNIST数据集已经集成在pytorch datasets中，可以直接调用
train_dataset = datasets.MNIST(root='./dataset/mnist/',
                               train=True,
                               transform=transforms.ToTensor(),
                               download=False)

test_dataset = datasets.MNIST(root='./dataset/mnist/',
                              train=False,
                              transform=transforms.ToTensor(),
                              download=False)

# Pytorch 的数据加载
train_loader = torch.utils.data.DataLoader(dataset=train_dataset,
                                           batch_size=batch_size,
                                           shuffle=True)

test_loader = torch.utils.data.DataLoader(dataset=test_dataset,
                                          batch_size=batch_size,
                                          shuffle=False)


## 利用pytorch定义整个网络形式

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        # 输入1通道，输出10通道，kernel 5*5
        self.conv1 = nn.Conv2d(in_channels=1, out_channels=10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, 5)
        self.conv3 = nn.Conv2d(20, 40, 3)

        self.mp = nn.MaxPool2d(2)
        # fully connect
        self.fc = nn.Linear(40, 10)  # （in_features, out_features）

    def forward(self, x):
        # in_size = 64
        # one batch 此时的x是包含batchsize维度为4的tensor，即(batchsize，channels，x，y)
        # x.size(0)指batchsize的值    把batchsize的值作为网络的in_size
        in_size = x.size(0)
        # x: 64*1*28*28
        x = F.relu(self.mp(self.conv1(x)))
        # x: 64*10*12*12  (n+2p-f)/s + 1 = 28 - 5 + 1 = 24,所以在没有池化的时候是24*24,池化层为2*2 ，所以池化之后为12*12
        x = F.relu(self.mp(self.conv2(x)))
        # x: 64*20*4*4 同理，没有池化的时候是12 - 5 + 1 = 8 ，池化后为4*4
        x = F.relu(self.mp(self.conv3(x)))
        # 输出x : 64*40*2*2

        x = x.view(in_size, -1)  # 平铺 tensor 相当于resharp
        # print(x.size())
        # x: 64*320
        x = self.fc(x)
        # x:64*10
        # print(x.size())
        return F.log_softmax(x, dim=0)  # 64*10


model = Net()
if USE_CUDA:
    model.cuda()

## 调用不同的优化器算法
# 优化器1 SGD 可实现SGD优化算法，带动量SGD优化算法，带NAG(Nesterov accelerated gradient)
# 动量SGD优化算法,并且均可拥有weight_decay项。
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.5)

# 优化器2 ASGD ASGD也称为SAG，表示随机平均梯度下降(Averaged Stochastic Gradient Descent)，
# 简单地说ASGD就是用空间换时间的一种SGD。
optimizer = optim.ASGD(model.parameters(), lr=0.01, lambd=0.0001, alpha=0.75, t0=1000000.0, weight_decay=0)

# 优化器3 Rprop 实现Rprop优化方法(弹性反向传播)，优化方法原文《Martin Riedmiller und Heinrich
# Braun: Rprop - A Fast Adaptive Learning Algorithm. Proceedings of the International
# Symposium on Computer and Information Science VII, 1992》
# 该优化方法适用于full-batch，不适用于mini-batch，因而在min-batch大行其道的时代里，很少见到。
optimizer = optim.Adagrad(model.parameters(), lr=0.01, lr_decay=0, weight_decay=0, initial_accumulator_value=0)

# 优化器4 Adagrad 实现Adagrad优化方法(Adaptive Gradient)，Adagrad是一种自适应优化方法，是自适应的
# 为各个参数分配不同的学习率。这个学习率的变化，会受到梯度的大小和迭代次数的影响。梯度越大，学习率越小；梯度越小，
# 学习率越大。缺点是训练后期，学习率过小，因为Adagrad累加之前所有的梯度平方作为分母。
optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

# 优化器5 Adadelta 实现Adadelta优化方法。Adadelta是Adagrad的改进。Adadelta分母中采用距离当前时间点
# 比较近的累计项，这可以避免在训练后期，学习率过小。
optimizer = optim.Adadelta(model.parameters(), lr=1.0, rho=0.9, eps=1e-06, weight_decay=0)

# 优化器6 RMSprop 实现RMSprop优化方法（Hinton提出），RMS是均方根（root meam square）的意思。
# RMSprop和Adadelta一样，也是对Adagrad的一种改进。RMSprop采用均方根作为分母，可缓解Adagrad学习率
# 下降较快的问题。并且引入均方根，可以减少摆动。
optimizer = optim.RMSprop(model.parameters(), lr=0.01, alpha=0.99, eps=1e-08, weight_decay=0, momentum=0,
                          centered=False)

# 优化器7 Adam 是一种自适应学习率的优化方法，Adam利用梯度的一阶矩估计和二阶矩估计动态的调整学习率。
# Adam是结合了Momentum和RMSprop，并进行了偏差修正。
optimizer = optim.Adam(model.parameters(), lr=0.001)

# 优化器8 Adamax 实现Adamax优化方法。Adamax是对Adam增加了一个学习率上限的概念，所以也称之为Adamax。
optimizer = optim.Adamax(model.parameters(), lr=0.002, betas=(0.9, 0.999), eps=1e-08, weight_decay=0)

# 优化器9 SparseAdam 针对稀疏张量的一种“阉割版”Adam优化方法。
# optimizer = optim.SparseAdam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08)

# 优化器10 实现L-BFGS（Limited-memory Broyden–Fletcher–Goldfarb–Shanno）优化方法。L-BFGS属于拟牛顿算法。L-BFGS是对BFGS的改进，特点就是节省内存。
# optimizer = optim.LBFGS(model.parameters(), lr=1, max_iter=20, max_eval=None, tolerance_grad=1e-05,
#                         tolerance_change=1e-09, history_size=100, line_search_fn=None)


def train(epoch):
    for batch_idx, (data, target) in enumerate(train_loader):  # batch_idx是enumerate（）函数自带的索引，从0开始
        # data.size():[64, 1, 28, 28]
        # target.size():[64]
        output = model(data.cude() if USE_CUDA else data)
        # print(batch_idx)
        # output:64*10
        loss = F.nll_loss(output, target.cuda() if USE_CUDA else target)
        # 每200次，输出一次数据
        if batch_idx % 200 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch,
                batch_idx * len(data),
                len(train_loader.dataset),
                100. * batch_idx / len(train_loader),
                loss.item()))

        optimizer.zero_grad()  # 所有参数的梯度清零
        loss.backward()  # 即反向传播求梯度
        optimizer.step()  # 调用optimizer进行梯度下降更新参数


def test():
    test_loss = 0
    correct = 0
    for data, target in test_loader:
        data, target = Variable(data), Variable(target)
        output = model(data.cuda() if USE_CUDA else data)
        # 累加loss
        test_loss += F.nll_loss(output, target.cuda() if USE_CUDA else target, reduction="sum").item()
        # get the index of the max log-probability
        # 找出每列（索引）概率意义下的最大值
        pred = output.data.max(1, keepdim=True)[1]
        # print(pred)
        if USE_CUDA:
            correct += pred.eq(target.data.view_as(pred).cuda()).cuda().sum()
        else:
            correct += pred.eq(target.data.view_as(pred)).sum()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


for epoch in range(1, 30):
    print("Epoch:" + str(epoch))
    train(epoch)
    test()
