from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from mnist_dataset import RandomRotate, MNISTDataSet


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.layers = nn.Sequential(nn.Conv2d(1, 10, kernel_size=5),
                                    nn.Conv2d(1, 10, kernel_size=5),
                                    nn.Conv2d(1, 10, kernel_size=5))
        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 20, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
        x = F.relu(F.max_pool2d(self.conv2_drop(self.conv2(x)), 2))
        x = x.view(-1, 320)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)


def train(log_interval, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            test_loss += F.nll_loss(output, target, size_average=False).item()  # sum up batch loss
            pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N',
                        help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.01, metavar='LR',
                        help='learning rate (default: 0.01)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()

    torch.manual_seed(args.seed)

    device = torch.device("cuda" if use_cuda else "cpu")

    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    train_loader = torch.utils.data.DataLoader(
        MNISTDataSet(r"./dataset",
                     train=True,
                     download=False,
                     transform=transforms.Compose([
                         RandomRotate(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))]
                     )
                     ),

        batch_size=args.batch_size, shuffle=True, **kwargs)
    test_loader = torch.utils.data.DataLoader(
        MNISTDataSet(r'./dataset',
                     train=False,
                     transform=transforms.Compose([
                         RandomRotate(),
                         transforms.ToTensor(),
                         transforms.Normalize((0.1307,), (0.3081,))
                     ])),
        batch_size=args.test_batch_size, shuffle=True, **kwargs)
    # train_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('data', train=True, download=True,
    #                    transform=transforms.Compose([
    #                        transforms.ToTensor(),
    #                        transforms.Normalize((0.1307,), (0.3081,))
    #                    ])),
    #     batch_size=args.batch_size, shuffle=True, **kwargs)
    # test_loader = torch.utils.data.DataLoader(
    #     datasets.MNIST('data', train=False, transform=transforms.Compose([
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.1307,), (0.3081,))
    #     ])),
    #     batch_size=args.test_batch_size, shuffle=True, **kwargs)
    model = Net().to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)

    for epoch in range(1, args.epochs + 1):
        train(args.log_interval, model, device, train_loader, optimizer, epoch)
        test(model, device, test_loader)


if __name__ == '__main__':
    main()

"""
/home/sparkai/miniconda3/envs/eecs/bin/python /home/sparkai/work/my-code/shenlanxueyuan/course05-code/minist-main.py
/home/sparkai/miniconda3/envs/eecs/lib/python3.8/site-packages/torch/nn/functional.py:718: UserWarning: Named tensors and all their associated APIs are an experimental feature and subject to change. Please do not use them for anything important until they are released as stable. (Triggered internally at  /pytorch/c10/core/TensorImpl.h:1156.)
  return torch.max_pool2d(input, kernel_size, stride, padding, dilation, ceil_mode)
Train Epoch: 1 [0/60000 (0%)]	Loss: 2.319051
Train Epoch: 1 [640/60000 (1%)]	Loss: 2.294956
Train Epoch: 1 [1280/60000 (2%)]	Loss: 2.283870
Train Epoch: 1 [1920/60000 (3%)]	Loss: 2.307473
Train Epoch: 1 [2560/60000 (4%)]	Loss: 2.250338
Train Epoch: 1 [3200/60000 (5%)]	Loss: 2.251483
Train Epoch: 1 [3840/60000 (6%)]	Loss: 2.296974
Train Epoch: 1 [4480/60000 (7%)]	Loss: 2.286903
Train Epoch: 1 [5120/60000 (9%)]	Loss: 2.230149
Train Epoch: 1 [5760/60000 (10%)]	Loss: 2.245445
Train Epoch: 1 [6400/60000 (11%)]	Loss: 2.245901
Train Epoch: 1 [7040/60000 (12%)]	Loss: 2.238893
Train Epoch: 1 [7680/60000 (13%)]	Loss: 2.255692
Train Epoch: 1 [8320/60000 (14%)]	Loss: 2.145864
Train Epoch: 1 [8960/60000 (15%)]	Loss: 2.174796
Train Epoch: 1 [9600/60000 (16%)]	Loss: 2.132703
Train Epoch: 1 [10240/60000 (17%)]	Loss: 2.146455
Train Epoch: 1 [10880/60000 (18%)]	Loss: 2.163989
Train Epoch: 1 [11520/60000 (19%)]	Loss: 2.068462
Train Epoch: 1 [12160/60000 (20%)]	Loss: 2.074987
Train Epoch: 1 [12800/60000 (21%)]	Loss: 2.144532
Train Epoch: 1 [13440/60000 (22%)]	Loss: 1.946129
Train Epoch: 1 [14080/60000 (23%)]	Loss: 1.832413
Train Epoch: 1 [14720/60000 (25%)]	Loss: 1.868353
Train Epoch: 1 [15360/60000 (26%)]	Loss: 1.964970
Train Epoch: 1 [16000/60000 (27%)]	Loss: 1.764685
Train Epoch: 1 [16640/60000 (28%)]	Loss: 1.961425
Train Epoch: 1 [17280/60000 (29%)]	Loss: 1.868137
Train Epoch: 1 [17920/60000 (30%)]	Loss: 1.867054
Train Epoch: 1 [18560/60000 (31%)]	Loss: 2.018369
Train Epoch: 1 [19200/60000 (32%)]	Loss: 1.880058
Train Epoch: 1 [19840/60000 (33%)]	Loss: 1.830642
Train Epoch: 1 [20480/60000 (34%)]	Loss: 1.822834
Train Epoch: 1 [21120/60000 (35%)]	Loss: 1.637251
Train Epoch: 1 [21760/60000 (36%)]	Loss: 1.812424
Train Epoch: 1 [22400/60000 (37%)]	Loss: 1.939192
Train Epoch: 1 [23040/60000 (38%)]	Loss: 1.783711
Train Epoch: 1 [23680/60000 (39%)]	Loss: 1.622438
Train Epoch: 1 [24320/60000 (41%)]	Loss: 1.502464
Train Epoch: 1 [24960/60000 (42%)]	Loss: 1.882413
Train Epoch: 1 [25600/60000 (43%)]	Loss: 1.557498
Train Epoch: 1 [26240/60000 (44%)]	Loss: 1.724004
Train Epoch: 1 [26880/60000 (45%)]	Loss: 1.695829
Train Epoch: 1 [27520/60000 (46%)]	Loss: 1.709006
Train Epoch: 1 [28160/60000 (47%)]	Loss: 1.672428
Train Epoch: 1 [28800/60000 (48%)]	Loss: 1.730074
Train Epoch: 1 [29440/60000 (49%)]	Loss: 1.782304
Train Epoch: 1 [30080/60000 (50%)]	Loss: 1.719991
Train Epoch: 1 [30720/60000 (51%)]	Loss: 1.420522
Train Epoch: 1 [31360/60000 (52%)]	Loss: 1.768927
Train Epoch: 1 [32000/60000 (53%)]	Loss: 1.600774
Train Epoch: 1 [32640/60000 (54%)]	Loss: 1.611652
Train Epoch: 1 [33280/60000 (55%)]	Loss: 1.784086
Train Epoch: 1 [33920/60000 (57%)]	Loss: 1.465564
Train Epoch: 1 [34560/60000 (58%)]	Loss: 1.310608
Train Epoch: 1 [35200/60000 (59%)]	Loss: 1.602988
Train Epoch: 1 [35840/60000 (60%)]	Loss: 1.554361
Train Epoch: 1 [36480/60000 (61%)]	Loss: 1.614415
Train Epoch: 1 [37120/60000 (62%)]	Loss: 1.544613
Train Epoch: 1 [37760/60000 (63%)]	Loss: 1.407831
Train Epoch: 1 [38400/60000 (64%)]	Loss: 1.640829
Train Epoch: 1 [39040/60000 (65%)]	Loss: 1.348647
Train Epoch: 1 [39680/60000 (66%)]	Loss: 1.537268
Train Epoch: 1 [40320/60000 (67%)]	Loss: 1.336337
Train Epoch: 1 [40960/60000 (68%)]	Loss: 1.641695
Train Epoch: 1 [41600/60000 (69%)]	Loss: 1.575412
Train Epoch: 1 [42240/60000 (70%)]	Loss: 1.490390
Train Epoch: 1 [42880/60000 (71%)]	Loss: 1.368557
Train Epoch: 1 [43520/60000 (72%)]	Loss: 1.347875
Train Epoch: 1 [44160/60000 (74%)]	Loss: 1.416831
Train Epoch: 1 [44800/60000 (75%)]	Loss: 1.605494
Train Epoch: 1 [45440/60000 (76%)]	Loss: 1.499969
Train Epoch: 1 [46080/60000 (77%)]	Loss: 1.417920
Train Epoch: 1 [46720/60000 (78%)]	Loss: 1.417639
Train Epoch: 1 [47360/60000 (79%)]	Loss: 1.274904
Train Epoch: 1 [48000/60000 (80%)]	Loss: 1.411709
Train Epoch: 1 [48640/60000 (81%)]	Loss: 1.475164
Train Epoch: 1 [49280/60000 (82%)]	Loss: 1.443393
Train Epoch: 1 [49920/60000 (83%)]	Loss: 1.251928
Train Epoch: 1 [50560/60000 (84%)]	Loss: 1.371827
Train Epoch: 1 [51200/60000 (85%)]	Loss: 1.320455
Train Epoch: 1 [51840/60000 (86%)]	Loss: 1.212857
Train Epoch: 1 [52480/60000 (87%)]	Loss: 1.404081
Train Epoch: 1 [53120/60000 (88%)]	Loss: 1.198149
Train Epoch: 1 [53760/60000 (90%)]	Loss: 1.356746
Train Epoch: 1 [54400/60000 (91%)]	Loss: 1.149200
Train Epoch: 1 [55040/60000 (92%)]	Loss: 1.415793
Train Epoch: 1 [55680/60000 (93%)]	Loss: 1.181037
Train Epoch: 1 [56320/60000 (94%)]	Loss: 1.462738
Train Epoch: 1 [56960/60000 (95%)]	Loss: 1.381571
Train Epoch: 1 [57600/60000 (96%)]	Loss: 1.272133
Train Epoch: 1 [58240/60000 (97%)]	Loss: 1.219676
Train Epoch: 1 [58880/60000 (98%)]	Loss: 1.349224
Train Epoch: 1 [59520/60000 (99%)]	Loss: 1.143028
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
/home/sparkai/miniconda3/envs/eecs/lib/python3.8/site-packages/torch/nn/_reduction.py:42: UserWarning: size_average and reduce args will be deprecated, please use reduction='sum' instead.
  warnings.warn(warning.format(ret))

Test set: Average loss: 0.9409, Accuracy: 7419/10000 (74%)

[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
Train Epoch: 2 [0/60000 (0%)]	Loss: 1.347339
Train Epoch: 2 [640/60000 (1%)]	Loss: 1.245474
Train Epoch: 2 [1280/60000 (2%)]	Loss: 1.333206
Train Epoch: 2 [1920/60000 (3%)]	Loss: 1.050553
Train Epoch: 2 [2560/60000 (4%)]	Loss: 1.175188
Train Epoch: 2 [3200/60000 (5%)]	Loss: 1.039275
Train Epoch: 2 [3840/60000 (6%)]	Loss: 1.268731
Train Epoch: 2 [4480/60000 (7%)]	Loss: 1.360415
Train Epoch: 2 [5120/60000 (9%)]	Loss: 1.426902
Train Epoch: 2 [5760/60000 (10%)]	Loss: 1.237946
Train Epoch: 2 [6400/60000 (11%)]	Loss: 1.263408
Train Epoch: 2 [7040/60000 (12%)]	Loss: 1.173490
Train Epoch: 2 [7680/60000 (13%)]	Loss: 1.095718
Train Epoch: 2 [8320/60000 (14%)]	Loss: 1.142465
Train Epoch: 2 [8960/60000 (15%)]	Loss: 0.983698
Train Epoch: 2 [9600/60000 (16%)]	Loss: 1.314781
Train Epoch: 2 [10240/60000 (17%)]	Loss: 1.021614
Train Epoch: 2 [10880/60000 (18%)]	Loss: 1.253943
Train Epoch: 2 [11520/60000 (19%)]	Loss: 1.147762
Train Epoch: 2 [12160/60000 (20%)]	Loss: 1.074106
Train Epoch: 2 [12800/60000 (21%)]	Loss: 1.253704
Train Epoch: 2 [13440/60000 (22%)]	Loss: 1.284225
Train Epoch: 2 [14080/60000 (23%)]	Loss: 1.226764
Train Epoch: 2 [14720/60000 (25%)]	Loss: 1.435900
Train Epoch: 2 [15360/60000 (26%)]	Loss: 1.058259
Train Epoch: 2 [16000/60000 (27%)]	Loss: 1.284812
Train Epoch: 2 [16640/60000 (28%)]	Loss: 1.054045
Train Epoch: 2 [17280/60000 (29%)]	Loss: 1.309140
Train Epoch: 2 [17920/60000 (30%)]	Loss: 1.038535
Train Epoch: 2 [18560/60000 (31%)]	Loss: 1.082023
Train Epoch: 2 [19200/60000 (32%)]	Loss: 1.240685
Train Epoch: 2 [19840/60000 (33%)]	Loss: 1.065863
Train Epoch: 2 [20480/60000 (34%)]	Loss: 1.032283
Train Epoch: 2 [21120/60000 (35%)]	Loss: 1.225776
Train Epoch: 2 [21760/60000 (36%)]	Loss: 1.136592
Train Epoch: 2 [22400/60000 (37%)]	Loss: 1.111588
Train Epoch: 2 [23040/60000 (38%)]	Loss: 1.042855
Train Epoch: 2 [23680/60000 (39%)]	Loss: 1.149043
Train Epoch: 2 [24320/60000 (41%)]	Loss: 0.952318
Train Epoch: 2 [24960/60000 (42%)]	Loss: 0.940471
Train Epoch: 2 [25600/60000 (43%)]	Loss: 1.061171
Train Epoch: 2 [26240/60000 (44%)]	Loss: 1.160529
Train Epoch: 2 [26880/60000 (45%)]	Loss: 0.963269
Train Epoch: 2 [27520/60000 (46%)]	Loss: 1.308426
Train Epoch: 2 [28160/60000 (47%)]	Loss: 1.078332
Train Epoch: 2 [28800/60000 (48%)]	Loss: 1.069173
Train Epoch: 2 [29440/60000 (49%)]	Loss: 0.992665
Train Epoch: 2 [30080/60000 (50%)]	Loss: 1.166365
Train Epoch: 2 [30720/60000 (51%)]	Loss: 0.858512
Train Epoch: 2 [31360/60000 (52%)]	Loss: 1.018014
Train Epoch: 2 [32000/60000 (53%)]	Loss: 1.085445
Train Epoch: 2 [32640/60000 (54%)]	Loss: 1.125119
Train Epoch: 2 [33280/60000 (55%)]	Loss: 1.155547
Train Epoch: 2 [33920/60000 (57%)]	Loss: 0.884206
Train Epoch: 2 [34560/60000 (58%)]	Loss: 1.186400
Train Epoch: 2 [35200/60000 (59%)]	Loss: 1.026202
Train Epoch: 2 [35840/60000 (60%)]	Loss: 1.202249
Train Epoch: 2 [36480/60000 (61%)]	Loss: 0.917943
Train Epoch: 2 [37120/60000 (62%)]	Loss: 0.806574
Train Epoch: 2 [37760/60000 (63%)]	Loss: 1.068870
Train Epoch: 2 [38400/60000 (64%)]	Loss: 1.113486
Train Epoch: 2 [39040/60000 (65%)]	Loss: 1.187288
Train Epoch: 2 [39680/60000 (66%)]	Loss: 1.061329
Train Epoch: 2 [40320/60000 (67%)]	Loss: 0.921223
Train Epoch: 2 [40960/60000 (68%)]	Loss: 1.102894
Train Epoch: 2 [41600/60000 (69%)]	Loss: 0.998386
Train Epoch: 2 [42240/60000 (70%)]	Loss: 1.142254
Train Epoch: 2 [42880/60000 (71%)]	Loss: 1.157872
Train Epoch: 2 [43520/60000 (72%)]	Loss: 0.891601
Train Epoch: 2 [44160/60000 (74%)]	Loss: 0.995620
Train Epoch: 2 [44800/60000 (75%)]	Loss: 1.052456
Train Epoch: 2 [45440/60000 (76%)]	Loss: 1.056571
Train Epoch: 2 [46080/60000 (77%)]	Loss: 1.620447
Train Epoch: 2 [46720/60000 (78%)]	Loss: 1.072866
Train Epoch: 2 [47360/60000 (79%)]	Loss: 1.123738
Train Epoch: 2 [48000/60000 (80%)]	Loss: 1.151056
Train Epoch: 2 [48640/60000 (81%)]	Loss: 1.005087
Train Epoch: 2 [49280/60000 (82%)]	Loss: 0.968054
Train Epoch: 2 [49920/60000 (83%)]	Loss: 0.755882
Train Epoch: 2 [50560/60000 (84%)]	Loss: 1.225605
Train Epoch: 2 [51200/60000 (85%)]	Loss: 1.046265
Train Epoch: 2 [51840/60000 (86%)]	Loss: 1.091449
Train Epoch: 2 [52480/60000 (87%)]	Loss: 0.960415
Train Epoch: 2 [53120/60000 (88%)]	Loss: 1.039601
Train Epoch: 2 [53760/60000 (90%)]	Loss: 0.791879
Train Epoch: 2 [54400/60000 (91%)]	Loss: 0.967381
Train Epoch: 2 [55040/60000 (92%)]	Loss: 1.116567
Train Epoch: 2 [55680/60000 (93%)]	Loss: 1.111074
Train Epoch: 2 [56320/60000 (94%)]	Loss: 1.175310
Train Epoch: 2 [56960/60000 (95%)]	Loss: 1.059342
Train Epoch: 2 [57600/60000 (96%)]	Loss: 0.920566
Train Epoch: 2 [58240/60000 (97%)]	Loss: 1.102950
Train Epoch: 2 [58880/60000 (98%)]	Loss: 0.891650
Train Epoch: 2 [59520/60000 (99%)]	Loss: 0.977210
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)

Test set: Average loss: 0.6164, Accuracy: 8345/10000 (83%)

[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
Train Epoch: 3 [0/60000 (0%)]	Loss: 0.942415
Train Epoch: 3 [640/60000 (1%)]	Loss: 0.948007
Train Epoch: 3 [1280/60000 (2%)]	Loss: 0.786885
Train Epoch: 3 [1920/60000 (3%)]	Loss: 1.085619
Train Epoch: 3 [2560/60000 (4%)]	Loss: 1.031746
Train Epoch: 3 [3200/60000 (5%)]	Loss: 1.327538
Train Epoch: 3 [3840/60000 (6%)]	Loss: 1.058733
Train Epoch: 3 [4480/60000 (7%)]	Loss: 0.778266
Train Epoch: 3 [5120/60000 (9%)]	Loss: 1.046429
Train Epoch: 3 [5760/60000 (10%)]	Loss: 0.960345
Train Epoch: 3 [6400/60000 (11%)]	Loss: 0.846575
Train Epoch: 3 [7040/60000 (12%)]	Loss: 1.096655
Train Epoch: 3 [7680/60000 (13%)]	Loss: 1.176782
Train Epoch: 3 [8320/60000 (14%)]	Loss: 1.119906
Train Epoch: 3 [8960/60000 (15%)]	Loss: 1.280651
Train Epoch: 3 [9600/60000 (16%)]	Loss: 1.347054
Train Epoch: 3 [10240/60000 (17%)]	Loss: 0.990020
Train Epoch: 3 [10880/60000 (18%)]	Loss: 0.969627
Train Epoch: 3 [11520/60000 (19%)]	Loss: 0.833654
Train Epoch: 3 [12160/60000 (20%)]	Loss: 0.988412
Train Epoch: 3 [12800/60000 (21%)]	Loss: 0.892335
Train Epoch: 3 [13440/60000 (22%)]	Loss: 1.199856
Train Epoch: 3 [14080/60000 (23%)]	Loss: 1.015060
Train Epoch: 3 [14720/60000 (25%)]	Loss: 1.078964
Train Epoch: 3 [15360/60000 (26%)]	Loss: 0.843887
Train Epoch: 3 [16000/60000 (27%)]	Loss: 0.929781
Train Epoch: 3 [16640/60000 (28%)]	Loss: 0.736974
Train Epoch: 3 [17280/60000 (29%)]	Loss: 1.096549
Train Epoch: 3 [17920/60000 (30%)]	Loss: 1.108946
Train Epoch: 3 [18560/60000 (31%)]	Loss: 0.883127
Train Epoch: 3 [19200/60000 (32%)]	Loss: 1.102761
Train Epoch: 3 [19840/60000 (33%)]	Loss: 0.860683
Train Epoch: 3 [20480/60000 (34%)]	Loss: 0.786329
Train Epoch: 3 [21120/60000 (35%)]	Loss: 0.961559
Train Epoch: 3 [21760/60000 (36%)]	Loss: 0.842642
Train Epoch: 3 [22400/60000 (37%)]	Loss: 1.103404
Train Epoch: 3 [23040/60000 (38%)]	Loss: 0.935696
Train Epoch: 3 [23680/60000 (39%)]	Loss: 0.879233
Train Epoch: 3 [24320/60000 (41%)]	Loss: 0.981904
Train Epoch: 3 [24960/60000 (42%)]	Loss: 1.085803
Train Epoch: 3 [25600/60000 (43%)]	Loss: 1.121705
Train Epoch: 3 [26240/60000 (44%)]	Loss: 0.903112
Train Epoch: 3 [26880/60000 (45%)]	Loss: 0.813022
Train Epoch: 3 [27520/60000 (46%)]	Loss: 0.998284
Train Epoch: 3 [28160/60000 (47%)]	Loss: 0.984988
Train Epoch: 3 [28800/60000 (48%)]	Loss: 1.053185
Train Epoch: 3 [29440/60000 (49%)]	Loss: 0.911030
Train Epoch: 3 [30080/60000 (50%)]	Loss: 1.072967
Train Epoch: 3 [30720/60000 (51%)]	Loss: 0.747019
Train Epoch: 3 [31360/60000 (52%)]	Loss: 0.878786
Train Epoch: 3 [32000/60000 (53%)]	Loss: 0.865337
Train Epoch: 3 [32640/60000 (54%)]	Loss: 0.589570
Train Epoch: 3 [33280/60000 (55%)]	Loss: 0.921980
Train Epoch: 3 [33920/60000 (57%)]	Loss: 1.124358
Train Epoch: 3 [34560/60000 (58%)]	Loss: 1.034551
Train Epoch: 3 [35200/60000 (59%)]	Loss: 1.035962
Train Epoch: 3 [35840/60000 (60%)]	Loss: 0.779502
Train Epoch: 3 [36480/60000 (61%)]	Loss: 1.028453
Train Epoch: 3 [37120/60000 (62%)]	Loss: 0.729331
Train Epoch: 3 [37760/60000 (63%)]	Loss: 1.163133
Train Epoch: 3 [38400/60000 (64%)]	Loss: 0.954292
Train Epoch: 3 [39040/60000 (65%)]	Loss: 0.997234
Train Epoch: 3 [39680/60000 (66%)]	Loss: 0.817958
Train Epoch: 3 [40320/60000 (67%)]	Loss: 0.924432
Train Epoch: 3 [40960/60000 (68%)]	Loss: 1.040023
Train Epoch: 3 [41600/60000 (69%)]	Loss: 1.008388
Train Epoch: 3 [42240/60000 (70%)]	Loss: 0.876850
Train Epoch: 3 [42880/60000 (71%)]	Loss: 0.967442
Train Epoch: 3 [43520/60000 (72%)]	Loss: 0.885733
Train Epoch: 3 [44160/60000 (74%)]	Loss: 0.804714
Train Epoch: 3 [44800/60000 (75%)]	Loss: 0.929316
Train Epoch: 3 [45440/60000 (76%)]	Loss: 1.008516
Train Epoch: 3 [46080/60000 (77%)]	Loss: 1.004932
Train Epoch: 3 [46720/60000 (78%)]	Loss: 0.767547
Train Epoch: 3 [47360/60000 (79%)]	Loss: 0.823151
Train Epoch: 3 [48000/60000 (80%)]	Loss: 1.275366
Train Epoch: 3 [48640/60000 (81%)]	Loss: 1.041649
Train Epoch: 3 [49280/60000 (82%)]	Loss: 0.935526
Train Epoch: 3 [49920/60000 (83%)]	Loss: 0.752468
Train Epoch: 3 [50560/60000 (84%)]	Loss: 0.618689
Train Epoch: 3 [51200/60000 (85%)]	Loss: 0.727527
Train Epoch: 3 [51840/60000 (86%)]	Loss: 0.872009
Train Epoch: 3 [52480/60000 (87%)]	Loss: 0.471255
Train Epoch: 3 [53120/60000 (88%)]	Loss: 0.935802
Train Epoch: 3 [53760/60000 (90%)]	Loss: 1.149332
Train Epoch: 3 [54400/60000 (91%)]	Loss: 0.752364
Train Epoch: 3 [55040/60000 (92%)]	Loss: 0.910417
Train Epoch: 3 [55680/60000 (93%)]	Loss: 0.768897
Train Epoch: 3 [56320/60000 (94%)]	Loss: 0.754893
Train Epoch: 3 [56960/60000 (95%)]	Loss: 0.882852
Train Epoch: 3 [57600/60000 (96%)]	Loss: 0.818495
Train Epoch: 3 [58240/60000 (97%)]	Loss: 0.943780
Train Epoch: 3 [58880/60000 (98%)]	Loss: 0.726719
Train Epoch: 3 [59520/60000 (99%)]	Loss: 0.884079
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)

Test set: Average loss: 0.4649, Accuracy: 8727/10000 (87%)

[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
Train Epoch: 4 [0/60000 (0%)]	Loss: 0.768669
Train Epoch: 4 [640/60000 (1%)]	Loss: 0.977168
Train Epoch: 4 [1280/60000 (2%)]	Loss: 0.723890
Train Epoch: 4 [1920/60000 (3%)]	Loss: 0.916480
Train Epoch: 4 [2560/60000 (4%)]	Loss: 0.877842
Train Epoch: 4 [3200/60000 (5%)]	Loss: 0.867066
Train Epoch: 4 [3840/60000 (6%)]	Loss: 1.194640
Train Epoch: 4 [4480/60000 (7%)]	Loss: 0.911039
Train Epoch: 4 [5120/60000 (9%)]	Loss: 1.120555
Train Epoch: 4 [5760/60000 (10%)]	Loss: 0.635624
Train Epoch: 4 [6400/60000 (11%)]	Loss: 0.897878
Train Epoch: 4 [7040/60000 (12%)]	Loss: 0.833619
Train Epoch: 4 [7680/60000 (13%)]	Loss: 0.940708
Train Epoch: 4 [8320/60000 (14%)]	Loss: 0.735665
Train Epoch: 4 [8960/60000 (15%)]	Loss: 1.012219
Train Epoch: 4 [9600/60000 (16%)]	Loss: 1.091058
Train Epoch: 4 [10240/60000 (17%)]	Loss: 0.940015
Train Epoch: 4 [10880/60000 (18%)]	Loss: 0.907288
Train Epoch: 4 [11520/60000 (19%)]	Loss: 0.771289
Train Epoch: 4 [12160/60000 (20%)]	Loss: 0.939249
Train Epoch: 4 [12800/60000 (21%)]	Loss: 0.993538
Train Epoch: 4 [13440/60000 (22%)]	Loss: 0.943545
Train Epoch: 4 [14080/60000 (23%)]	Loss: 0.936737
Train Epoch: 4 [14720/60000 (25%)]	Loss: 1.009208
Train Epoch: 4 [15360/60000 (26%)]	Loss: 0.560013
Train Epoch: 4 [16000/60000 (27%)]	Loss: 0.729850
Train Epoch: 4 [16640/60000 (28%)]	Loss: 0.610075
Train Epoch: 4 [17280/60000 (29%)]	Loss: 0.788355
Train Epoch: 4 [17920/60000 (30%)]	Loss: 0.966152
Train Epoch: 4 [18560/60000 (31%)]	Loss: 0.495603
Train Epoch: 4 [19200/60000 (32%)]	Loss: 0.671616
Train Epoch: 4 [19840/60000 (33%)]	Loss: 1.073597
Train Epoch: 4 [20480/60000 (34%)]	Loss: 0.797264
Train Epoch: 4 [21120/60000 (35%)]	Loss: 0.718066
Train Epoch: 4 [21760/60000 (36%)]	Loss: 0.830149
Train Epoch: 4 [22400/60000 (37%)]	Loss: 0.913436
Train Epoch: 4 [23040/60000 (38%)]	Loss: 1.024973
Train Epoch: 4 [23680/60000 (39%)]	Loss: 0.732845
Train Epoch: 4 [24320/60000 (41%)]	Loss: 0.616291
Train Epoch: 4 [24960/60000 (42%)]	Loss: 0.952666
Train Epoch: 4 [25600/60000 (43%)]	Loss: 0.690597
Train Epoch: 4 [26240/60000 (44%)]	Loss: 0.768566
Train Epoch: 4 [26880/60000 (45%)]	Loss: 1.033514
Train Epoch: 4 [27520/60000 (46%)]	Loss: 0.780827
Train Epoch: 4 [28160/60000 (47%)]	Loss: 0.770235
Train Epoch: 4 [28800/60000 (48%)]	Loss: 0.730677
Train Epoch: 4 [29440/60000 (49%)]	Loss: 0.996368
Train Epoch: 4 [30080/60000 (50%)]	Loss: 0.859434
Train Epoch: 4 [30720/60000 (51%)]	Loss: 0.925733
Train Epoch: 4 [31360/60000 (52%)]	Loss: 0.586616
Train Epoch: 4 [32000/60000 (53%)]	Loss: 0.773134
Train Epoch: 4 [32640/60000 (54%)]	Loss: 0.850603
Train Epoch: 4 [33280/60000 (55%)]	Loss: 0.685862
Train Epoch: 4 [33920/60000 (57%)]	Loss: 0.944447
Train Epoch: 4 [34560/60000 (58%)]	Loss: 0.678485
Train Epoch: 4 [35200/60000 (59%)]	Loss: 0.852381
Train Epoch: 4 [35840/60000 (60%)]	Loss: 0.672467
Train Epoch: 4 [36480/60000 (61%)]	Loss: 0.838241
Train Epoch: 4 [37120/60000 (62%)]	Loss: 0.716843
Train Epoch: 4 [37760/60000 (63%)]	Loss: 1.063545
Train Epoch: 4 [38400/60000 (64%)]	Loss: 0.760309
Train Epoch: 4 [39040/60000 (65%)]	Loss: 0.756472
Train Epoch: 4 [39680/60000 (66%)]	Loss: 0.822523
Train Epoch: 4 [40320/60000 (67%)]	Loss: 0.690123
Train Epoch: 4 [40960/60000 (68%)]	Loss: 0.961949
Train Epoch: 4 [41600/60000 (69%)]	Loss: 1.066585
Train Epoch: 4 [42240/60000 (70%)]	Loss: 0.775305
Train Epoch: 4 [42880/60000 (71%)]	Loss: 0.729686
Train Epoch: 4 [43520/60000 (72%)]	Loss: 1.145254
Train Epoch: 4 [44160/60000 (74%)]	Loss: 0.703635
Train Epoch: 4 [44800/60000 (75%)]	Loss: 0.721731
Train Epoch: 4 [45440/60000 (76%)]	Loss: 0.664871
Train Epoch: 4 [46080/60000 (77%)]	Loss: 0.705843
Train Epoch: 4 [46720/60000 (78%)]	Loss: 0.749343
Train Epoch: 4 [47360/60000 (79%)]	Loss: 0.758669
Train Epoch: 4 [48000/60000 (80%)]	Loss: 0.788508
Train Epoch: 4 [48640/60000 (81%)]	Loss: 0.775767
Train Epoch: 4 [49280/60000 (82%)]	Loss: 0.622524
Train Epoch: 4 [49920/60000 (83%)]	Loss: 0.815505
Train Epoch: 4 [50560/60000 (84%)]	Loss: 0.640356
Train Epoch: 4 [51200/60000 (85%)]	Loss: 0.918041
Train Epoch: 4 [51840/60000 (86%)]	Loss: 0.815764
Train Epoch: 4 [52480/60000 (87%)]	Loss: 0.966684
Train Epoch: 4 [53120/60000 (88%)]	Loss: 0.782149
Train Epoch: 4 [53760/60000 (90%)]	Loss: 0.707492
Train Epoch: 4 [54400/60000 (91%)]	Loss: 0.645954
Train Epoch: 4 [55040/60000 (92%)]	Loss: 0.807256
Train Epoch: 4 [55680/60000 (93%)]	Loss: 0.981960
Train Epoch: 4 [56320/60000 (94%)]	Loss: 0.916757
Train Epoch: 4 [56960/60000 (95%)]	Loss: 0.712529
Train Epoch: 4 [57600/60000 (96%)]	Loss: 0.833771
Train Epoch: 4 [58240/60000 (97%)]	Loss: 1.007091
Train Epoch: 4 [58880/60000 (98%)]	Loss: 0.943375
Train Epoch: 4 [59520/60000 (99%)]	Loss: 0.812929
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)

Test set: Average loss: 0.3929, Accuracy: 8936/10000 (89%)

[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
Train Epoch: 5 [0/60000 (0%)]	Loss: 0.584294
Train Epoch: 5 [640/60000 (1%)]	Loss: 0.702034
Train Epoch: 5 [1280/60000 (2%)]	Loss: 0.788549
Train Epoch: 5 [1920/60000 (3%)]	Loss: 0.869216
Train Epoch: 5 [2560/60000 (4%)]	Loss: 0.741394
Train Epoch: 5 [3200/60000 (5%)]	Loss: 0.755590
Train Epoch: 5 [3840/60000 (6%)]	Loss: 0.562761
Train Epoch: 5 [4480/60000 (7%)]	Loss: 0.722872
Train Epoch: 5 [5120/60000 (9%)]	Loss: 0.752977
Train Epoch: 5 [5760/60000 (10%)]	Loss: 0.811368
Train Epoch: 5 [6400/60000 (11%)]	Loss: 0.796170
Train Epoch: 5 [7040/60000 (12%)]	Loss: 1.091521
Train Epoch: 5 [7680/60000 (13%)]	Loss: 0.833255
Train Epoch: 5 [8320/60000 (14%)]	Loss: 0.780689
Train Epoch: 5 [8960/60000 (15%)]	Loss: 0.708085
Train Epoch: 5 [9600/60000 (16%)]	Loss: 0.861150
Train Epoch: 5 [10240/60000 (17%)]	Loss: 0.805887
Train Epoch: 5 [10880/60000 (18%)]	Loss: 0.843154
Train Epoch: 5 [11520/60000 (19%)]	Loss: 0.726924
Train Epoch: 5 [12160/60000 (20%)]	Loss: 0.808584
Train Epoch: 5 [12800/60000 (21%)]	Loss: 0.801551
Train Epoch: 5 [13440/60000 (22%)]	Loss: 0.910230
Train Epoch: 5 [14080/60000 (23%)]	Loss: 0.799476
Train Epoch: 5 [14720/60000 (25%)]	Loss: 0.606704
Train Epoch: 5 [15360/60000 (26%)]	Loss: 0.571000
Train Epoch: 5 [16000/60000 (27%)]	Loss: 0.686178
Train Epoch: 5 [16640/60000 (28%)]	Loss: 0.646732
Train Epoch: 5 [17280/60000 (29%)]	Loss: 0.876205
Train Epoch: 5 [17920/60000 (30%)]	Loss: 0.740199
Train Epoch: 5 [18560/60000 (31%)]	Loss: 0.694995
Train Epoch: 5 [19200/60000 (32%)]	Loss: 0.780339
Train Epoch: 5 [19840/60000 (33%)]	Loss: 0.598599
Train Epoch: 5 [20480/60000 (34%)]	Loss: 0.815254
Train Epoch: 5 [21120/60000 (35%)]	Loss: 0.472954
Train Epoch: 5 [21760/60000 (36%)]	Loss: 0.653084
Train Epoch: 5 [22400/60000 (37%)]	Loss: 0.716123
Train Epoch: 5 [23040/60000 (38%)]	Loss: 0.642279
Train Epoch: 5 [23680/60000 (39%)]	Loss: 0.700955
Train Epoch: 5 [24320/60000 (41%)]	Loss: 0.801266
Train Epoch: 5 [24960/60000 (42%)]	Loss: 0.818425
Train Epoch: 5 [25600/60000 (43%)]	Loss: 0.739550
Train Epoch: 5 [26240/60000 (44%)]	Loss: 0.792889
Train Epoch: 5 [26880/60000 (45%)]	Loss: 0.712154
Train Epoch: 5 [27520/60000 (46%)]	Loss: 0.557275
Train Epoch: 5 [28160/60000 (47%)]	Loss: 0.805634
Train Epoch: 5 [28800/60000 (48%)]	Loss: 0.619337
Train Epoch: 5 [29440/60000 (49%)]	Loss: 0.630538
Train Epoch: 5 [30080/60000 (50%)]	Loss: 0.810812
Train Epoch: 5 [30720/60000 (51%)]	Loss: 0.595721
Train Epoch: 5 [31360/60000 (52%)]	Loss: 0.781448
Train Epoch: 5 [32000/60000 (53%)]	Loss: 0.691883
Train Epoch: 5 [32640/60000 (54%)]	Loss: 0.838382
Train Epoch: 5 [33280/60000 (55%)]	Loss: 0.756405
Train Epoch: 5 [33920/60000 (57%)]	Loss: 0.972742
Train Epoch: 5 [34560/60000 (58%)]	Loss: 0.787852
Train Epoch: 5 [35200/60000 (59%)]	Loss: 0.702367
Train Epoch: 5 [35840/60000 (60%)]	Loss: 0.994428
Train Epoch: 5 [36480/60000 (61%)]	Loss: 0.715860
Train Epoch: 5 [37120/60000 (62%)]	Loss: 0.810936
Train Epoch: 5 [37760/60000 (63%)]	Loss: 0.777601
Train Epoch: 5 [38400/60000 (64%)]	Loss: 1.020633
Train Epoch: 5 [39040/60000 (65%)]	Loss: 0.888612
Train Epoch: 5 [39680/60000 (66%)]	Loss: 0.757675
Train Epoch: 5 [40320/60000 (67%)]	Loss: 0.613634
Train Epoch: 5 [40960/60000 (68%)]	Loss: 0.835350
Train Epoch: 5 [41600/60000 (69%)]	Loss: 0.985203
Train Epoch: 5 [42240/60000 (70%)]	Loss: 0.994960
Train Epoch: 5 [42880/60000 (71%)]	Loss: 0.811218
Train Epoch: 5 [43520/60000 (72%)]	Loss: 0.843782
Train Epoch: 5 [44160/60000 (74%)]	Loss: 0.800181
Train Epoch: 5 [44800/60000 (75%)]	Loss: 0.589761
Train Epoch: 5 [45440/60000 (76%)]	Loss: 0.645106
Train Epoch: 5 [46080/60000 (77%)]	Loss: 0.808097
Train Epoch: 5 [46720/60000 (78%)]	Loss: 0.649742
Train Epoch: 5 [47360/60000 (79%)]	Loss: 0.678556
Train Epoch: 5 [48000/60000 (80%)]	Loss: 0.859790
Train Epoch: 5 [48640/60000 (81%)]	Loss: 0.636586
Train Epoch: 5 [49280/60000 (82%)]	Loss: 0.483546
Train Epoch: 5 [49920/60000 (83%)]	Loss: 0.703626
Train Epoch: 5 [50560/60000 (84%)]	Loss: 0.645162
Train Epoch: 5 [51200/60000 (85%)]	Loss: 0.750438
Train Epoch: 5 [51840/60000 (86%)]	Loss: 0.537934
Train Epoch: 5 [52480/60000 (87%)]	Loss: 0.624096
Train Epoch: 5 [53120/60000 (88%)]	Loss: 0.691094
Train Epoch: 5 [53760/60000 (90%)]	Loss: 0.710470
Train Epoch: 5 [54400/60000 (91%)]	Loss: 0.865340
Train Epoch: 5 [55040/60000 (92%)]	Loss: 0.996537
Train Epoch: 5 [55680/60000 (93%)]	Loss: 0.900697
Train Epoch: 5 [56320/60000 (94%)]	Loss: 0.825120
Train Epoch: 5 [56960/60000 (95%)]	Loss: 0.885755
Train Epoch: 5 [57600/60000 (96%)]	Loss: 0.686889
Train Epoch: 5 [58240/60000 (97%)]	Loss: 0.710638
Train Epoch: 5 [58880/60000 (98%)]	Loss: 0.930715
Train Epoch: 5 [59520/60000 (99%)]	Loss: 0.592939
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)

Test set: Average loss: 0.3364, Accuracy: 9006/10000 (90%)

[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
Train Epoch: 6 [0/60000 (0%)]	Loss: 0.519532
Train Epoch: 6 [640/60000 (1%)]	Loss: 1.037046
Train Epoch: 6 [1280/60000 (2%)]	Loss: 0.542710
Train Epoch: 6 [1920/60000 (3%)]	Loss: 0.594631
Train Epoch: 6 [2560/60000 (4%)]	Loss: 1.081660
Train Epoch: 6 [3200/60000 (5%)]	Loss: 0.696961
Train Epoch: 6 [3840/60000 (6%)]	Loss: 0.740645
Train Epoch: 6 [4480/60000 (7%)]	Loss: 0.805133
Train Epoch: 6 [5120/60000 (9%)]	Loss: 0.580431
Train Epoch: 6 [5760/60000 (10%)]	Loss: 0.460867
Train Epoch: 6 [6400/60000 (11%)]	Loss: 0.939416
Train Epoch: 6 [7040/60000 (12%)]	Loss: 0.694352
Train Epoch: 6 [7680/60000 (13%)]	Loss: 0.751146
Train Epoch: 6 [8320/60000 (14%)]	Loss: 0.778630
Train Epoch: 6 [8960/60000 (15%)]	Loss: 0.587282
Train Epoch: 6 [9600/60000 (16%)]	Loss: 0.705371
Train Epoch: 6 [10240/60000 (17%)]	Loss: 0.571227
Train Epoch: 6 [10880/60000 (18%)]	Loss: 0.810753
Train Epoch: 6 [11520/60000 (19%)]	Loss: 0.629408
Train Epoch: 6 [12160/60000 (20%)]	Loss: 0.686027
Train Epoch: 6 [12800/60000 (21%)]	Loss: 0.560827
Train Epoch: 6 [13440/60000 (22%)]	Loss: 0.870457
Train Epoch: 6 [14080/60000 (23%)]	Loss: 0.696409
Train Epoch: 6 [14720/60000 (25%)]	Loss: 0.729584
Train Epoch: 6 [15360/60000 (26%)]	Loss: 0.646043
Train Epoch: 6 [16000/60000 (27%)]	Loss: 0.372504
Train Epoch: 6 [16640/60000 (28%)]	Loss: 0.574992
Train Epoch: 6 [17280/60000 (29%)]	Loss: 0.951679
Train Epoch: 6 [17920/60000 (30%)]	Loss: 0.593548
Train Epoch: 6 [18560/60000 (31%)]	Loss: 0.735420
Train Epoch: 6 [19200/60000 (32%)]	Loss: 0.977771
Train Epoch: 6 [19840/60000 (33%)]	Loss: 0.843979
Train Epoch: 6 [20480/60000 (34%)]	Loss: 0.769431
Train Epoch: 6 [21120/60000 (35%)]	Loss: 1.008069
Train Epoch: 6 [21760/60000 (36%)]	Loss: 0.492040
Train Epoch: 6 [22400/60000 (37%)]	Loss: 0.683452
Train Epoch: 6 [23040/60000 (38%)]	Loss: 0.655528
Train Epoch: 6 [23680/60000 (39%)]	Loss: 0.893405
Train Epoch: 6 [24320/60000 (41%)]	Loss: 0.664137
Train Epoch: 6 [24960/60000 (42%)]	Loss: 0.658683
Train Epoch: 6 [25600/60000 (43%)]	Loss: 0.743006
Train Epoch: 6 [26240/60000 (44%)]	Loss: 0.593071
Train Epoch: 6 [26880/60000 (45%)]	Loss: 0.908920
Train Epoch: 6 [27520/60000 (46%)]	Loss: 0.782316
Train Epoch: 6 [28160/60000 (47%)]	Loss: 0.863372
Train Epoch: 6 [28800/60000 (48%)]	Loss: 0.770900
Train Epoch: 6 [29440/60000 (49%)]	Loss: 0.753336
Train Epoch: 6 [30080/60000 (50%)]	Loss: 0.705974
Train Epoch: 6 [30720/60000 (51%)]	Loss: 0.672230
Train Epoch: 6 [31360/60000 (52%)]	Loss: 0.588636
Train Epoch: 6 [32000/60000 (53%)]	Loss: 0.545254
Train Epoch: 6 [32640/60000 (54%)]	Loss: 0.745948
Train Epoch: 6 [33280/60000 (55%)]	Loss: 0.604609
Train Epoch: 6 [33920/60000 (57%)]	Loss: 0.789983
Train Epoch: 6 [34560/60000 (58%)]	Loss: 0.660512
Train Epoch: 6 [35200/60000 (59%)]	Loss: 0.746651
Train Epoch: 6 [35840/60000 (60%)]	Loss: 0.827999
Train Epoch: 6 [36480/60000 (61%)]	Loss: 0.785850
Train Epoch: 6 [37120/60000 (62%)]	Loss: 0.676891
Train Epoch: 6 [37760/60000 (63%)]	Loss: 0.720178
Train Epoch: 6 [38400/60000 (64%)]	Loss: 0.525398
Train Epoch: 6 [39040/60000 (65%)]	Loss: 0.765047
Train Epoch: 6 [39680/60000 (66%)]	Loss: 0.887466
Train Epoch: 6 [40320/60000 (67%)]	Loss: 0.806764
Train Epoch: 6 [40960/60000 (68%)]	Loss: 0.675670
Train Epoch: 6 [41600/60000 (69%)]	Loss: 0.562905
Train Epoch: 6 [42240/60000 (70%)]	Loss: 0.512158
Train Epoch: 6 [42880/60000 (71%)]	Loss: 0.759140
Train Epoch: 6 [43520/60000 (72%)]	Loss: 0.680200
Train Epoch: 6 [44160/60000 (74%)]	Loss: 0.642582
Train Epoch: 6 [44800/60000 (75%)]	Loss: 0.707231
Train Epoch: 6 [45440/60000 (76%)]	Loss: 0.596457
Train Epoch: 6 [46080/60000 (77%)]	Loss: 0.705999
Train Epoch: 6 [46720/60000 (78%)]	Loss: 0.807577
Train Epoch: 6 [47360/60000 (79%)]	Loss: 0.732639
Train Epoch: 6 [48000/60000 (80%)]	Loss: 0.925131
Train Epoch: 6 [48640/60000 (81%)]	Loss: 0.785398
Train Epoch: 6 [49280/60000 (82%)]	Loss: 0.674538
Train Epoch: 6 [49920/60000 (83%)]	Loss: 0.884968
Train Epoch: 6 [50560/60000 (84%)]	Loss: 0.738120
Train Epoch: 6 [51200/60000 (85%)]	Loss: 0.455863
Train Epoch: 6 [51840/60000 (86%)]	Loss: 0.769018
Train Epoch: 6 [52480/60000 (87%)]	Loss: 0.677780
Train Epoch: 6 [53120/60000 (88%)]	Loss: 0.915004
Train Epoch: 6 [53760/60000 (90%)]	Loss: 0.596896
Train Epoch: 6 [54400/60000 (91%)]	Loss: 0.611141
Train Epoch: 6 [55040/60000 (92%)]	Loss: 0.694584
Train Epoch: 6 [55680/60000 (93%)]	Loss: 0.655248
Train Epoch: 6 [56320/60000 (94%)]	Loss: 0.625898
Train Epoch: 6 [56960/60000 (95%)]	Loss: 0.983437
Train Epoch: 6 [57600/60000 (96%)]	Loss: 0.716865
Train Epoch: 6 [58240/60000 (97%)]	Loss: 0.715149
Train Epoch: 6 [58880/60000 (98%)]	Loss: 0.728304
Train Epoch: 6 [59520/60000 (99%)]	Loss: 0.702129
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)

Test set: Average loss: 0.3213, Accuracy: 9115/10000 (91%)

[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
Train Epoch: 7 [0/60000 (0%)]	Loss: 0.812081
Train Epoch: 7 [640/60000 (1%)]	Loss: 0.870381
Train Epoch: 7 [1280/60000 (2%)]	Loss: 0.965321
Train Epoch: 7 [1920/60000 (3%)]	Loss: 0.659881
Train Epoch: 7 [2560/60000 (4%)]	Loss: 0.742517
Train Epoch: 7 [3200/60000 (5%)]	Loss: 0.662181
Train Epoch: 7 [3840/60000 (6%)]	Loss: 0.444929
Train Epoch: 7 [4480/60000 (7%)]	Loss: 0.681270
Train Epoch: 7 [5120/60000 (9%)]	Loss: 0.737047
Train Epoch: 7 [5760/60000 (10%)]	Loss: 0.856504
Train Epoch: 7 [6400/60000 (11%)]	Loss: 0.684951
Train Epoch: 7 [7040/60000 (12%)]	Loss: 0.557685
Train Epoch: 7 [7680/60000 (13%)]	Loss: 0.659950
Train Epoch: 7 [8320/60000 (14%)]	Loss: 0.815684
Train Epoch: 7 [8960/60000 (15%)]	Loss: 0.776752
Train Epoch: 7 [9600/60000 (16%)]	Loss: 0.528628
Train Epoch: 7 [10240/60000 (17%)]	Loss: 0.712441
Train Epoch: 7 [10880/60000 (18%)]	Loss: 0.796645
Train Epoch: 7 [11520/60000 (19%)]	Loss: 0.532408
Train Epoch: 7 [12160/60000 (20%)]	Loss: 0.647000
Train Epoch: 7 [12800/60000 (21%)]	Loss: 0.478404
Train Epoch: 7 [13440/60000 (22%)]	Loss: 0.516391
Train Epoch: 7 [14080/60000 (23%)]	Loss: 0.463970
Train Epoch: 7 [14720/60000 (25%)]	Loss: 0.652738
Train Epoch: 7 [15360/60000 (26%)]	Loss: 0.528860
Train Epoch: 7 [16000/60000 (27%)]	Loss: 0.782153
Train Epoch: 7 [16640/60000 (28%)]	Loss: 0.667849
Train Epoch: 7 [17280/60000 (29%)]	Loss: 0.707953
Train Epoch: 7 [17920/60000 (30%)]	Loss: 0.783702
Train Epoch: 7 [18560/60000 (31%)]	Loss: 0.610266
Train Epoch: 7 [19200/60000 (32%)]	Loss: 0.719588
Train Epoch: 7 [19840/60000 (33%)]	Loss: 0.672957
Train Epoch: 7 [20480/60000 (34%)]	Loss: 0.528706
Train Epoch: 7 [21120/60000 (35%)]	Loss: 0.653301
Train Epoch: 7 [21760/60000 (36%)]	Loss: 0.855872
Train Epoch: 7 [22400/60000 (37%)]	Loss: 0.547812
Train Epoch: 7 [23040/60000 (38%)]	Loss: 0.600127
Train Epoch: 7 [23680/60000 (39%)]	Loss: 0.628648
Train Epoch: 7 [24320/60000 (41%)]	Loss: 0.920549
Train Epoch: 7 [24960/60000 (42%)]	Loss: 0.672164
Train Epoch: 7 [25600/60000 (43%)]	Loss: 0.740594
Train Epoch: 7 [26240/60000 (44%)]	Loss: 0.868826
Train Epoch: 7 [26880/60000 (45%)]	Loss: 0.680436
Train Epoch: 7 [27520/60000 (46%)]	Loss: 0.467218
Train Epoch: 7 [28160/60000 (47%)]	Loss: 0.973473
Train Epoch: 7 [28800/60000 (48%)]	Loss: 0.583361
Train Epoch: 7 [29440/60000 (49%)]	Loss: 1.017473
Train Epoch: 7 [30080/60000 (50%)]	Loss: 0.742814
Train Epoch: 7 [30720/60000 (51%)]	Loss: 0.654992
Train Epoch: 7 [31360/60000 (52%)]	Loss: 0.913827
Train Epoch: 7 [32000/60000 (53%)]	Loss: 0.529535
Train Epoch: 7 [32640/60000 (54%)]	Loss: 0.642715
Train Epoch: 7 [33280/60000 (55%)]	Loss: 0.790316
Train Epoch: 7 [33920/60000 (57%)]	Loss: 0.637835
Train Epoch: 7 [34560/60000 (58%)]	Loss: 0.754427
Train Epoch: 7 [35200/60000 (59%)]	Loss: 0.622874
Train Epoch: 7 [35840/60000 (60%)]	Loss: 0.874770
Train Epoch: 7 [36480/60000 (61%)]	Loss: 0.781229
Train Epoch: 7 [37120/60000 (62%)]	Loss: 0.791635
Train Epoch: 7 [37760/60000 (63%)]	Loss: 0.750560
Train Epoch: 7 [38400/60000 (64%)]	Loss: 0.860168
Train Epoch: 7 [39040/60000 (65%)]	Loss: 0.643608
Train Epoch: 7 [39680/60000 (66%)]	Loss: 0.794594
Train Epoch: 7 [40320/60000 (67%)]	Loss: 0.742787
Train Epoch: 7 [40960/60000 (68%)]	Loss: 0.753292
Train Epoch: 7 [41600/60000 (69%)]	Loss: 0.662667
Train Epoch: 7 [42240/60000 (70%)]	Loss: 0.731721
Train Epoch: 7 [42880/60000 (71%)]	Loss: 0.762484
Train Epoch: 7 [43520/60000 (72%)]	Loss: 0.448194
Train Epoch: 7 [44160/60000 (74%)]	Loss: 0.772577
Train Epoch: 7 [44800/60000 (75%)]	Loss: 0.764851
Train Epoch: 7 [45440/60000 (76%)]	Loss: 0.648125
Train Epoch: 7 [46080/60000 (77%)]	Loss: 0.544488
Train Epoch: 7 [46720/60000 (78%)]	Loss: 0.917852
Train Epoch: 7 [47360/60000 (79%)]	Loss: 0.712430
Train Epoch: 7 [48000/60000 (80%)]	Loss: 0.617906
Train Epoch: 7 [48640/60000 (81%)]	Loss: 0.580676
Train Epoch: 7 [49280/60000 (82%)]	Loss: 0.807021
Train Epoch: 7 [49920/60000 (83%)]	Loss: 0.613444
Train Epoch: 7 [50560/60000 (84%)]	Loss: 0.634260
Train Epoch: 7 [51200/60000 (85%)]	Loss: 0.814932
Train Epoch: 7 [51840/60000 (86%)]	Loss: 0.655988
Train Epoch: 7 [52480/60000 (87%)]	Loss: 0.697038
Train Epoch: 7 [53120/60000 (88%)]	Loss: 0.530144
Train Epoch: 7 [53760/60000 (90%)]	Loss: 0.666397
Train Epoch: 7 [54400/60000 (91%)]	Loss: 0.523613
Train Epoch: 7 [55040/60000 (92%)]	Loss: 0.556630
Train Epoch: 7 [55680/60000 (93%)]	Loss: 0.593061
Train Epoch: 7 [56320/60000 (94%)]	Loss: 0.871977
Train Epoch: 7 [56960/60000 (95%)]	Loss: 0.774657
Train Epoch: 7 [57600/60000 (96%)]	Loss: 0.484636
Train Epoch: 7 [58240/60000 (97%)]	Loss: 0.531363
Train Epoch: 7 [58880/60000 (98%)]	Loss: 0.762934
Train Epoch: 7 [59520/60000 (99%)]	Loss: 0.838743
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)

Test set: Average loss: 0.2887, Accuracy: 9142/10000 (91%)

[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
Train Epoch: 8 [0/60000 (0%)]	Loss: 0.545300
Train Epoch: 8 [640/60000 (1%)]	Loss: 0.732279
Train Epoch: 8 [1280/60000 (2%)]	Loss: 0.556092
Train Epoch: 8 [1920/60000 (3%)]	Loss: 0.864336
Train Epoch: 8 [2560/60000 (4%)]	Loss: 0.386126
Train Epoch: 8 [3200/60000 (5%)]	Loss: 0.567421
Train Epoch: 8 [3840/60000 (6%)]	Loss: 0.914241
Train Epoch: 8 [4480/60000 (7%)]	Loss: 0.631655
Train Epoch: 8 [5120/60000 (9%)]	Loss: 0.734193
Train Epoch: 8 [5760/60000 (10%)]	Loss: 0.493919
Train Epoch: 8 [6400/60000 (11%)]	Loss: 0.702290
Train Epoch: 8 [7040/60000 (12%)]	Loss: 0.837426
Train Epoch: 8 [7680/60000 (13%)]	Loss: 0.571231
Train Epoch: 8 [8320/60000 (14%)]	Loss: 0.594225
Train Epoch: 8 [8960/60000 (15%)]	Loss: 0.549097
Train Epoch: 8 [9600/60000 (16%)]	Loss: 0.584907
Train Epoch: 8 [10240/60000 (17%)]	Loss: 0.689095
Train Epoch: 8 [10880/60000 (18%)]	Loss: 0.778111
Train Epoch: 8 [11520/60000 (19%)]	Loss: 0.505485
Train Epoch: 8 [12160/60000 (20%)]	Loss: 0.708942
Train Epoch: 8 [12800/60000 (21%)]	Loss: 0.623432
Train Epoch: 8 [13440/60000 (22%)]	Loss: 0.732075
Train Epoch: 8 [14080/60000 (23%)]	Loss: 0.601791
Train Epoch: 8 [14720/60000 (25%)]	Loss: 0.608199
Train Epoch: 8 [15360/60000 (26%)]	Loss: 0.620736
Train Epoch: 8 [16000/60000 (27%)]	Loss: 0.489355
Train Epoch: 8 [16640/60000 (28%)]	Loss: 0.533909
Train Epoch: 8 [17280/60000 (29%)]	Loss: 0.851386
Train Epoch: 8 [17920/60000 (30%)]	Loss: 0.545971
Train Epoch: 8 [18560/60000 (31%)]	Loss: 0.455955
Train Epoch: 8 [19200/60000 (32%)]	Loss: 0.857997
Train Epoch: 8 [19840/60000 (33%)]	Loss: 0.706021
Train Epoch: 8 [20480/60000 (34%)]	Loss: 0.613442
Train Epoch: 8 [21120/60000 (35%)]	Loss: 1.033512
Train Epoch: 8 [21760/60000 (36%)]	Loss: 0.585569
Train Epoch: 8 [22400/60000 (37%)]	Loss: 0.569441
Train Epoch: 8 [23040/60000 (38%)]	Loss: 0.632392
Train Epoch: 8 [23680/60000 (39%)]	Loss: 0.571356
Train Epoch: 8 [24320/60000 (41%)]	Loss: 0.635547
Train Epoch: 8 [24960/60000 (42%)]	Loss: 0.758841
Train Epoch: 8 [25600/60000 (43%)]	Loss: 0.578460
Train Epoch: 8 [26240/60000 (44%)]	Loss: 0.519483
Train Epoch: 8 [26880/60000 (45%)]	Loss: 0.670597
Train Epoch: 8 [27520/60000 (46%)]	Loss: 0.632382
Train Epoch: 8 [28160/60000 (47%)]	Loss: 0.681048
Train Epoch: 8 [28800/60000 (48%)]	Loss: 0.617827
Train Epoch: 8 [29440/60000 (49%)]	Loss: 0.742719
Train Epoch: 8 [30080/60000 (50%)]	Loss: 0.630589
Train Epoch: 8 [30720/60000 (51%)]	Loss: 0.879869
Train Epoch: 8 [31360/60000 (52%)]	Loss: 0.600611
Train Epoch: 8 [32000/60000 (53%)]	Loss: 0.630095
Train Epoch: 8 [32640/60000 (54%)]	Loss: 0.666438
Train Epoch: 8 [33280/60000 (55%)]	Loss: 0.751712
Train Epoch: 8 [33920/60000 (57%)]	Loss: 0.535851
Train Epoch: 8 [34560/60000 (58%)]	Loss: 0.471027
Train Epoch: 8 [35200/60000 (59%)]	Loss: 0.812434
Train Epoch: 8 [35840/60000 (60%)]	Loss: 0.597091
Train Epoch: 8 [36480/60000 (61%)]	Loss: 0.781169
Train Epoch: 8 [37120/60000 (62%)]	Loss: 0.696026
Train Epoch: 8 [37760/60000 (63%)]	Loss: 0.528295
Train Epoch: 8 [38400/60000 (64%)]	Loss: 1.009545
Train Epoch: 8 [39040/60000 (65%)]	Loss: 0.658693
Train Epoch: 8 [39680/60000 (66%)]	Loss: 0.460113
Train Epoch: 8 [40320/60000 (67%)]	Loss: 0.581584
Train Epoch: 8 [40960/60000 (68%)]	Loss: 0.435689
Train Epoch: 8 [41600/60000 (69%)]	Loss: 0.804556
Train Epoch: 8 [42240/60000 (70%)]	Loss: 0.956822
Train Epoch: 8 [42880/60000 (71%)]	Loss: 0.603841
Train Epoch: 8 [43520/60000 (72%)]	Loss: 0.487952
Train Epoch: 8 [44160/60000 (74%)]	Loss: 0.854909
Train Epoch: 8 [44800/60000 (75%)]	Loss: 0.505981
Train Epoch: 8 [45440/60000 (76%)]	Loss: 0.711943
Train Epoch: 8 [46080/60000 (77%)]	Loss: 0.475801
Train Epoch: 8 [46720/60000 (78%)]	Loss: 0.646754
Train Epoch: 8 [47360/60000 (79%)]	Loss: 0.531868
Train Epoch: 8 [48000/60000 (80%)]	Loss: 0.884119
Train Epoch: 8 [48640/60000 (81%)]	Loss: 0.604928
Train Epoch: 8 [49280/60000 (82%)]	Loss: 0.617891
Train Epoch: 8 [49920/60000 (83%)]	Loss: 0.684946
Train Epoch: 8 [50560/60000 (84%)]	Loss: 0.628150
Train Epoch: 8 [51200/60000 (85%)]	Loss: 0.672675
Train Epoch: 8 [51840/60000 (86%)]	Loss: 0.793558
Train Epoch: 8 [52480/60000 (87%)]	Loss: 0.383031
Train Epoch: 8 [53120/60000 (88%)]	Loss: 0.461128
Train Epoch: 8 [53760/60000 (90%)]	Loss: 0.885543
Train Epoch: 8 [54400/60000 (91%)]	Loss: 0.710831
Train Epoch: 8 [55040/60000 (92%)]	Loss: 0.657191
Train Epoch: 8 [55680/60000 (93%)]	Loss: 0.571935
Train Epoch: 8 [56320/60000 (94%)]	Loss: 0.530606
Train Epoch: 8 [56960/60000 (95%)]	Loss: 0.547300
Train Epoch: 8 [57600/60000 (96%)]	Loss: 0.622733
Train Epoch: 8 [58240/60000 (97%)]	Loss: 0.580943
Train Epoch: 8 [58880/60000 (98%)]	Loss: 0.596418
Train Epoch: 8 [59520/60000 (99%)]	Loss: 0.786484
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)

Test set: Average loss: 0.2756, Accuracy: 9219/10000 (92%)

[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
Train Epoch: 9 [0/60000 (0%)]	Loss: 0.476628
Train Epoch: 9 [640/60000 (1%)]	Loss: 0.742248
Train Epoch: 9 [1280/60000 (2%)]	Loss: 0.891240
Train Epoch: 9 [1920/60000 (3%)]	Loss: 0.745276
Train Epoch: 9 [2560/60000 (4%)]	Loss: 0.617175
Train Epoch: 9 [3200/60000 (5%)]	Loss: 0.409639
Train Epoch: 9 [3840/60000 (6%)]	Loss: 0.632548
Train Epoch: 9 [4480/60000 (7%)]	Loss: 0.667092
Train Epoch: 9 [5120/60000 (9%)]	Loss: 0.535684
Train Epoch: 9 [5760/60000 (10%)]	Loss: 0.526260
Train Epoch: 9 [6400/60000 (11%)]	Loss: 0.868763
Train Epoch: 9 [7040/60000 (12%)]	Loss: 0.572482
Train Epoch: 9 [7680/60000 (13%)]	Loss: 0.767661
Train Epoch: 9 [8320/60000 (14%)]	Loss: 0.736408
Train Epoch: 9 [8960/60000 (15%)]	Loss: 0.667085
Train Epoch: 9 [9600/60000 (16%)]	Loss: 0.624356
Train Epoch: 9 [10240/60000 (17%)]	Loss: 0.707467
Train Epoch: 9 [10880/60000 (18%)]	Loss: 0.548567
Train Epoch: 9 [11520/60000 (19%)]	Loss: 0.600019
Train Epoch: 9 [12160/60000 (20%)]	Loss: 0.681690
Train Epoch: 9 [12800/60000 (21%)]	Loss: 0.585703
Train Epoch: 9 [13440/60000 (22%)]	Loss: 0.696176
Train Epoch: 9 [14080/60000 (23%)]	Loss: 0.931637
Train Epoch: 9 [14720/60000 (25%)]	Loss: 0.907039
Train Epoch: 9 [15360/60000 (26%)]	Loss: 0.615734
Train Epoch: 9 [16000/60000 (27%)]	Loss: 0.540708
Train Epoch: 9 [16640/60000 (28%)]	Loss: 0.540940
Train Epoch: 9 [17280/60000 (29%)]	Loss: 1.036808
Train Epoch: 9 [17920/60000 (30%)]	Loss: 0.589472
Train Epoch: 9 [18560/60000 (31%)]	Loss: 0.493852
Train Epoch: 9 [19200/60000 (32%)]	Loss: 0.644807
Train Epoch: 9 [19840/60000 (33%)]	Loss: 0.559911
Train Epoch: 9 [20480/60000 (34%)]	Loss: 0.582729
Train Epoch: 9 [21120/60000 (35%)]	Loss: 0.634841
Train Epoch: 9 [21760/60000 (36%)]	Loss: 0.680488
Train Epoch: 9 [22400/60000 (37%)]	Loss: 0.560358
Train Epoch: 9 [23040/60000 (38%)]	Loss: 0.656928
Train Epoch: 9 [23680/60000 (39%)]	Loss: 0.542422
Train Epoch: 9 [24320/60000 (41%)]	Loss: 0.743577
Train Epoch: 9 [24960/60000 (42%)]	Loss: 0.616481
Train Epoch: 9 [25600/60000 (43%)]	Loss: 0.823534
Train Epoch: 9 [26240/60000 (44%)]	Loss: 0.615524
Train Epoch: 9 [26880/60000 (45%)]	Loss: 0.642980
Train Epoch: 9 [27520/60000 (46%)]	Loss: 0.455591
Train Epoch: 9 [28160/60000 (47%)]	Loss: 0.721140
Train Epoch: 9 [28800/60000 (48%)]	Loss: 0.573046
Train Epoch: 9 [29440/60000 (49%)]	Loss: 0.733240
Train Epoch: 9 [30080/60000 (50%)]	Loss: 0.626926
Train Epoch: 9 [30720/60000 (51%)]	Loss: 0.571588
Train Epoch: 9 [31360/60000 (52%)]	Loss: 0.510743
Train Epoch: 9 [32000/60000 (53%)]	Loss: 0.579665
Train Epoch: 9 [32640/60000 (54%)]	Loss: 0.390165
Train Epoch: 9 [33280/60000 (55%)]	Loss: 0.838093
Train Epoch: 9 [33920/60000 (57%)]	Loss: 0.526741
Train Epoch: 9 [34560/60000 (58%)]	Loss: 0.648159
Train Epoch: 9 [35200/60000 (59%)]	Loss: 0.676569
Train Epoch: 9 [35840/60000 (60%)]	Loss: 0.646155
Train Epoch: 9 [36480/60000 (61%)]	Loss: 0.522618
Train Epoch: 9 [37120/60000 (62%)]	Loss: 0.569667
Train Epoch: 9 [37760/60000 (63%)]	Loss: 0.734929
Train Epoch: 9 [38400/60000 (64%)]	Loss: 0.751885
Train Epoch: 9 [39040/60000 (65%)]	Loss: 0.623726
Train Epoch: 9 [39680/60000 (66%)]	Loss: 0.625259
Train Epoch: 9 [40320/60000 (67%)]	Loss: 0.532557
Train Epoch: 9 [40960/60000 (68%)]	Loss: 0.644453
Train Epoch: 9 [41600/60000 (69%)]	Loss: 0.681302
Train Epoch: 9 [42240/60000 (70%)]	Loss: 0.511865
Train Epoch: 9 [42880/60000 (71%)]	Loss: 0.697414
Train Epoch: 9 [43520/60000 (72%)]	Loss: 0.673190
Train Epoch: 9 [44160/60000 (74%)]	Loss: 0.712953
Train Epoch: 9 [44800/60000 (75%)]	Loss: 0.490930
Train Epoch: 9 [45440/60000 (76%)]	Loss: 0.647010
Train Epoch: 9 [46080/60000 (77%)]	Loss: 0.496890
Train Epoch: 9 [46720/60000 (78%)]	Loss: 0.603977
Train Epoch: 9 [47360/60000 (79%)]	Loss: 0.627887
Train Epoch: 9 [48000/60000 (80%)]	Loss: 0.823113
Train Epoch: 9 [48640/60000 (81%)]	Loss: 0.432136
Train Epoch: 9 [49280/60000 (82%)]	Loss: 0.681094
Train Epoch: 9 [49920/60000 (83%)]	Loss: 0.534987
Train Epoch: 9 [50560/60000 (84%)]	Loss: 0.516200
Train Epoch: 9 [51200/60000 (85%)]	Loss: 0.708939
Train Epoch: 9 [51840/60000 (86%)]	Loss: 0.617040
Train Epoch: 9 [52480/60000 (87%)]	Loss: 0.776158
Train Epoch: 9 [53120/60000 (88%)]	Loss: 0.556002
Train Epoch: 9 [53760/60000 (90%)]	Loss: 0.463790
Train Epoch: 9 [54400/60000 (91%)]	Loss: 0.509891
Train Epoch: 9 [55040/60000 (92%)]	Loss: 0.569496
Train Epoch: 9 [55680/60000 (93%)]	Loss: 0.509110
Train Epoch: 9 [56320/60000 (94%)]	Loss: 0.832549
Train Epoch: 9 [56960/60000 (95%)]	Loss: 0.653966
Train Epoch: 9 [57600/60000 (96%)]	Loss: 0.583373
Train Epoch: 9 [58240/60000 (97%)]	Loss: 0.739660
Train Epoch: 9 [58880/60000 (98%)]	Loss: 0.477606
Train Epoch: 9 [59520/60000 (99%)]	Loss: 0.725795
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)

Test set: Average loss: 0.2590, Accuracy: 9260/10000 (93%)

[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)
Train Epoch: 10 [0/60000 (0%)]	Loss: 0.686875
Train Epoch: 10 [640/60000 (1%)]	Loss: 0.704639
Train Epoch: 10 [1280/60000 (2%)]	Loss: 0.761315
Train Epoch: 10 [1920/60000 (3%)]	Loss: 0.559238
Train Epoch: 10 [2560/60000 (4%)]	Loss: 0.579395
Train Epoch: 10 [3200/60000 (5%)]	Loss: 0.590017
Train Epoch: 10 [3840/60000 (6%)]	Loss: 0.631428
Train Epoch: 10 [4480/60000 (7%)]	Loss: 0.983485
Train Epoch: 10 [5120/60000 (9%)]	Loss: 0.830615
Train Epoch: 10 [5760/60000 (10%)]	Loss: 0.472453
Train Epoch: 10 [6400/60000 (11%)]	Loss: 0.546135
Train Epoch: 10 [7040/60000 (12%)]	Loss: 0.503827
Train Epoch: 10 [7680/60000 (13%)]	Loss: 0.472043
Train Epoch: 10 [8320/60000 (14%)]	Loss: 0.662744
Train Epoch: 10 [8960/60000 (15%)]	Loss: 0.421280
Train Epoch: 10 [9600/60000 (16%)]	Loss: 0.808916
Train Epoch: 10 [10240/60000 (17%)]	Loss: 0.466505
Train Epoch: 10 [10880/60000 (18%)]	Loss: 0.559782
Train Epoch: 10 [11520/60000 (19%)]	Loss: 0.719283
Train Epoch: 10 [12160/60000 (20%)]	Loss: 0.560325
Train Epoch: 10 [12800/60000 (21%)]	Loss: 0.755864
Train Epoch: 10 [13440/60000 (22%)]	Loss: 0.572163
Train Epoch: 10 [14080/60000 (23%)]	Loss: 0.766367
Train Epoch: 10 [14720/60000 (25%)]	Loss: 0.655101
Train Epoch: 10 [15360/60000 (26%)]	Loss: 0.532389
Train Epoch: 10 [16000/60000 (27%)]	Loss: 0.578950
Train Epoch: 10 [16640/60000 (28%)]	Loss: 0.491566
Train Epoch: 10 [17280/60000 (29%)]	Loss: 0.613495
Train Epoch: 10 [17920/60000 (30%)]	Loss: 0.489433
Train Epoch: 10 [18560/60000 (31%)]	Loss: 0.649323
Train Epoch: 10 [19200/60000 (32%)]	Loss: 0.644204
Train Epoch: 10 [19840/60000 (33%)]	Loss: 0.698320
Train Epoch: 10 [20480/60000 (34%)]	Loss: 0.519439
Train Epoch: 10 [21120/60000 (35%)]	Loss: 0.512537
Train Epoch: 10 [21760/60000 (36%)]	Loss: 0.625074
Train Epoch: 10 [22400/60000 (37%)]	Loss: 0.749649
Train Epoch: 10 [23040/60000 (38%)]	Loss: 0.589130
Train Epoch: 10 [23680/60000 (39%)]	Loss: 0.525833
Train Epoch: 10 [24320/60000 (41%)]	Loss: 0.663020
Train Epoch: 10 [24960/60000 (42%)]	Loss: 0.771062
Train Epoch: 10 [25600/60000 (43%)]	Loss: 0.906682
Train Epoch: 10 [26240/60000 (44%)]	Loss: 0.604946
Train Epoch: 10 [26880/60000 (45%)]	Loss: 0.769174
Train Epoch: 10 [27520/60000 (46%)]	Loss: 0.608802
Train Epoch: 10 [28160/60000 (47%)]	Loss: 0.699060
Train Epoch: 10 [28800/60000 (48%)]	Loss: 0.632508
Train Epoch: 10 [29440/60000 (49%)]	Loss: 0.709181
Train Epoch: 10 [30080/60000 (50%)]	Loss: 0.606269
Train Epoch: 10 [30720/60000 (51%)]	Loss: 0.533253
Train Epoch: 10 [31360/60000 (52%)]	Loss: 0.363474
Train Epoch: 10 [32000/60000 (53%)]	Loss: 0.561926
Train Epoch: 10 [32640/60000 (54%)]	Loss: 0.594320
Train Epoch: 10 [33280/60000 (55%)]	Loss: 0.575763
Train Epoch: 10 [33920/60000 (57%)]	Loss: 0.426388
Train Epoch: 10 [34560/60000 (58%)]	Loss: 0.488490
Train Epoch: 10 [35200/60000 (59%)]	Loss: 0.772393
Train Epoch: 10 [35840/60000 (60%)]	Loss: 0.699675
Train Epoch: 10 [36480/60000 (61%)]	Loss: 0.601580
Train Epoch: 10 [37120/60000 (62%)]	Loss: 0.552199
Train Epoch: 10 [37760/60000 (63%)]	Loss: 0.531176
Train Epoch: 10 [38400/60000 (64%)]	Loss: 0.527892
Train Epoch: 10 [39040/60000 (65%)]	Loss: 0.453218
Train Epoch: 10 [39680/60000 (66%)]	Loss: 0.696934
Train Epoch: 10 [40320/60000 (67%)]	Loss: 0.480923
Train Epoch: 10 [40960/60000 (68%)]	Loss: 0.573260
Train Epoch: 10 [41600/60000 (69%)]	Loss: 0.729898
Train Epoch: 10 [42240/60000 (70%)]	Loss: 0.796018
Train Epoch: 10 [42880/60000 (71%)]	Loss: 0.549223
Train Epoch: 10 [43520/60000 (72%)]	Loss: 0.687782
Train Epoch: 10 [44160/60000 (74%)]	Loss: 0.565116
Train Epoch: 10 [44800/60000 (75%)]	Loss: 0.618200
Train Epoch: 10 [45440/60000 (76%)]	Loss: 0.606978
Train Epoch: 10 [46080/60000 (77%)]	Loss: 0.374312
Train Epoch: 10 [46720/60000 (78%)]	Loss: 0.644292
Train Epoch: 10 [47360/60000 (79%)]	Loss: 0.684182
Train Epoch: 10 [48000/60000 (80%)]	Loss: 0.524912
Train Epoch: 10 [48640/60000 (81%)]	Loss: 0.717785
Train Epoch: 10 [49280/60000 (82%)]	Loss: 0.683891
Train Epoch: 10 [49920/60000 (83%)]	Loss: 0.547944
Train Epoch: 10 [50560/60000 (84%)]	Loss: 0.505640
Train Epoch: 10 [51200/60000 (85%)]	Loss: 0.666886
Train Epoch: 10 [51840/60000 (86%)]	Loss: 0.536955
Train Epoch: 10 [52480/60000 (87%)]	Loss: 0.650717
Train Epoch: 10 [53120/60000 (88%)]	Loss: 0.819586
Train Epoch: 10 [53760/60000 (90%)]	Loss: 0.643048
Train Epoch: 10 [54400/60000 (91%)]	Loss: 0.562197
Train Epoch: 10 [55040/60000 (92%)]	Loss: 0.302358
Train Epoch: 10 [55680/60000 (93%)]	Loss: 0.388260
Train Epoch: 10 [56320/60000 (94%)]	Loss: 0.679476
Train Epoch: 10 [56960/60000 (95%)]	Loss: 0.532152
Train Epoch: 10 [57600/60000 (96%)]	Loss: 0.730348
Train Epoch: 10 [58240/60000 (97%)]	Loss: 0.660330
Train Epoch: 10 [58880/60000 (98%)]	Loss: 0.517699
Train Epoch: 10 [59520/60000 (99%)]	Loss: 0.585188
[W pthreadpool-cpp.cc:90] Warning: Leaking Caffe2 thread-pool after fork. (function pthreadpool)

Test set: Average loss: 0.2553, Accuracy: 9265/10000 (93%)

Process finished with exit code 0


"""
