import argparse
import time
from typing import List, Tuple
import sys

sys.path.append('../')
import torch
import torch.nn as nn
import torch.optim as optim
from utils.train import train, test, train_ewc, test_ewc, train_ewc_vgg, test_ewc_vgg
from utils.network_wider_cifar100_2 import Netwider, Netwider_multi
from utils.cifar100_dataloader import get_permute_cifar100, get_single_agent_cifar100

parser = argparse.ArgumentParser(description='single_agent_cifar100_wEWC')
parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')
parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')
parser.add_argument('--lr', type=float, default=0.005, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--w_kd', type=float, default=1, metavar='M', help='SGD momentum (default: 0.9)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')
parser.add_argument('--agent', type=int, default=0, help='number of tasks')
parser.add_argument('--num_imgs_per_cat_train', type=int, default=500)
parser.add_argument('--path', type=str, default='./', help='path of base classes')
parser.add_argument('--gpu_num', type=int, default=0)
parser.add_argument('--subtask_classes_num', type=int, default=25)

args = parser.parse_args()
#print("\n".join([f"{key}: {value}" for key, value in vars(args).items()]))
torch.cuda.set_device(args.gpu_num)
args.cuda = not args.no_cuda and torch.cuda.is_available()
torch.manual_seed(args.seed)
if args.cuda:
    torch.cuda.manual_seed(args.seed)
CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)
cfg_m = [64, 64,    64, 64,     128, 128, 128,    256, 256, 256,    256, 256, 256]
cfg_m_multi = [64, 64,    128, 128,     256, 256, 256,    512, 512, 512,    512, 512, 512]

def main():
    print("Data loading...")
    args.path = '/home/visiondata/yaoxinjie/MAC_29150/MAC/utils/exp_data/data_cifar100/2023-06-16_08:59:22/continualdataset'
    trainloader_single_agent, testloader_single_agent = get_single_agent_cifar100(args.agent,
                                                                                  args.batch_size,
                                                                                  subtask_classes_num=args.subtask_classes_num,
                                                                                  num_imgs_per_cate=args.num_imgs_per_cat_train,
                                                                                  path=args.path)
    print("Training single agent: Model constructing...")
    model_t = Netwider(13)
    model = Netwider_multi(13)
    print(model)
    if args.cuda:
        model = model.cuda()
    for epoch in range(args.epochs):
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)
        criterion = nn.CrossEntropyLoss()
        train_loss, train_acc = train(trainloader_single_agent[args.agent](epoch), model, criterion, optimizer, args)
        test_loss, test_acc = test(testloader_single_agent[args.agent](epoch), model, criterion, optimizer, args)
        print('epoch:{} Train{:.4f} {:.4f} Test{:.4f} {:.4f}'.format(epoch, train_loss, train_acc.item(), test_loss, test_acc.item()))

class ResidualBlock(torch.nn.Module):
    def __init__(self, inplanes: int, planes: int, stride: int = 1):
        super(ResidualBlock, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=inplanes,
                                     out_channels=planes,
                                     kernel_size=(3, 3),
                                     stride=stride,
                                     padding=1,
                                     bias=False)

        self.relu = torch.nn.ReLU(inplace=True)

        self.conv2 = torch.nn.Conv2d(in_channels=planes,
                                     out_channels=planes,
                                     kernel_size=(3, 3),
                                     stride=stride,
                                     padding=1,
                                     bias=False)
        self.downsample = None
        if stride > 1 or inplanes != planes:
            self.downsample = torch.nn.Sequential(torch.nn.Conv2d(in_channels=inplanes,
                                                                  out_channels=planes,
                                                                  kernel_size=(1, 1),
                                                                  stride=stride,
                                                                  bias=False)
                                                  )

        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.conv1(x)
        x = self.relu(x)
        x = self.conv2(x)

        if self.downsample is not None:
            residual = self.downsample(residual)

        x += residual
        x = self.relu(x)
        return x
class CommonFeatureBlocks(torch.nn.Module):
    def __init__(self, n_student_channels: int, n_teacher_channels: List[int], n_hidden_channel: int):
        super(CommonFeatureBlocks, self).__init__()

        ch_s = n_student_channels  # Readability
        ch_ts = n_teacher_channels  # Readability
        ch_h = n_hidden_channel  # Readability

        self.align_t = torch.nn.ModuleList()
        for ch_t in ch_ts:
            self.align_t.append(
                torch.nn.Sequential(torch.nn.Conv2d(in_channels=ch_t, out_channels=2 * ch_h, kernel_size=(1, 1), bias=False), torch.nn.ReLU(inplace=True)))
        self.align_s = torch.nn.Sequential(torch.nn.Conv2d(in_channels=ch_s, out_channels=2 * ch_h, kernel_size=(1, 1), bias=False), torch.nn.ReLU(inplace=True))
        self.extractor = torch.nn.Sequential(ResidualBlock(inplanes=2 * ch_h, planes=ch_h, stride=1),
                                             ResidualBlock(inplanes=ch_h, planes=ch_h, stride=1),
                                             ResidualBlock(inplanes=ch_h, planes=ch_h, stride=1))
        self.dec_t = torch.nn.ModuleList()
        for ch_t in ch_ts:
            self.dec_t.append(
                torch.nn.Sequential(torch.nn.Conv2d(in_channels=ch_h, out_channels=ch_t, kernel_size=(3, 3), stride=1, padding=1, bias=False), torch.nn.ReLU(inplace=True),
                                    torch.nn.Conv2d(in_channels=ch_t, out_channels=ch_t, kernel_size=(1, 1), stride=1, padding=0, bias=False)))

    def forward(self, fs: torch.Tensor, ft: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        aligned_t = [align(f) for align, f in zip(self.align_t, ft)]
        aligned_s = self.align_s(fs)
        ht = [self.extractor(f) for f in aligned_t]
        hs = self.extractor(aligned_s)
        ft_ = [dec(h) for dec, h in zip(self.dec_t, ht)]
        return hs, ht, ft_

def train(train_loader, model, criterion, optimizer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    model.train()
    end = time.time()
    #if args.agent == 1:
    #    model_t =
    for i, (images, targets) in enumerate(train_loader):
        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        #model.layer4.output
        loss = criterion(outputs, targets)
        loss_kd = 0
        # if args.agent==1:
        #     for layer in range(len(cfg_m)):
        #         cfl_blk = CommonFeatureBlocks(cfg_m_multi[layer], cfg_m[layer], cfg_m[layer])
        #         #1加载模型
        #
        #         #2 输入
        #         mean, var = torch.mean(out), torch.var(out)
        #         #3 输出特征
        #         fs = model.con2d
        #         ft = model_t
        #         hs, ht, ft_ = cfl_blk(fs, ft)
        #         loss_kd += torch.nn.functional.kl_div(torch.log_softmax(hs, dim=1), torch.softmax(ht, dim=1))
        #     loss_kd /= len(cfg_m)
        #         #共同子空间计算蒸馏损失
        #     print('loss', loss, loss_kd)
        #     loss = loss + args.w_kd * loss_kd

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()
    return losses.avg, top1.avg

def test(test_loader, model, criterion, optimizer, args):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')
    model.eval()
    with torch.no_grad():
        end = time.time()
        for i, (images, target) in enumerate(test_loader):
            if args.cuda:
                images = images.cuda(non_blocking=True)
                target = target.cuda(non_blocking=True)
            output = model(images)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            acc.update(acc1[0], images.size(0))
            batch_time.update(time.time() - end)
            end = time.time()

    return losses.sum / (len(test_loader.dataset)), acc.sum / (len(test_loader.dataset))
def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].contiguous().view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res
class AverageMeter(object):
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)
if __name__ == "__main__":
    main()

