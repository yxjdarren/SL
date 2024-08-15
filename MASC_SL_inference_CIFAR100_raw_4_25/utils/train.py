import time
import sys

sys.path.append('../')
import copy
import numpy as np
import os
import shutil
import random
import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.autograd import Variable

from .Ewc_class import EWC, EWC_vgg


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


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


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


def train(train_loader, model, criterion, optimizer, epoch, args, snapshot, name):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(len(train_loader), [batch_time, data_time, losses, top1, top5], prefix="Epoch: [{}]".format(epoch))

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))

        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) + \
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def test(test_loader, model, criterion, optimizer, epoch, args, snapshot, name):
    batch_time = AverageMeter('Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    acc = AverageMeter('Acc@1', ':6.2f')
    # progress = ProgressMeter(len(test_loader), [batch_time, losses, acc], prefix='Test: ')

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

        print(' * Test-Acc {top1.avg:.3f} '.format(top1=acc))

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        ' Test set: Average loss: ' + str(losses.sum / len(test_loader.dataset)) + ', Accuracy: ' + str(acc.avg) + '\n')

    return losses.sum / (len(test_loader.dataset)), acc.sum / (len(test_loader.dataset))

def train_ewc(train_loader, model, criterion, optimizer, epoch, args, snapshot, name,
              fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        loss = ce_loss

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg

def train_fd(train_loader, model, criterion, optimizer, epoch, args, snapshot, name,
              fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss

        # 获取模型的classifier模块列表
        classifier_modules_s = model.classifier
        classifier_modules_save_s = model.classifier

        # 移除最后一个元素（softmax层）
        classifier_modules_s = classifier_modules_s[:-1]

        # 将更新后的classifier模块列表设置回模型
        model.classifier = nn.ModuleList(classifier_modules_s)
        logit_output_s = model(images)

        model.classifier = nn.ModuleList(classifier_modules_save_s)
        # logit_output_s = model_s(images)

        output_div_t = -1.0 * args.energy_beta * logit_output_s
        output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
        free_energy = -1.0 * output_logsumexp / args.energy_beta
        align_loss = args.align_lambda * ((free_energy - args.anchor_energy) ** 2).mean()

        loss = ce_loss + align_loss

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg

def train_fd_thin(delete_layer, train_loader, model, criterion, optimizer, epoch, args, snapshot, name,
              fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)

    # thin_list = nn.ModuleList()
    # thin_list_1 = model.layers[0:delete_layer]
    # thin_list_2 = model.layers[delete_layer+1:]
    # thin_list = thin_list_1.extend(thin_list_2)
    # model.layers = thin_list
    # print("After delete!!!")
    # model.printf()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss

        # 获取模型的classifier模块列表
        classifier_modules_s = model.classifier
        classifier_modules_save_s = model.classifier

        # 移除最后一个元素（softmax层）
        classifier_modules_s = classifier_modules_s[:-1]

        # 将更新后的classifier模块列表设置回模型
        model.classifier = nn.ModuleList(classifier_modules_s)
        logit_output_s = model(images)

        model.classifier = nn.ModuleList(classifier_modules_save_s)
        # logit_output_s = model_s(images)

        output_div_t = -1.0 * args.energy_beta * logit_output_s
        output_logsumexp = torch.logsumexp(output_div_t, dim=1, keepdim=False)
        free_energy = -1.0 * output_logsumexp / args.energy_beta
        align_loss = args.align_lambda * ((free_energy - args.anchor_energy) ** 2).mean()

        loss = ce_loss + align_loss

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg

def train_kd(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch, args,
             snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg

def train_kd_0(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch, args,
               snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        # outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        # kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        # outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        # kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        # outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        # kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)
        loss = ce_loss + kd_lambda * (kd_loss_0)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def train_kd_1(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch, args,
               snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        # outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        # kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        # outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        # kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        # outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        # kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)
        loss = ce_loss + kd_lambda * (kd_loss_1)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def train_kd_2(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch, args,
               snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        # outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        # kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        # outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        # kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        # outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        # kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)
        loss = ce_loss + kd_lambda * (kd_loss_2)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def train_kd_3(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch, args,
               snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        # outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        # kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        # outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        # kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        # outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        # kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)
        loss = ce_loss + kd_lambda * (kd_loss_3)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg

def train_fd_kd_0(train_loader, model, model_0, kd_lambda, criterion, optimizer, epoch, args, snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    # model_1.eval()
    # model_2.eval()
    # model_3.eval()

    model.train()

    end = time.time()

    layers = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]
    gap = [0, 0, 64, 64, 128, 128, 128, 256, 256, 256, 256]
    def get_outputs(inputs_, model_):
        layerss, classifiers = model_.get_all()
        if gap[args.layer] != 0:
            gap_model = CNNModel(gap[args.layer])
            device = torch.device("cuda:{0}".format(args.gpu_num))
            gap_model = gap_model.to(device=device)
            inputs_ = gap_model(inputs_)
        outputs_ = layerss[layers[args.layer]](inputs_)
        for i in range(layers[args.layer] + 1, len(layerss)):
            outputs_ = layerss[i](outputs_)
        outputs_ = outputs_.view(outputs_.size()[0], -1)
        for classifier in classifiers:
            outputs_ = classifier(outputs_)
        return outputs_

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs, outputs_all = model(images)
        ce_loss = criterion(outputs, targets)

        outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        outputs_0 = get_outputs(outputs_all[layers[args.layer]], model_0)
        kd_loss_0 = F.kl_div(torch.log_softmax(outputs_0, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        # outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        # outputs_1 = get_outputs(outputs_all[layers[args.layer]], model_1)
        # kd_loss_1 = F.kl_div(torch.log_softmax(outputs_1, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        # outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        # outputs_2 = get_outputs(outputs_all[layers[args.layer]], model_2)
        # kd_loss_2 = F.kl_div(torch.log_softmax(outputs_2, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        # outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        # outputs_3 = get_outputs(outputs_all[layers[args.layer]], model_3)
        # kd_loss_3 = F.kl_div(torch.log_softmax(outputs_3, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # print('loss', kd_loss_0.item(), kd_loss_1.item(), kd_loss_2.item(), kd_loss_3.item())
        loss = ce_loss + kd_lambda * (kd_loss_0)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg

def train_fd_kd_1(train_loader, model, model_1, kd_lambda, criterion, optimizer, epoch, args, snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model_1.eval()

    model.train()

    end = time.time()

    layers = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]
    gap = [0, 0, 64, 64, 128, 128, 128, 256, 256, 256, 256]
    def get_outputs(inputs_, model_):
        layerss, classifiers = model_.get_all()
        if gap[args.layer] != 0:
            gap_model = CNNModel(gap[args.layer])
            device = torch.device("cuda:{0}".format(args.gpu_num))
            gap_model = gap_model.to(device=device)
            inputs_ = gap_model(inputs_)
        outputs_ = layerss[layers[args.layer]](inputs_)
        for i in range(layers[args.layer] + 1, len(layerss)):
            outputs_ = layerss[i](outputs_)
        outputs_ = outputs_.view(outputs_.size()[0], -1)
        for classifier in classifiers:
            outputs_ = classifier(outputs_)
        return outputs_

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs, outputs_all = model(images)
        ce_loss = criterion(outputs, targets)

        outputs_teacher_1 = model_1(images)
        outputs_1 = get_outputs(outputs_all[layers[args.layer]], model_1)
        kd_loss_1 = F.kl_div(torch.log_softmax(outputs_1, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        loss = ce_loss + kd_lambda * (kd_loss_1)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg

def train_fd_kd_2(train_loader, model, model_2, kd_lambda, criterion, optimizer, epoch, args, snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model_2.eval()

    model.train()

    end = time.time()

    layers = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]
    gap = [0, 0, 64, 64, 128, 128, 128, 256, 256, 256, 256]
    def get_outputs(inputs_, model_):
        layerss, classifiers = model_.get_all()
        if gap[args.layer] != 0:
            gap_model = CNNModel(gap[args.layer])
            device = torch.device("cuda:{0}".format(args.gpu_num))
            gap_model = gap_model.to(device=device)
            inputs_ = gap_model(inputs_)
        outputs_ = layerss[layers[args.layer]](inputs_)
        for i in range(layers[args.layer] + 1, len(layerss)):
            outputs_ = layerss[i](outputs_)
        outputs_ = outputs_.view(outputs_.size()[0], -1)
        for classifier in classifiers:
            outputs_ = classifier(outputs_)
        return outputs_

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs, outputs_all = model(images)
        ce_loss = criterion(outputs, targets)

        outputs_teacher_2 = model_2(images)
        outputs_2 = get_outputs(outputs_all[layers[args.layer]], model_2)
        kd_loss_2 = F.kl_div(torch.log_softmax(outputs_2, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        loss = ce_loss + kd_lambda * (kd_loss_2)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg

def train_fd_kd_3(train_loader, model, model_3, kd_lambda, criterion, optimizer, epoch, args, snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')

    model_3.eval()

    model.train()

    end = time.time()

    layers = [3, 7, 10, 14, 17, 20, 24, 27, 30, 34, 37]
    gap = [0, 0, 64, 64, 128, 128, 128, 256, 256, 256, 256]
    def get_outputs(inputs_, model_):
        layerss, classifiers = model_.get_all()
        if gap[args.layer] != 0:
            gap_model = CNNModel(gap[args.layer])
            device = torch.device("cuda:{0}".format(args.gpu_num))
            gap_model = gap_model.to(device=device)
            inputs_ = gap_model(inputs_)
        outputs_ = layerss[layers[args.layer]](inputs_)
        for i in range(layers[args.layer] + 1, len(layerss)):
            outputs_ = layerss[i](outputs_)
        outputs_ = outputs_.view(outputs_.size()[0], -1)
        for classifier in classifiers:
            outputs_ = classifier(outputs_)
        return outputs_

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs, outputs_all = model(images)
        ce_loss = criterion(outputs, targets)

        outputs_teacher_3 = model_3(images)
        outputs_3 = get_outputs(outputs_all[layers[args.layer]], model_3)
        kd_loss_3 = F.kl_div(torch.log_softmax(outputs_3, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        loss = ce_loss + kd_lambda * (kd_loss_3)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg

def train_kd_0_1(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch, args,
                 snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        # outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        # kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        # outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        # kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)
        loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def train_kd_0_2(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch, args,
                 snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        # outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        # kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        # outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        # kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)
        loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_2)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def train_kd_0_3(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch, args,
                 snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        # outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        # kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        # outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        # kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)
        loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_3)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def train_kd_0_1_2(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch,
                   args, snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        # outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        # kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)
        loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def train_kd_0_1_3(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch,
                   args, snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        # outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        # kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)
        loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_3)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def train_kd_0_2_3(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch,
                   args, snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        # outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        # kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)
        loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_2 + kd_loss_3)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def train_kd_1_2(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch, args,
                 snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        # outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        # kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        # outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        # kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)
        loss = ce_loss + kd_lambda * (kd_loss_1 + kd_loss_2)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def train_kd_1_3(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch, args,
                 snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        # outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        # kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        # outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        # kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)
        loss = ce_loss + kd_lambda * (kd_loss_1 + kd_loss_3)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def train_kd_1_2_3(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch,
                   args, snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        # outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        # kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)
        loss = ce_loss + kd_lambda * (kd_loss_1 + kd_loss_2 + kd_loss_3)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def train_kd_2_3(train_loader, model, model_0, model_1, model_2, model_3, kd_lambda, criterion, optimizer, epoch, args,
                 snapshot, name, fisher_estimation_sample_size=128):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    # temperature = 10
    # knowledge_distillation_loss = nn.KLDivLoss(reduction='batchmean')
    # progress = ProgressMeter(
    #     len(train_loader),
    #     [batch_time, data_time, losses, top1, top5],
    #     prefix="Epoch: [{}]".format(epoch))

    # ewc = EWC(model, args.cuda)
    model_0.eval()
    model_1.eval()
    model_2.eval()
    model_3.eval()

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        # outputs_teacher_0 = model_0(images)
        # a_0 = F.softmax(outputs/temperature,dim=1)
        # b_0 = F.softmax(outputs_teacher_0/temperature,dim=1)
        # kd_loss_0 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_0, dim=1))

        # outputs_teacher_1 = model_1(images)
        # a_1 = F.softmax(outputs/temperature,dim=1)
        # b_1 = F.softmax(outputs_teacher_1/temperature,dim=1)
        # kd_loss_1 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_1, dim=1))

        outputs_teacher_2 = model_2(images)
        # a_2 = F.softmax(outputs/temperature,dim=1)
        # b_2 = F.softmax(outputs_teacher_2/temperature,dim=1)
        kd_loss_2 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_2, dim=1))

        outputs_teacher_3 = model_3(images)
        # a_3 = F.softmax(outputs/temperature,dim=1)
        # b_3 = F.softmax(outputs_teacher_3/temperature,dim=1)
        kd_loss_3 = F.kl_div(torch.log_softmax(outputs, dim=1), torch.softmax(outputs_teacher_3, dim=1))

        # ewc_loss = ewc.ewc_loss(args.cuda)
        # loss = ce_loss + ewc_loss*1000
        # loss = ce_loss + kd_lambda * (kd_loss_0 + kd_loss_1 + kd_loss_2 + kd_loss_3)
        loss = ce_loss + kd_lambda * (kd_loss_2 + kd_loss_3)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        # if i % args.print_freq == 0:
        #     progress.display(i)

    # print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n',)

    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print('Training Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def test_ewc(test_loader, model, criterion, optimizer, epoch, args, snapshot, name, fisher_estimation_sample_size=128):
    # print(len(test_loader.dataset))
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    acc = 0.0
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, top1, top5],
        prefix="Epoch: [{}](test)".format(epoch))

    # ewc = EWC(model, args.cuda)

    model.eval()

    end = time.time()

    with torch.no_grad():

        for i, (images, targets) in enumerate(test_loader):

            data_time.update(time.time() - end)

            if args.cuda:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            # thin_list = nn.ModuleList()
            # thin_list_1 = model.layers[0:3]
            # thin_list_2 = model.layers[4:]
            # thin_list = thin_list_1.extend(thin_list_2)
            # model.layers = thin_list
            # model.printf()

            outputs = model(images)
            ce_loss = criterion(outputs, targets)
            # ewc_loss = ewc.ewc_loss(args.cuda)
            # loss = ce_loss + ewc_loss*1000
            loss = ce_loss

            # loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            pred = outputs.data.max(1, keepdim=True)[1]
            acc += pred.eq(targets.data.view_as(pred)).sum()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)

        progress.display(len(test_loader))

    # print('=> Estimating diagonals of the fisher information matrix...',flush=True, end='\n',)
    # #os.system("nvidia-smi")
    # #train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    # print(' Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write('Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(
        losses.sum / len(test_loader.dataset)) + ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg

def test_ewc_ts(test_loader, model_s, model_t, criterion, optimizer, epoch, args, snapshot, name, fisher_estimation_sample_size=128):
    # print(len(test_loader.dataset))
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    acc = 0.0
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, top1, top5],
        prefix="Epoch: [{}](test)".format(epoch))

    # ewc = EWC(model, args.cuda)

    model_s.eval()
    model_t.eval()

    end = time.time()

    with torch.no_grad():

        for i, (images, targets) in enumerate(test_loader):

            data_time.update(time.time() - end)

            if args.cuda:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            outputs_s = model_s(images)
            outputs_t = model_t(images)

            # 获取模型的classifier模块列表
            classifier_modules_s = model_s.classifier
            classifier_modules_save_s = model_s.classifier

            # 移除最后一个元素（softmax层）
            classifier_modules_s = classifier_modules_s[:-1]

            # 将更新后的classifier模块列表设置回模型
            model_s.classifier = nn.ModuleList(classifier_modules_s)
            logit_output_s = model_s(images)

            model_s.classifier = nn.ModuleList(classifier_modules_save_s)
            # logit_output_s = model_s(images)


            # 获取模型的classifier模块列表
            classifier_modules_t = model_t.classifier
            classifier_modules_save_t = model_t.classifier

            # 移除最后一个元素（softmax层）
            classifier_modules_t = classifier_modules_t[:-1]

            # 将更新后的classifier模块列表设置回模型
            model_t.classifier = nn.ModuleList(classifier_modules_t)
            logit_output_t = model_t(images)

            model_t.classifier = nn.ModuleList(classifier_modules_save_t)
            # logit_output_t = model_t(images)

            # eg_output_s = torch.logsumexp(logit_output_s / 1.0, axis=-1)
            # eg_output_t = torch.logsumexp(logit_output_t / 1.0, axis=-1)

            logit_stack = torch.stack((logit_output_s, logit_output_t))
            outputs_stack = torch.stack((outputs_s, outputs_t))
            eg_output = torch.logsumexp(logit_stack / 1.0, axis=-1)
            _, max_eg_model = torch.max(eg_output, 0)
            output_all = outputs_stack.permute(1, 0, 2).contiguous().view(-1, 100)
            max_eg_model_all = max_eg_model + torch.tensor(
                [i * 2 for i in range(max_eg_model.size(0))]).to(max_eg_model.device)
            outputs = output_all[max_eg_model_all]

            # if eg_output_s > eg_output_t:
            #     outputs = outputs_s
            # else:
            #     outputs = outputs_t

            ce_loss = criterion(outputs, targets)
            # ewc_loss = ewc.ewc_loss(args.cuda)
            # loss = ce_loss + ewc_loss*1000
            loss = ce_loss

            # loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            pred = outputs.data.max(1, keepdim=True)[1]
            acc += pred.eq(targets.data.view_as(pred)).sum()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)

        progress.display(len(test_loader))

    # print('=> Estimating diagonals of the fisher information matrix...',flush=True, end='\n',)
    # #os.system("nvidia-smi")
    # #train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    # print(' Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write('Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(
        losses.sum / len(test_loader.dataset)) + ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg

def test_ewc_t4(test_loader, model_t_0, model_t_1, model_t_2, model_t_3, criterion, optimizer, epoch, args, snapshot, name, fisher_estimation_sample_size=128):
    # print(len(test_loader.dataset))
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    acc = 0.0
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, top1, top5],
        prefix="Epoch: [{}](test)".format(epoch))

    # ewc = EWC(model, args.cuda)

    model_t_0.eval()
    model_t_1.eval()
    model_t_2.eval()
    model_t_3.eval()

    end = time.time()

    with torch.no_grad():

        for i, (images, targets) in enumerate(test_loader):

            data_time.update(time.time() - end)

            if args.cuda:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            outputs_t_0 = model_t_0(images)
            outputs_t_1 = model_t_1(images)
            outputs_t_2 = model_t_2(images)
            outputs_t_3 = model_t_3(images)

            # 获取t0模型的classifier模块列表
            classifier_modules_t_0 = model_t_0.classifier
            classifier_modules_save_t_0 = model_t_0.classifier

            # 移除最后一个元素（softmax层）
            classifier_modules_t_0 = classifier_modules_t_0[:-1]

            # 将更新后的classifier模块列表设置回模型
            model_t_0.classifier = nn.ModuleList(classifier_modules_t_0)
            logit_output_t_0 = model_t_0(images)

            model_t_0.classifier = nn.ModuleList(classifier_modules_save_t_0)
            # logit_output_t_0 = model_t_0(images)



            # 获取t1模型的classifier模块列表
            classifier_modules_t_1 = model_t_1.classifier
            classifier_modules_save_t_1 = model_t_1.classifier

            # 移除最后一个元素（softmax层）
            classifier_modules_t_1 = classifier_modules_t_1[:-1]

            # 将更新后的classifier模块列表设置回模型
            model_t_1.classifier = nn.ModuleList(classifier_modules_t_1)
            logit_output_t_1 = model_t_1(images)

            model_t_1.classifier = nn.ModuleList(classifier_modules_save_t_1)
            # logit_output_t_1 = model_t_1(images)



            # 获取t2模型的classifier模块列表
            classifier_modules_t_2 = model_t_2.classifier
            classifier_modules_save_t_2 = model_t_2.classifier

            # 移除最后一个元素（softmax层）
            classifier_modules_t_2 = classifier_modules_t_2[:-1]

            # 将更新后的classifier模块列表设置回模型
            model_t_2.classifier = nn.ModuleList(classifier_modules_t_2)
            logit_output_t_2 = model_t_2(images)

            model_t_2.classifier = nn.ModuleList(classifier_modules_save_t_2)
            # logit_output_t_2 = model_t_2(images)



            # 获取t3模型的classifier模块列表
            classifier_modules_t_3 = model_t_3.classifier
            classifier_modules_save_t_3 = model_t_3.classifier

            # 移除最后一个元素（softmax层）
            classifier_modules_t_3 = classifier_modules_t_3[:-1]

            # 将更新后的classifier模块列表设置回模型
            model_t_3.classifier = nn.ModuleList(classifier_modules_t_3)
            logit_output_t_3 = model_t_3(images)

            model_t_3.classifier = nn.ModuleList(classifier_modules_save_t_3)
            # logit_output_t_3 = model_t_3(images)

            logit_stack = torch.stack((logit_output_t_0, logit_output_t_1, logit_output_t_2, logit_output_t_3))
            outputs_stack = torch.stack((outputs_t_0, outputs_t_1, outputs_t_2, outputs_t_3))
            eg_output = torch.logsumexp(logit_stack / 1.0, axis=-1)
            _, max_eg_model = torch.max(eg_output, 0)
            output_all = outputs_stack.permute(1, 0, 2).contiguous().view(-1, 100)
            max_eg_model_all = max_eg_model + torch.tensor(
                [i * 4 for i in range(max_eg_model.size(0))]).to(max_eg_model.device)
            outputs = output_all[max_eg_model_all]

            ce_loss = criterion(outputs, targets)
            # ewc_loss = ewc.ewc_loss(args.cuda)
            # loss = ce_loss + ewc_loss*1000
            loss = ce_loss

            # loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            pred = outputs.data.max(1, keepdim=True)[1]
            acc += pred.eq(targets.data.view_as(pred)).sum()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)

        progress.display(len(test_loader))

    # print('=> Estimating diagonals of the fisher information matrix...',flush=True, end='\n',)
    # #os.system("nvidia-smi")
    # #train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    # print(' Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write('Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(
        losses.sum / len(test_loader.dataset)) + ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg

def test_ewc_t4_s(test_loader, model_t_0, model_t_1, model_t_2, model_t_3, model_s_0, criterion, optimizer, epoch, args, snapshot, name, fisher_estimation_sample_size=128):
    # print(len(test_loader.dataset))
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    acc = 0.0
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, top1, top5],
        prefix="Epoch: [{}](test)".format(epoch))

    # ewc = EWC(model, args.cuda)

    model_t_0.eval()
    model_t_1.eval()
    model_t_2.eval()
    model_t_3.eval()
    model_s_0.eval()

    end = time.time()

    with torch.no_grad():

        for i, (images, targets) in enumerate(test_loader):

            data_time.update(time.time() - end)

            if args.cuda:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            outputs_t_0 = model_t_0(images)
            outputs_t_1 = model_t_1(images)
            outputs_t_2 = model_t_2(images)
            outputs_t_3 = model_t_3(images)
            outputs_s_0 = model_s_0(images)


            # 获取t0模型的classifier模块列表
            classifier_modules_t_0 = model_t_0.classifier
            classifier_modules_save_t_0 = model_t_0.classifier

            # 移除最后一个元素（softmax层）
            classifier_modules_t_0 = classifier_modules_t_0[:-1]

            # 将更新后的classifier模块列表设置回模型
            model_t_0.classifier = nn.ModuleList(classifier_modules_t_0)
            logit_output_t_0 = model_t_0(images)

            model_t_0.classifier = nn.ModuleList(classifier_modules_save_t_0)
            # logit_output_t_0 = model_t_0(images)



            # 获取t1模型的classifier模块列表
            classifier_modules_t_1 = model_t_1.classifier
            classifier_modules_save_t_1 = model_t_1.classifier

            # 移除最后一个元素（softmax层）
            classifier_modules_t_1 = classifier_modules_t_1[:-1]

            # 将更新后的classifier模块列表设置回模型
            model_t_1.classifier = nn.ModuleList(classifier_modules_t_1)
            logit_output_t_1 = model_t_1(images)

            model_t_1.classifier = nn.ModuleList(classifier_modules_save_t_1)
            # logit_output_t_1 = model_t_1(images)



            # 获取t2模型的classifier模块列表
            classifier_modules_t_2 = model_t_2.classifier
            classifier_modules_save_t_2 = model_t_2.classifier

            # 移除最后一个元素（softmax层）
            classifier_modules_t_2 = classifier_modules_t_2[:-1]

            # 将更新后的classifier模块列表设置回模型
            model_t_2.classifier = nn.ModuleList(classifier_modules_t_2)
            logit_output_t_2 = model_t_2(images)

            model_t_2.classifier = nn.ModuleList(classifier_modules_save_t_2)
            # logit_output_t_2 = model_t_2(images)



            # 获取t3模型的classifier模块列表
            classifier_modules_t_3 = model_t_3.classifier
            classifier_modules_save_t_3 = model_t_3.classifier

            # 移除最后一个元素（softmax层）
            classifier_modules_t_3 = classifier_modules_t_3[:-1]

            # 将更新后的classifier模块列表设置回模型
            model_t_3.classifier = nn.ModuleList(classifier_modules_t_3)
            logit_output_t_3 = model_t_3(images)

            model_t_3.classifier = nn.ModuleList(classifier_modules_save_t_3)
            # logit_output_t_3 = model_t_3(images)

            logit_stack = torch.stack((logit_output_t_0, logit_output_t_1, logit_output_t_2, logit_output_t_3))
            outputs_stack = torch.stack((outputs_t_0, outputs_t_1, outputs_t_2, outputs_t_3, outputs_s_0))
            eg_output = torch.logsumexp(logit_stack / 1.0, axis=-1)
            # _, max_eg_model = torch.max(eg_output, 0)

            top_values, top_indices = torch.topk(eg_output, dim=0, k=2)

            # 获取次大值和其索引
            max_value = top_values[0]
            max_index = top_indices[0]

            second_max_value = top_values[1]
            second_max_index = top_indices[1]

            # 计算最大值和第二大值之间的差异
            diff = max_value - second_max_value

            # 根据差异判断最终索引
            max_eg_model = torch.where(diff >= 6.398, max_index, torch.tensor(4).cuda())

            output_all = outputs_stack.permute(1, 0, 2).contiguous().view(-1, 100)
            max_eg_model_all = max_eg_model + torch.tensor(
                [i * 5 for i in range(max_eg_model.size(0))]).to(max_eg_model.device)
            outputs = output_all[max_eg_model_all]

            ce_loss = criterion(outputs, targets)
            # ewc_loss = ewc.ewc_loss(args.cuda)
            # loss = ce_loss + ewc_loss*1000
            loss = ce_loss

            # loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            pred = outputs.data.max(1, keepdim=True)[1]
            acc += pred.eq(targets.data.view_as(pred)).sum()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)

        progress.display(len(test_loader))

    # print('=> Estimating diagonals of the fisher information matrix...',flush=True, end='\n',)
    # #os.system("nvidia-smi")
    # #train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    # print(' Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write('Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(
        losses.sum / len(test_loader.dataset)) + ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg

def test_ewc_t4_s_softmax(test_loader, model_t_0, model_t_1, model_t_2, model_t_3, model_s_0, criterion, optimizer, epoch, args, snapshot, name, fisher_estimation_sample_size=128):
    # print(len(test_loader.dataset))
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    acc = 0.0
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, top1, top5],
        prefix="Epoch: [{}](test)".format(epoch))

    # ewc = EWC(model, args.cuda)

    model_t_0.eval()
    model_t_1.eval()
    model_t_2.eval()
    model_t_3.eval()
    model_s_0.eval()

    end = time.time()

    with torch.no_grad():

        for i, (images, targets) in enumerate(test_loader):

            data_time.update(time.time() - end)

            if args.cuda:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            outputs_t_0 = model_t_0(images)
            outputs_t_1 = model_t_1(images)
            outputs_t_2 = model_t_2(images)
            outputs_t_3 = model_t_3(images)
            outputs_s_0 = model_s_0(images)


            # 获取t0模型的classifier模块列表
            classifier_modules_t_0 = model_t_0.classifier
            classifier_modules_save_t_0 = model_t_0.classifier

            # 移除最后一个元素（softmax层）
            classifier_modules_t_0 = classifier_modules_t_0[:-1]

            # 将更新后的classifier模块列表设置回模型
            model_t_0.classifier = nn.ModuleList(classifier_modules_t_0)
            logit_output_t_0 = model_t_0(images)

            model_t_0.classifier = nn.ModuleList(classifier_modules_save_t_0)
            # logit_output_t_0 = model_t_0(images)



            # 获取t1模型的classifier模块列表
            classifier_modules_t_1 = model_t_1.classifier
            classifier_modules_save_t_1 = model_t_1.classifier

            # 移除最后一个元素（softmax层）
            classifier_modules_t_1 = classifier_modules_t_1[:-1]

            # 将更新后的classifier模块列表设置回模型
            model_t_1.classifier = nn.ModuleList(classifier_modules_t_1)
            logit_output_t_1 = model_t_1(images)

            model_t_1.classifier = nn.ModuleList(classifier_modules_save_t_1)
            # logit_output_t_1 = model_t_1(images)



            # 获取t2模型的classifier模块列表
            classifier_modules_t_2 = model_t_2.classifier
            classifier_modules_save_t_2 = model_t_2.classifier

            # 移除最后一个元素（softmax层）
            classifier_modules_t_2 = classifier_modules_t_2[:-1]

            # 将更新后的classifier模块列表设置回模型
            model_t_2.classifier = nn.ModuleList(classifier_modules_t_2)
            logit_output_t_2 = model_t_2(images)

            model_t_2.classifier = nn.ModuleList(classifier_modules_save_t_2)
            # logit_output_t_2 = model_t_2(images)



            # 获取t3模型的classifier模块列表
            classifier_modules_t_3 = model_t_3.classifier
            classifier_modules_save_t_3 = model_t_3.classifier

            # 移除最后一个元素（softmax层）
            classifier_modules_t_3 = classifier_modules_t_3[:-1]

            # 将更新后的classifier模块列表设置回模型
            model_t_3.classifier = nn.ModuleList(classifier_modules_t_3)
            logit_output_t_3 = model_t_3(images)

            model_t_3.classifier = nn.ModuleList(classifier_modules_save_t_3)
            # logit_output_t_3 = model_t_3(images)

            logit_stack = torch.stack((logit_output_t_0, logit_output_t_1, logit_output_t_2, logit_output_t_3))
            logit_stack_softmax = torch.stack((outputs_t_0, outputs_t_1, outputs_t_2, outputs_t_3))
            outputs_stack = torch.stack((outputs_t_0, outputs_t_1, outputs_t_2, outputs_t_3, outputs_s_0))
            eg_output = torch.logsumexp(logit_stack / 1.0, axis=-1)
            # eg_output_softmax = torch.logsumexp(logit_stack_softmax / 1.0, axis=-1)
            eg_output_softmax, _ = logit_stack_softmax.topk(1, 2, True, True)
            eg_output_softmax = torch.squeeze(eg_output_softmax)
            # _, max_eg_model = torch.max(eg_output, 0)

            top_values, top_indices = torch.topk(eg_output, dim=0, k=2)
            top_values_softmax, top_indices_softmax = torch.topk(eg_output_softmax, dim=0, k=2)

            # 获取次大值和其索引
            max_value = top_values[0]
            max_index = top_indices[0]

            second_max_value = top_values[1]
            second_max_index = top_indices[1]

            # 获取次大值和其索引
            max_value_softmax = top_values_softmax[0]
            max_index_softmax = top_indices_softmax[0]

            second_max_value_softmax = top_values_softmax[1]
            second_max_index_softmax = top_indices_softmax[1]

            # 计算最大值和第二大值之间的差异
            diff = max_value - second_max_value

            # 计算最大值和第二大值之间的差异
            diff_softmax = max_value_softmax - second_max_value_softmax

            # 根据差异判断最终索引
            # max_eg_model = torch.where(diff >= 6.398, max_index, torch.tensor(4).cuda())
            # max_eg_model = torch.where((diff >= 6.398) & (diff_softmax >= 0.07), max_index, torch.tensor(4).cuda())
            # max_eg_model = torch.where((diff >= 5) & (diff_softmax >= 0.001), max_index, torch.tensor(4).cuda())
            max_eg_model = torch.where((diff_softmax >= 0.01) & (diff >= 5), max_index, torch.tensor(4).cuda())
            # max_eg_model = torch.where((diff_softmax >= 0.62), max_index, torch.tensor(4).cuda())
            # max_eg_model = torch.where((diff >= 5), max_index, torch.tensor(4).cuda())


            output_all = outputs_stack.permute(1, 0, 2).contiguous().view(-1, 100)
            max_eg_model_all = max_eg_model + torch.tensor(
                [i * 5 for i in range(max_eg_model.size(0))]).to(max_eg_model.device)
            outputs = output_all[max_eg_model_all]

            ce_loss = criterion(outputs, targets)
            # ewc_loss = ewc.ewc_loss(args.cuda)
            # loss = ce_loss + ewc_loss*1000
            loss = ce_loss

            # loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            pred = outputs.data.max(1, keepdim=True)[1]
            acc += pred.eq(targets.data.view_as(pred)).sum()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)

        progress.display(len(test_loader))

    # print('=> Estimating diagonals of the fisher information matrix...',flush=True, end='\n',)
    # #os.system("nvidia-smi")
    # #train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    # print(' Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write('Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(
        losses.sum / len(test_loader.dataset)) + ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg

def train_ewc_vgg(train_loader, model, criterion, optimizer, epoch, args, snapshot, name,
                  fisher_estimation_sample_size=128):
    # print(len(train_loader.dataset))
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses, top1, top5],
        prefix="Epoch: [{}]".format(epoch))

    ewc = EWC_vgg(model, args.cuda)

    model.train()

    end = time.time()

    for i, (images, targets) in enumerate(train_loader):

        data_time.update(time.time() - end)

        if args.cuda:
            images = images.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

        outputs = model(images)
        ce_loss = criterion(outputs, targets)

        ewc_loss = ewc.ewc_loss(args.cuda)
        loss = ce_loss + ewc_loss * 1000
        # loss = criterion(outputs, targets)

        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        losses.update(loss.item(), images.size(0))
        top1.update(acc1[0], images.size(0))
        top5.update(acc5[0], images.size(0))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            progress.display(i)

    print('=> Estimating diagonals of the fisher information matrix...', flush=True, end='\n', )
    # os.system("nvidia-smi")
    # train_sample_loader = old_task

    ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    print(' Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write(
        'Training Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(losses.sum / len(train_loader.dataset)) +
        ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def test_ewc_vgg(test_loader, model, criterion, optimizer, epoch, args, snapshot, name,
                 fisher_estimation_sample_size=128):
    # print(len(test_loader.dataset))
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4e')
    top1 = AverageMeter('Acc@1', ':6.2f')
    top5 = AverageMeter('Acc@5', ':6.2f')
    acc = 0.0
    progress = ProgressMeter(
        len(test_loader),
        [batch_time, data_time, top1, top5],
        prefix="Epoch: [{}](test)".format(epoch))

    ewc = EWC_vgg(model, args.cuda)

    model.eval()

    end = time.time()

    with torch.no_grad():

        for i, (images, targets) in enumerate(test_loader):

            data_time.update(time.time() - end)

            if args.cuda:
                images = images.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            outputs = model(images)
            ce_loss = criterion(outputs, targets)
            ewc_loss = ewc.ewc_loss(args.cuda)
            loss = ce_loss + ewc_loss * 1000
            # loss = criterion(outputs, targets)

            acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
            losses.update(loss.item(), images.size(0))
            top1.update(acc1[0], images.size(0))
            top5.update(acc5[0], images.size(0))

            pred = outputs.data.max(1, keepdim=True)[1]
            acc += pred.eq(targets.data.view_as(pred)).sum()

            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()

            batch_time.update(time.time() - end)
            end = time.time()

            # if i % args.print_freq == 0:
            #     progress.display(i)

        progress.display(len(test_loader))

    # print('=> Estimating diagonals of the fisher information matrix...',flush=True, end='\n',)
    # #os.system("nvidia-smi")
    # #train_sample_loader = old_task

    # ewc.consolidate(ewc.estimate_fisher(train_loader, fisher_estimation_sample_size))
    # print(' Done!')

    RESULT_PATH = os.path.join(snapshot, name)
    file = open(RESULT_PATH, 'a')
    file.write('Epoch: ' + str(epoch) + ' Test set: Average loss: ' + str(
        losses.sum / len(test_loader.dataset)) + ', Accuracy: ' + str(top1.avg) + '\n')

    return losses.avg, top1.avg


def sig(test_loader, model_one, args, snapshot, name='test', fisher_estimation_sample_size=128):
    print(len(test_loader.dataset))

    RESULT_PATH = os.path.join(snapshot, name)
    # RESULT_PATH = './outputs/VGG_lifelong_classes_five/vgg_comp_acc-{}'.format(name)
    file = open(RESULT_PATH, 'a')

    with torch.no_grad():
        cout = 0
        for i, (images, targets) in enumerate(test_loader):
            # measure data loading time

            if args.cuda:
                images = images.cuda(non_blocking=True)
                # targets = targets.cuda(non_blocking=True)
            targets = targets.tolist()

            # compute output
            outputs_one = model_one(images).tolist()

            print(len(outputs_one[1]))
            for i in range(len(targets)):
                cout = cout + 1
                sigma_one = outputs_one[i][targets[i]]
                file.write('count: ' + str(cout) + ' sigma_one: ' + str(float(sigma_one)) + '\n')

    return sigma_one

# def knowledge_distillation_loss(outputs_student, outputs_teacher, temperature):
#     """
#     知识蒸馏损失函数
#
#     Args:
#         outputs_student: 小型模型的输出
#         outputs_teacher: 大型模型的输出
#         labels: 真实标签
#         temperature: 温度参数，用于软化预测分布
#         alpha: 控制蒸馏损失和交叉熵损失的权重
#
#     Returns:
#         知识蒸馏损失
#     """
#
#     # 对大型模型的输出进行软化处理
#     soft_targets = F.softmax(outputs_teacher / temperature, dim=1)
#
#     # 计算知识蒸馏损失
#     distillation_loss = torch.mean(torch.sum(-soft_targets * F.log_softmax(outputs_student / temperature, dim=1), dim=1))
#
#     # 综合交叉熵损失和知识蒸馏损失
#
#     return distillation_loss