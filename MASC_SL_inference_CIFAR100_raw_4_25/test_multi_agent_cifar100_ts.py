import argparse
import time
import datetime
import sys
import copy

sys.path.append('../')
import xlwt
import torch
import torch.nn as nn
import torch.optim as optim
from utils.train import train, test, train_ewc, test_ewc, train_ewc_vgg, test_ewc_vgg, test_ewc_ts
from utils.network_wider_cifar100 import Netwider, Netwider_multi
from utils.cifar100_dataloader import get_permute_cifar100, get_multi_agent_cifar100

import os
from openpyxl import Workbook
import numpy as np

# <editor-fold desc="hyper-parameter">
parser = argparse.ArgumentParser(description='multi_agent_cifar100')

parser.add_argument('--batch_size', type=int, default=32, metavar='N',
                    help='input batch size for training (default: 64)')

parser.add_argument('--epochs', type=int, default=100, metavar='N', help='number of epochs to train (default: 10)')

parser.add_argument('--lr', type=float, default=0.005, metavar='LR', help='learning rate (default: 0.01)')

parser.add_argument('--momentum', type=float, default=0.9, metavar='M', help='SGD momentum (default: 0.9)')

parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')

parser.add_argument('--seed', type=int, default=1, metavar='S', help='random seed (default: 1)')

parser.add_argument('--num_works_multi_task', type=int, default=0, help='number of tasks')

parser.add_argument('--num_works_multi_agent', type=int, default=0, help='number of agent')

parser.add_argument('--num_imgs_per_cat_train', type=int, default=500)

parser.add_argument('--path', type=str, default='./', help='path of base classes')

parser.add_argument('--gpu_num', type=int, default=0)

parser.add_argument('--subtask_classes_num', type=int, default=25)

parser.add_argument('--model_path_s', type=str, default='./', help='path of best model_s')

parser.add_argument('--model_path_t', type=str, default='./', help='path of best model_s')

# </editor-fold>

# <editor-fold desc="warm-up">
args = parser.parse_args()

# print("\n", vars(args))
print("\n".join([f"{key}: {value}" for key, value in vars(args).items()]))

torch.cuda.set_device(args.gpu_num)

args.cuda = not args.no_cuda and torch.cuda.is_available()

torch.manual_seed(args.seed)

if args.cuda:
    torch.cuda.manual_seed(args.seed)

lr_ = args.lr

CIFAR100_TRAIN_MEAN = (0.5070751592371323, 0.48654887331495095, 0.4409178433670343)
CIFAR100_TRAIN_STD = (0.2673342858792401, 0.2564384629170883, 0.27615047132568404)

record_time = str((datetime.datetime.now() + datetime.timedelta(hours=8)).strftime('%Y-%m-%d_%H:%M:%S'))
start_time = time.time()
print(record_time)

RESULT_PATH_VAL = ''


# </editor-fold>

# <editor-fold desc="save_model">
def save_checkpoint(states, is_best, output_dir, filename='checkpoint.pth'):
    if not os.path.isdir(output_dir):
        os.makedirs(output_dir)
        print('making dir: %s' % output_dir)

    torch.save(states, os.path.join(output_dir, filename))

    if is_best and 'state_dict' in states:
        torch.save(states['state_dict'], os.path.join(output_dir, 'model_best.pth'))
# </editor-fold>

def main():
    # <editor-fold desc="data_loading">
    print("Testing multi agents: Data loading...")
    best_acc = 0.0

    # cifar100
    multi_agent_path = args.path
    trainloader_multi_agent, testloader_multi_agent = get_multi_agent_cifar100(args.num_works_multi_task,
                                                                                  args.batch_size,
                                                                                  subtask_classes_num=args.subtask_classes_num,
                                                                                  num_imgs_per_cate=args.num_imgs_per_cat_train,
                                                                                  path=multi_agent_path)

    book = xlwt.Workbook(encoding='utf-8', style_compression=0)
    task_id = args.num_works_multi_task
    agent_id = args.num_works_multi_agent
    # </editor-fold>

    # <editor-fold desc="model_s constructing">
    print("Test multi agent: Model constructing...")
    model_s = Netwider_multi(13)
    # model_s.printf()

    model_t = Netwider_multi(13)
    # model_t.printf()

    if args.cuda:
        model_s = model_s.cuda()
        model_t = model_t.cuda()

    model_path_s = args.model_path_s
    # snapshot_model = model_path_s+'/task_{0}'.format(task_id)
    snapshot_model = model_path_s
    checkpoint_path = os.path.join(snapshot_model, 'checkpoint.pth')
    optimizer = optim.SGD(model_s.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)

    if os.path.isfile(checkpoint_path):
        print("Model student loading success")
        print(checkpoint_path)
        target_device = torch.device("cuda:{0}".format(args.gpu_num))  # 目标 GPU 设备
        torch.cuda.set_device(target_device)
        checkpoint = torch.load(checkpoint_path, map_location=target_device)
        # best_epoch = checkpoint['epoch']
        # best_acc = checkpoint['best_acc']
        # optimizer.load_state_dict(checkpoint['optimizer'])
        model_s.load_state_dict(checkpoint['state_dict'])
    else:
        print("Model student loading fail")
        print(checkpoint_path)

    model_path_t = args.model_path_t
    # snapshot_model = model_path_t+'/task_{0}'.format(task_id)
    snapshot_model = model_path_t
    checkpoint_path = os.path.join(snapshot_model, 'checkpoint.pth')
    optimizer = optim.SGD(model_t.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)

    if os.path.isfile(checkpoint_path):
        print("Model teacher loading success")
        print(checkpoint_path)
        target_device = torch.device("cuda:{0}".format(args.gpu_num))  #
        torch.cuda.set_device(target_device)
        checkpoint = torch.load(checkpoint_path, map_location=target_device)
        model_t.load_state_dict(checkpoint['state_dict'])
    else:
        print("Model teacher loading fail")
        print(checkpoint_path)
    # </editor-fold>

    # <editor-fold desc="log">
    # for task_id in range(args.num_works_multi_task):

    args.lr = lr_

    snapshot = './log_multi_agent/agent_{0}/test_record_{1}/task_{2}'.format(agent_id, record_time, task_id)

    if not os.path.isdir(snapshot):
        print("Building snapshot file: {0}".format(snapshot))
        os.makedirs(snapshot)

    print("Testing multi agents: Task_{0}_multi_agent_{1} begin !".format(task_id, agent_id))

    # train_name = 'train_task_{0}_multi_agent_{1}'.format(task_id, agent_id)
    test_name = 'test_task_{0}_multi_agent_{1}'.format(task_id, agent_id)

    sheet_task = book.add_sheet('task_{0}_multi_agent_{1}'.format(task_id, agent_id), cell_overwrite_ok=True)
    cnt_epoch = 0
    sheet_task.write(cnt_epoch, 0, 'Epoch')
    sheet_task.write(cnt_epoch, 1, 'train_loss')
    sheet_task.write(cnt_epoch, 2, 'train_acc')
    sheet_task.write(cnt_epoch, 3, 'test_loss')
    sheet_task.write(cnt_epoch, 4, 'test_acc')
    cnt_epoch = cnt_epoch + 1
    # </editor-fold>

    # <editor-fold desc="tran_test_save">
    for epoch in range(args.epochs):

        optimizer = optim.SGD(model_s.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=5e-4)

        criterion = nn.CrossEntropyLoss()

        test_loss, test_acc = test_ewc_ts(testloader_multi_agent[task_id](epoch), model_s, model_t, criterion, optimizer, epoch,
                                       args, \
                                       snapshot=snapshot, name=test_name)

        sheet_task.write(cnt_epoch, 0, 'Epoch_{0}'.format(cnt_epoch))
        sheet_task.write(cnt_epoch, 3, test_loss)
        sheet_task.write(cnt_epoch, 4, test_acc.item())

        cnt_epoch = cnt_epoch + 1

    print("Testing multi agents: Task_{0}_multi_agent_{1} finish !".format(task_id, agent_id))
    # </editor-fold>

    # <editor-fold desc="log">
    filepath = snapshot+"test_multi_agent_{0}".format(agent_id)+".xlsx"
    filename = "test_multi_agent_{0}".format(agent_id)+".xlsx"

    if not os.path.exists(filepath):
        # Excel
        workbook = Workbook()
        workbook.save(filepath)
        workbook.save("test_multi_agent_{0}".format(agent_id)+".xlsx")
        print(f"Create Excel: {filepath}")
    else:
        print(f"Excel exist: {filename}")

    book.save(filepath)
    book.save("test_multi_agent_{0}".format(agent_id) + ".xlsx")

    end_time = time.time()
    run_time = end_time - start_time
    hours = int(run_time // 3600)
    minutes = int((run_time % 3600) // 60)
    seconds = int(run_time % 60)

    print(f"runtime: {hours} hours, {minutes} minutes, {seconds} seconds")
    # </editor-fold>


if __name__ == "__main__":
    main()

