import argparse
import os
import time
import pickle

import logging
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from visdom import Visdom
from torch.optim.lr_scheduler import StepLR

from models.dataset import *
from models.resnet_cbam import *

# create logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# home_dir = os.environ['HOME']

work_dir = os.environ['PWD'][:-19]

# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
parser = argparse.ArgumentParser(description='VacuumDexterousGrasp')
"""
    run$ python -m visdom.server (-port XXXX)    to activate Visdom before train/test
    python -m visdom.server -env_path ~/VacuumGrasp/vacuum-pointnet/NeuralNetwork/assets/logs/
"""
"""
    These parameters should be changed in different projects
"""
port_num = 8031
# python -m visdom.server -port 8031 -env_path ~/Desktop/Dexterous_grasp/NeuralNetwork/data
model_path = work_dir + '/NeuralNetwork/'

parser.add_argument('--model-path', type=str, default=model_path)
path_grasps = work_dir + '/dataset'
path_pcs = work_dir + '/dataset'
parser.add_argument('--grasps-path', type=str, default=path_grasps, help='grasps path')
parser.add_argument('--pcs-path', type=str, default=path_pcs, help='point clouds path')
parser.add_argument('--test-grasps-path', type=str, default=path_grasps + '/test', help='test grasps path')
"""
    in test/continue mode, --load-epoch and --load-mode are mandatory
"""
args = parser.parse_args()
"""
    Default values are enough in most of cases
"""
parser.add_argument('--arch', type=str, default="resnet_linr", choices=['resnet_clss', 'resnet_linr'])
parser.add_argument('--input_channel', type=int, default=1, choices=[1, 3])
parser.add_argument('--epoch', type=int, default=200)
parser.add_argument('--batch-size', type=int, default=128)
parser.add_argument('--lr_adam', type=float, default=0.000005)
parser.add_argument('--log-interval', type=int, default=10)
parser.add_argument('--save-interval', type=int, default=20)   # save model per 10 epochs
parser.add_argument('--num_classes', default=5, type=int, help='number of class for classification')
parser.add_argument('--depth', default=50, type=int, help='layer of CNN')
parser.add_argument('--flg_drop', default=True, type=bool)
parser.add_argument('--r_drop', default=0.5, type=float)

args = parser.parse_args()
# args.device = torch.device('cuda') if torch.cuda.is_available else False

# logger = SummaryWriter(os.path.join('./assets/log/', args.tag))
np.random.seed(int(time.time()))


def get_gpu_tem():
    shell_str = "tem_line=`nvidia-smi | grep %` && tem1=`echo $tem_line | cut -d C -f 1` " \
                "&& tem2=`echo $tem1 | cut -d % -f 2` && echo $tem2"
    result = os.popen(shell_str)
    result_str = result.read()
    tem_str = result_str.split(' ', 3)[2]
    result.close()
    return float(tem_str)


# create model
if args.arch == "resnet_linr":
    model = ResidualNet_Linr('ImageNet',
                             args.input_channel,
                             args.depth,
                             att_type='CBAM',
                             flg_drop=args.flg_drop,
                             r_drop=args.r_drop).cuda()
'''
trained_list = list(model.state_dict().keys())
for i in range(0, len(trained_list)):
    print(i, trained_list[i])
'''

viz = Visdom(env='train_' + model.model_name, port=port_num)

grasps = DexterousVacuumGraspOneViewDataset(
    path_grasps=args.grasps_path,
    path_pcs=args.pcs_path,
    num_classes=args.num_classes,
    ranges_clss=np.array([0.2, 0.5, 0.7, 1.0]),
    flg_shuffle=True,
    flg_normalize=False,
    flg_smooth=True, thresh_max_num=9000,
    flg_cut_value=True, thresh_min_quality=0.3, thresh_max_quality=1.0)

# print(grasps[0])
train_loader = torch.utils.data.DataLoader(
    grasps,
    batch_size=args.batch_size,
    shuffle=True)

test_grasps = DexterousVacuumGraspOneViewDataset(
    path_grasps=args.test_grasps_path,
    path_pcs=args.pcs_path,
    num_classes=args.num_classes,
    ranges_clss=np.array([0.2, 0.5, 0.7, 1.0]),
    flg_shuffle=True,
    flg_normalize=False,
    flg_smooth=True, thresh_max_num=350,
    flg_cut_value=True, thresh_min_quality=0.3, thresh_max_quality=1.0)

test_loader = torch.utils.data.DataLoader(
    test_grasps,
    batch_size=args.batch_size,
    shuffle=True)

''' for linear regression'''
criterion = nn.MSELoss().cuda()
''''''
optimizer_adam = optim.Adam(model.parameters(), lr=args.lr_adam)
scheduler_adam = StepLR(optimizer_adam, step_size=10, gamma=0.9)


viz.line([[0., 0.]], [0.], win='AVE_LOSS',
         opts=dict(title='AVE_LOSS', legend=['ave_loss_train', 'ave_loss_test']))
viz.line([[0., 0.]], [0.], win='AVE_DIS',
         opts=dict(title='AVE_DIS', legend=['ave_dis_train', 'ave_dis_test']))
viz.line([[0., 0.]], [0.], win='RELA_DIS',
         opts=dict(title='RELA_DIS', legend=['ave_rela_dis_train', 'ave_rela_dis_test']))


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
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


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def train(model, loader, epoch):
    model.train()
    torch.set_grad_enabled(True)
    dataset_size = 0
    ave_loss = 0.0
    abs_dis = 0.0
    rel_dis = 0.0
    offset_step = 0

    for batch_idx, (data, score, label) in enumerate(loader):
        dataset_size += data.shape[0]
        #
        _, _, data = data.split([1, 1, 1], dim=1)
        #
        data, score, label = data.cuda(), \
                             (score.view(-1, 1)).cuda(), \
                             label.cuda()
        output = model(data) #
        loss = criterion(output, score)
        optimizer_adam.zero_grad()
        loss.backward()
        optimizer_adam.step()

        offset_step += 1
        '''
        tmp = (output - score).cpu()
        tmp = tmp.detach()
        tmp = tmp.numpy()
        '''
        abs_dis += np.mean(np.abs((output - score).cpu().detach().numpy()))
        rel_dis += np.mean(np.abs(((output - score) / score).cpu().detach().numpy()))
        ave_loss += loss.item()

        # viz.line([[dis, _]], [global_step], win='distance_batch', update='append')
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}\t'.format(
            epoch, batch_idx * args.batch_size, len(loader.dataset),
            100. * batch_idx * args.batch_size / len(loader.dataset), loss.item()))
            '''
            logger.add_scalar('train_loss', loss.cpu().item(),
                    batch_idx + epoch * len(loader))
            '''
        # print("current gpu temprature: ", gpu_temprature)

    scheduler_adam.step()
    ave_loss = ave_loss / float(batch_idx + 1)
    abs_dis = abs_dis / float(batch_idx + 1)
    rel_dis = rel_dis / float(batch_idx + 1)

    return offset_step, ave_loss, abs_dis, rel_dis


def test(model, loader):
    model.eval()
    torch.set_grad_enabled(False)
    dataset_size = 0

    ave_loss = 0.0
    rel_dis = 0.0
    abs_dis = 0.0

    for batch_idx, (data, score, label) in enumerate(loader):
        dataset_size += data.shape[0]
        #
        _, _, data = data.split([1, 1, 1], dim=1)
        #
        data, score, label = data.cuda(), \
                             (score.view(-1, 1)).cuda(), \
                             label.cuda()
        output = model(data) #
        loss = criterion(output, score)
        abs_dis += np.mean(np.abs((output - score).cpu().detach().numpy()))
        rel_dis += np.mean(np.abs(((output - score) / score).cpu().detach().numpy()))
        ave_loss += loss.item()

    ave_loss = ave_loss / float(batch_idx + 1)
    abs_dis = abs_dis / float(batch_idx + 1)
    rel_dis = rel_dis / float(batch_idx + 1)
    return ave_loss, abs_dis, rel_dis


def main():
    global_step = 0
    best_ave_loss = 99999.0
    best_rel_dis = 99999.0

    data_loss_train = np.zeros(args.epoch + 1)
    data_rela_dis_train = np.zeros(args.epoch + 1)
    data_loss_test = np.zeros(args.epoch + 1)
    data_rela_dis_test = np.zeros(args.epoch + 1)

    torch.backends.cudnn.benchmark = True
    for epoch in range(0, args.epoch + 1):
        accumulate_step, loss_train, abs_dis_train, rela_dis_train = \
            train(model, train_loader, epoch)
        global_step += accumulate_step
        print('Train done, ave_dis={}, rel_dis={}'.format(loss_train ** 0.5, rela_dis_train))

        loss_test, abs_dis_test, rela_dis_test = test(model, test_loader)

        viz.line([[loss_train, loss_test]], [epoch], win='AVE_LOSS', update='append')
        viz.line([[abs_dis_train, abs_dis_test]], [epoch], win='AVE_DIS', update='append')
        viz.line([[rela_dis_train, rela_dis_test]], [epoch], win='RELA_DIS', update='append')

        data_loss_train[epoch] = loss_train
        data_rela_dis_train[epoch] = rela_dis_train

        data_loss_test[epoch] = loss_test
        data_rela_dis_test[epoch] = rela_dis_test

        np.save(args.model_path + '/data/data_loss_train.npy', data_loss_train)
        np.save(args.model_path + '/data/data_rela_dis_train.npy', data_rela_dis_train)

        np.save(args.model_path + '/data/data_loss_test.npy', data_loss_test)
        np.save(args.model_path + '/data/data_rela_dis_test.npy', data_rela_dis_test)

        viz.save(['train_' + str(len(train_loader.dataset)) + model.model_name])
        if epoch % args.save_interval == 0:
            path = os.path.join(args.model_path, '{}_s{}_ep{}.model'.format(model.model_name,
                                                                            len(train_loader.dataset),
                                                                            epoch))
            torch.save(model, path)
            print('Save model @ {}'.format(path))
        if best_ave_loss > loss_test and epoch > int(0.5*args.epoch):
            best_ave_loss = 1.0 * loss_test
            path = os.path.join(args.model_path, '{}_best_ave_loss.model'.format(model.model_name))
            torch.save(model, path)
            print('Save model @ {}'.format(path))
        if best_rel_dis > rela_dis_test and epoch > int(0.5*args.epoch):
            best_rel_dis = 1.0 * rela_dis_test
            path = os.path.join(args.model_path, '{}_best_rela_dis.model'.format(model.model_name))
            torch.save(model, path)
            print('Save model @ {}'.format(path))
        ''''''
    viz.save(['train_' + str(len(train_loader.dataset)) + model.model_name])


if __name__ == "__main__":
    main()
