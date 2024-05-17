import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *
from torch.utils.data import TensorDataset, DataLoader
from model import DCDDM_models
import model.TSmodels as TSmodels
import copy
import numpy as np
import warnings
import yaml
import time

warnings.filterwarnings("ignore", category=DeprecationWarning)
set_seed(7)
torch.set_default_dtype(torch.float)

def main(args):

    with open(args.config_filename) as f:
        config = yaml.load(f)
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    data_args = config['data']
    args.data_args = data_args

    args.num_classes = data_args[args.dataset]['num_classes']
    args.channel = data_args[args.dataset]['channel']
    args.time_step = data_args[args.dataset]['time_step']
    args.aug_para = data_args[args.dataset]['aug_para']
    args.jitter_scale_ratio = args.aug_para['jitter_scale_ratio']
    args.jitter_ratio = args.aug_para['jitter_ratio']
    args.max_seg = args.aug_para['max_seg']
    
    aug_list = args.aug.split('_')
    if('fourier' in aug_list and 'ifourier' not in aug_list):
        args.time_step = int(args.time_step//2+1) * 2
    
    print("Config : {}\n----------------".format(config))
    print("Args : {}\n----------------".format(args.__dict__))
    
    data_loaders = {}

    data_devided = {}
    data_devided['train'], data_devided['val'], data_devided['test'] = get_ts_data(args.dataset,merge_train_val = True)

    # data sets and data loaders
    data_sets, data_loaders = {}, {}
    for k,v in data_devided.items():
        data_sets[k] = TensorDataset(v['samples'].detach().to(torch.float32).to(args.device),v['labels'].detach().to(args.device))
        print('{} : X is {}, Y is {}'.format(k, v['samples'].shape, v['labels'].shape))
        data_loaders[k] = DataLoader(data_sets[k], batch_size = args.batch_train, shuffle = True)

    if(args.aug == None):
        args.aug = 'None'
    args.buffer_path += '/{}'.format(args.framework)
    save_dir = os.path.join(args.buffer_path, args.dataset)
    save_dir = os.path.join(save_dir, args.aug)
    save_dir = os.path.join(save_dir, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    criterion = nn.CrossEntropyLoss().to(args.device)
    trajectories = []
    
    inputaug = args.inputaug
    args.inputaug_list = list(inputaug.split('_'))
    
    
    for it in range(0, args.num_experts):
        print("======================")
        print("Trajectory ID : {}".format(it))
        
        ## Train synthetic data
        # 一定要用随机初始化的模型来训练得到Trajectory.
        # teacher_net = TSmodels.get_network(args).to(args.device)
        teacher_net = DCDDM_models.Dualmodel(args).to(args.device)
        count = sm = sum(p.numel() for p in teacher_net.parameters())

        print("model count is : {}\tfinal_rep is : {}".format(count,teacher_net.final_rep))
        lr = args.lr_teacher
        
        teacher_optim = torch.optim.SGD(teacher_net.parameters(), lr=lr, momentum=args.mom, weight_decay=args.l2)
        # teacher_optim = torch.optim.Adam(teacher_net.parameters(), lr=lr, )
        teacher_optim.zero_grad()
        
        # timestamps 储存每一步的参数
        timestamps = []
        timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])
        
        lr_schedule = [args.train_epochs//2 + 1]
        time0 = time.time()
        for e in range(args.train_epochs):
            
            
            # 先尝试不augmentation的
            train_loss, train_metrics = epoch('train', data={'data_loader':data_loaders['train'], 'model' : args.model}, net = teacher_net, optimizer=teacher_optim, criterion=criterion, args = args, aug = True)
            # print('finish train')
            test_loss, test_metrics = epoch_origin('test', data={'data_loader':data_loaders['test'], 'model' : args.model}, net = teacher_net, optimizer=teacher_optim, criterion=criterion, args = args, aug = False)
            # print('finish test')
            train_str, test_str = 'Epoch: {}\tTrain Loss : {:.6f}\t'.format(e, train_loss), 'Epoch: {}\tTest Loss : {:.6f}\t'.format(e, test_loss)
            for k in train_metrics.keys():
                # print(k, train_metrics[k].compute())
                train_str += '{}: {:.4f}, '.format(k, train_metrics[k].compute())
                test_str += '{}: {:.4f}, '.format(k, test_metrics[k].compute())
            # log_verbose(args,train_str)
            # log_verbose(args,test_str)
            # log_verbose(args,"Cost : {:.3f}s".format(time.time()-time0))
            print(train_str)
            print(test_str)
            print("Cost : {:.3f}s".format(time.time()-time0))
            time0 = time.time()
            

            timestamps.append([p.detach().cpu() for p in teacher_net.parameters()])
            
        trajectories.append(timestamps)
        
        if(len(trajectories) == args.save_interval):
            n = 0
            while(os.path.exists(os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))):
                n+=1
            print("Saving {}".format(os.path.join(save_dir, "replay_buffer_{}.pt".format(n))))
            torch.save(trajectories, os.path.join(save_dir, "replay_buffer_{}.pt".format(n)))
            trajectories = []
            
            
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--config_filename', default='config.yml')
    # parser.add_argument('--dataset', type=str, default='pems-bay', help='dataset')
    parser.add_argument('--dataset', type=str, default='pems-bay', help='dataset')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--model', type=str, default='CNNIN', help='model')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--num_experts', type=int, default=100, help='training iterations')
    parser.add_argument('--lr_teacher', type=float, default=0.002, help='learning rate for updating network parameters')
    parser.add_argument('--mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--l2', type=float, default=0, help='l2 regularization')
    parser.add_argument('--train_epochs', type=int, default=30)
    parser.add_argument('--save_interval', type=int, default=5)
    parser.add_argument('--aug',type=str, default='scale_jitter')    
    parser.add_argument('--framework',type=str, default='DCDDM')
    parser.add_argument('--inputaug',type=str,default='raw')
    
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--subset', type=str, default='imagenette', help='subset')
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    parser.add_argument('--zca', action='store_true')
    parser.add_argument('--decay', action='store_true')
    parser.add_argument('--dual', default=1,type=int)
    args = parser.parse_args()
    main(args)

