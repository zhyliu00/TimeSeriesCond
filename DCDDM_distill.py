import os
import argparse
import torch
import torch.nn as nn
from tqdm import tqdm
from utils import *
from torch.utils.data import TensorDataset, DataLoader
from model import TSmodels
from model import DCDDM_models
import copy
import numpy as np
import warnings
import yaml
import time
from torchreparam import ReparamModule
import warnings
import datetime
warnings.filterwarnings("ignore", category=DeprecationWarning)
from fast_pytorch_kmeans import KMeans

def main(args):

    curr_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    args.curr_time = curr_time
    framework = args.framework
    save_dir = os.path.join(".", "logged_files", framework, args.dataset, args.curr_time)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    args.save_dir = save_dir
    
        
    inputaug = args.inputaug
    args.inputaug_list = list(inputaug.split('_'))

    
    args.curr_time = curr_time
    with open(args.config_filename) as f:
        config = yaml.load(f)
    yaml.dump(dict(vars(args)), open(os.path.join(args.save_dir, 'config.yaml'), 'w'))

    
    args.device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    log_verbose(args,"Config : {}\n----------------".format(config))
    log_verbose(args,"Args : {}\n----------------".format(args.__dict__))
    data_args = config['data']
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
    
    if args.max_experts is not None and args.max_files is not None:
        args.total_experts = args.max_experts * args.max_files
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    data_devided = {}
    data_devided['train'], data_devided['val'], data_devided['test'] = get_ts_data(args.dataset,merge_train_val = True)

    # data sets and data loaders
    data_sets, data_loaders = {}, {}
    for k,v in data_devided.items():
        # if('fourier' not in aug_list):
        data_sets[k] = TensorDataset(v['samples'].detach().to(torch.float32).to(args.device),v['labels'].detach().to(args.device))
        print('{} : X is {}, Y is {}'.format(k, v['samples'].shape, v['labels'].shape))
        data_loaders[k] = DataLoader(data_sets[k], batch_size = args.batch_train, shuffle = True)

    # 储存evaluation的iteration number [0, 100, 200...]
    eval_it_pool = np.arange(0, args.Iteration + 1, args.eval_it).tolist()
    
    # 这里直接把ablation的各种情况考虑到了。我们也可以借鉴一下
    # model_eval_pool是一个模型名字的list 
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)
    
    log_verbose(args,"Eval_it_pool : {}\tmodel_eval_pool : {}".format(eval_it_pool, model_eval_pool))
    
    # organize the real dataset '''
    indices_class = [[] for c in range(args.num_classes)]
    print("BUILDING DATASET")
    
    for i in range(len(data_devided['train']['samples'])):
        # [1, L]
        lab = data_devided['train']['labels'][i].to(torch.long)
        indices_class[lab].append(i)
    
    for c in range(args.num_classes):
        print('class c = %d: %d real images'%(c, len(indices_class[c])))

    images_all = data_devided['train']['samples']
    
    for ch in range(args.channel):
        print('real images channel %d, mean = %.4f, std = %.4f'%(ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))


    def get_images(c, n):  # get random n images from class c
        idx_shuffle = np.random.permutation(indices_class[c])[:n]
        return images_all[idx_shuffle]

    def get_images_clustering(c, n):
        all_imgaes = torch.tensor(data_devided['train']['samples'][indices_class[c]]).to(torch.float).to(args.device)
        temp_model = args.model
        args.model = 'CNNBN'
        net_temp = TSmodels.get_network(args).to(args.device)
        args.model = temp_model
        print(all_imgaes.shape)
        embeddings = net_temp.embed(all_imgaes)
        kmeans = KMeans(n_clusters=args.ipc, mode='euclidean', verbose=1)
        labels = kmeans.fit_predict(embeddings)
        centers = kmeans.centroids
        
        dis_mat_torch = torch.cdist(centers,embeddings,p=2)
        clustered_images = all_imgaes[torch.argmin(dis_mat_torch,dim=1)]
        return clustered_images


    

    channel = args.channel
    time_step = args.time_step
    
    ''' initialize the synthetic data '''
    label_syn = torch.tensor([np.ones(args.ipc,dtype=np.int_)*i for i in range(args.num_classes)], dtype=torch.long, requires_grad=False, device=args.device).view(-1) # [0,0,0, 1,1,1, ..., 9,9,9]
    
    # [ipc*C, channel, H, W]
    image_syn = nn.Parameter(torch.FloatTensor(args.num_classes * args.ipc, channel, time_step).to(args.device))
    print('{} : X is {}, Y is {}'.format('syn', image_syn.shape, label_syn.shape))


    if args.pix_init == 'real':
        print('initialize synthetic data from random real images')
        for c in range(args.num_classes):
            # image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
            ################
            # clustering
            image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images_clustering(c, args.ipc).detach().data
    else:
        image_syn.data.copy_(torch.randn(size=(args.num_classes * args.ipc, channel, time_step), dtype=torch.float))
        print('initialize synthetic data from random noise')

    image_syn_init = copy.deepcopy(image_syn.data)
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    syn_lr = nn.Parameter(torch.FloatTensor(1).to(args.device))
    syn_lr.data = torch.tensor(args.lr_teacher)
    
    if(args.aug == None):
        args.aug = 'None'
    
    expert_dir = os.path.join(args.buffer_path,args.framework, args.dataset)
    expert_dir = os.path.join(expert_dir, args.aug)
    expert_dir = os.path.join(expert_dir, args.model)
    
    log_verbose(args,"Expert Dir : {}".format(expert_dir))
    optimizer_img = torch.optim.SGD([image_syn], lr=args.lr_feat, momentum=0.5)
    optimizer_lr = torch.optim.SGD([syn_lr], lr = args.lr_lr,momentum=0.5)
    # optimizer_img = torch.optim.Adam([image_syn], lr=args.lr_feat)
    # optimizer_lr = torch.optim.Adam([syn_lr], lr = args.lr_lr)
    
    if args.load_all:
        buffer = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            buffer = buffer + torch.load(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))

    else:
        expert_files = []
        n = 0
        while os.path.exists(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n))):
            expert_files.append(os.path.join(expert_dir, "replay_buffer_{}.pt".format(n)))
            n += 1
        if n == 0:
            raise AssertionError("No buffers detected at {}".format(expert_dir))
        file_idx = 0
        expert_idx = 0
        random.shuffle(expert_files)
        if args.max_files is not None:
            expert_files = expert_files[:args.max_files]
        # log_verbose(args,"loading file {}".format(expert_files[file_idx]))
        buffer = torch.load(expert_files[file_idx])
        if args.max_experts is not None:
            buffer = buffer[:args.max_experts]
        random.shuffle(buffer)
    
    best_ACC = {m: -1.0 for m in model_eval_pool}
    best_ACC_std = {m: -1.0 for m in model_eval_pool}
    
    time0 = time.time()
    for it in range(0, args.Iteration + 1):
        save_this_it = False
        
        # Evaluate synthetic data
        if it in eval_it_pool:
            for model_eval in model_eval_pool:
                log_verbose(args,'-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d'%(args.model, model_eval, it))
                
                test_metrics_list = {}
                for it_eval in range(args.num_eval):
                    net_eval = DCDDM_models.Dualmodel(args).to(args.device)
                    # net_eval = TSmodels.get_network(args).to(args.device)

                    # add synthetic data into the dataloaders
                    test_syn, test_label = copy.deepcopy(image_syn).detach().cpu(), copy.deepcopy(label_syn).detach().cpu()
                    data_sets['syn'] = TensorDataset(test_syn,test_label)
                    data_loaders['syn'] = DataLoader(data_sets['syn'], batch_size = args.batch_train, shuffle = True)

                    # eval_data : [N', L']
                    __, test_metrics = evaluate_synset(it_eval = it_eval, net = net_eval, real_data_dict = data_loaders, args=args, use_data = 'val', aug=True)
                    
                    for k in test_metrics.keys():
                        res_k = test_metrics[k].compute()
                        if(k not in test_metrics_list.keys()):
                            test_metrics_list[k] = [res_k]
                        else:
                            test_metrics_list[k].append(res_k)
                        # print("DEBUG : in train_eval, metric {} : {}".format(k, res_k))
                # print(test_metrics_list['Accuracy'])
                for k in test_metrics_list.keys():
                    test_metrics_list[k] =  torch.stack(test_metrics_list[k])
                    log_verbose(args,"Eval {}: {:.5f} +/- {:.5f}".format(k, test_metrics_list[k].mean(), test_metrics_list[k].std()))

                ACC_mean, ACC_std = test_metrics_list['Accuracy'].mean(), test_metrics_list['Accuracy'].std()
                if(ACC_mean>best_ACC[model_eval]):
                    best_ACC[model_eval] = ACC_mean
                    best_ACC_std[model_eval] = ACC_std
                    save_this_it = True
                    
                    test_metrics_list = {}
                    for it_eval in range(args.num_eval):
                        net_eval = DCDDM_models.Dualmodel(args).to(args.device)
                        # net_eval = TSmodels.get_network(args).to(args.device)

                        # eval_data : [N', L']
                        __, test_metrics = evaluate_synset(it_eval = it_eval, net = net_eval, real_data_dict = data_loaders, args=args, use_data = 'test', aug=True)
                        
                        for k in test_metrics.keys():
                            res_k = test_metrics[k].compute()
                            if(k not in test_metrics_list.keys()):
                                test_metrics_list[k] = [res_k]
                            else:
                                test_metrics_list[k].append(res_k)
                    
                    log_verbose(args,"This is the best model so far!")
                    for k in test_metrics_list.keys():
                        test_metrics_list[k] = torch.stack(test_metrics_list[k])
                        log_verbose(args,"Test {}: {:.5f} +/- {:.5f}".format(k, test_metrics_list[k].mean(), test_metrics_list[k].std()))
 
                    
        # save the best
        if it in eval_it_pool and (save_this_it or it % 1000 == 0):
            with torch.no_grad():
                image_save = image_syn.cpu()
                
                torch.save(image_save.cpu(), os.path.join(save_dir, "images_{}.pt".format(it)))
                torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_{}.pt".format(it)))

                if save_this_it:
                    torch.save(image_save.cpu(), os.path.join(save_dir, "images_best.pt".format(it)))
                    torch.save(label_syn.cpu(), os.path.join(save_dir, "labels_best.pt".format(it)))
                    log_verbose(args,"Synthetic Training Epoch : {}, Saved Best!".format(it))
                    
        # Train the Synthetic data
        student_net = DCDDM_models.Dualmodel(args).to(args.device)
        
        student_net = ReparamModule(student_net)
        student_net.train()
        
        num_params = sum([np.prod(p.size()) for p in (student_net.parameters())])
        
        if args.load_all:
            expert_trajectory = buffer[np.random.randint(0, len(buffer))]
        else:
            expert_trajectory = buffer[expert_idx]
            expert_idx += 1
            if expert_idx == len(buffer):
                expert_idx = 0
                file_idx += 1
                if file_idx == len(expert_files):
                    file_idx = 0
                    random.shuffle(expert_files)
                # log_verbose(args,"loading file {}".format(expert_files[file_idx]))
                if args.max_files != 1:
                    del buffer
                    buffer = torch.load(expert_files[file_idx])
                if args.max_experts is not None:
                    buffer = buffer[:args.max_experts]
                random.shuffle(buffer)
            
            
        # expert trajectory use EPOCH, and each epoch contains many iterations
        #############
        # gradually
        start_epoch = np.random.randint(0, args.max_start_epoch)
        # if(args.split_gradually == 0):
        #     now_min = 0
        # else:
        #     now_min = int(it / args.Iteration) * 10
        # now_max = (int(it / args.Iteration) + 1) * 10
        # now_max = min(now_max, args.max_start_epoch)
        # start_epoch = np.random.randint(now_min, now_max)
        ########################
        starting_params = expert_trajectory[start_epoch]
        target_params = expert_trajectory[start_epoch + args.expert_epochs]
        
        target_params = torch.cat([p.data.reshape(-1).to(args.device) for p in target_params], 0)
        
        student_params = torch.cat([p.data.reshape(-1).to(args.device) for p in starting_params], 0).requires_grad_(True)
        starting_params = torch.cat([p.data.reshape(-1).to(args.device) for p in starting_params], 0)
        
        param_loss_list = []
        param_dist_list = []
        

        ###########  Must Reparam the model
        # Update N steps 
        for step in range(args.syn_steps):
            
            syn_X, syn_Y = image_syn, label_syn
            grad = None
            for inputaug in args.inputaug_list:
                syn_X_this = Input_Augmentation(syn_X,inputaug)
                # print('use inputaug: ',inputaug)
                aug =args.aug
                if(aug in ["weak", "strong"] and aug):
                    weak_aug_X, strong_aug_X = Augment.DataTransform(syn_X_this, args)
                if(args.aug =='weak' and aug):
                    syn_X_this = weak_aug_X
                elif(args.aug =='strong' and aug):
                    syn_X_this = strong_aug_X
                elif(aug != None):
                    syn_X_this = Augment.Aug_data(syn_X_this,args)
                else:
                    syn_X_this = syn_X_this

                # print("DEBUG syn_X : {}, syn_Y : {}".format(syn_X.grad, syn_Y))
                y_pred, _, _ = student_net(syn_X_this,  flat_param = student_params)
                CE_loss = criterion(y_pred, syn_Y)
                if(grad == None):
                    grad = torch.autograd.grad(CE_loss, student_params, create_graph=True, allow_unused=True)[0]
                else:
                    grad += torch.autograd.grad(CE_loss, student_params, create_graph=True, allow_unused=True)[0]
            # print(grad)
            student_params = student_params - syn_lr * grad
        
        param_loss = torch.nn.functional.mse_loss(student_params, target_params, reduction="sum")
        param_dist = torch.nn.functional.mse_loss(starting_params, target_params, reduction="sum")
                
        param_loss_list.append(param_loss)
        param_dist_list.append(param_dist)

        param_loss /= num_params
        param_dist /= num_params

        param_loss /= param_dist

        grand_loss = param_loss
                
        # now calculate the contrastive loss
        syn_X, syn_Y = image_syn, label_syn
        aug =args.aug
        if(aug in ["weak", "strong"] and aug):
            weak_aug_X, strong_aug_X = Augment.DataTransform(syn_X, args)
        if(args.aug =='weak' and aug):
            syn_X = weak_aug_X
        elif(args.aug =='strong' and aug):
            syn_X = strong_aug_X
        elif(aug != None):
            syn_X = Augment.Aug_data(syn_X,args)
        else:
            syn_X = syn_X
            
            
        syn_X, syn_Y = image_syn, label_syn
        
        _, t_emb, f_emb = student_net(syn_X,flat_param = student_params)
        # _, t_emb_init, f_emb_init = student_net(image_syn_init,flat_param = target_params)
    
        DM_loss_t = 0.0 
        DM_loss_f = 0.0 
        
        ##################################
        _, t_emb_target, f_emb_target = student_net(syn_X, flat_param = target_params)
        # for i in range(args.num_classes):
        #     DM_loss_t += torch.nn.functional.mse_loss(torch.mean(t_emb_target[i * args.ipc,(i+1)*args.ipc], dim = 0), torch.mean(t_emb[i * args.ipc,(i+1)*args.ipc], dim = 0), reduction="mean")
        #     DM_loss_f += torch.nn.functional.mse_loss(torch.mean(f_emb[i * args.ipc,(i+1)*args.ipc], dim = 0), torch.mean(f_emb_target[i * args.ipc,(i+1)*args.ipc], dim = 0), reduction="mean")
        
        DM_loss_t = torch.nn.functional.mse_loss(t_emb_target, t_emb, reduction="mean")
        DM_loss_f = torch.nn.functional.mse_loss(f_emb, f_emb_target, reduction="mean")
        # DM_loss_t = torch.nn.functional.mse_loss(torch.mean(t_emb_target, dim = 0), torch.mean(t_emb, dim = 0), reduction="mean")
        # DM_loss_f = torch.nn.functional.mse_loss(torch.mean(f_emb, dim = 0), torch.mean(f_emb_target, dim = 0), reduction="mean")
        ##################################\
            
            
        ##################################
        # for c in range(args.num_classes):
        #     this_class_true_img = get_images(c, args.batch_train).detach().data.to(args.device).to(torch.float)
        #     _, t_emb_init, f_emb_init = student_net(this_class_true_img,flat_param = target_params)
        #     DM_loss_t += torch.nn.functional.mse_loss(torch.mean(t_emb[c * args.ipc: (c+1) * args.ipc], dim = 0), torch.mean(t_emb_init, dim = 0), reduction="mean")
        #     DM_loss_f += torch.nn.functional.mse_loss(torch.mean(f_emb[c * args.ipc: (c+1) * args.ipc], dim = 0), torch.mean(f_emb_init, dim = 0), reduction="mean")
        ##################################
        
        # torch.nn.functional.mse_loss(t_emb, t_emb_init, reduction="sum")
        # torch.nn.functional.mse_loss(f_emb, f_emb_init, reduction="sum")
        
        total_loss = args.lambda_DM * (DM_loss_t + DM_loss_f) + grand_loss
        
        
    

        
        optimizer_img.zero_grad()
        optimizer_lr.zero_grad()
        
        total_loss.backward()

        
        optimizer_img.step()
        optimizer_lr.step()
        args.lr_teacher = syn_lr
        
        if(it % 10 == 0):
            # print('DEBUG : inner : {}, logits : {}'.format(inner, logits))
            
            log_verbose(args,"DEBUG : syn_lr: {}".format(syn_lr))
            log_verbose(args,"Synthetic Training Epoch : {}/{}, Matching_loss : {:.4f}, DM_loss_t  : {:.4f}, DM_loss_f : {:.4f}, cost {:.4f}s".format(it, args.Iteration, grand_loss.item(), DM_loss_t.item(), DM_loss_f.item(), time.time()-time0))
            time0 = time.time()
        
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--config_filename', default='config.yml')
    parser.add_argument('--max_files', type=int, default=None, help='number of expert files to read (leave as None unless doing ablations)')
    parser.add_argument('--max_experts', type=int, default=None, help='number of experts to read per file (leave as None unless doing ablations)')
    
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--dataset', type=str, default='pems-bay', help='dataset')
    parser.add_argument('--model', type=str, default='CNNBN', help='model')
    parser.add_argument('--Iteration', type=int, default=5000, help='how many distillation steps to perform')
    parser.add_argument('--eval_it', type=int, default=10, help='how often to evaluate')
    parser.add_argument('--lr_teacher', type=float, default=1e-3, help='initialization for synthetic learning rate')
    parser.add_argument('--lr_feat', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_lr', type=float, default=1e-6, help='learning rate for updating... learning rate')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--buffer_path', type=str, default='./buffers', help='buffer path')
    parser.add_argument('--load_all', action='store_true', help="only use if you can fit all expert trajectories into RAM")
    parser.add_argument('--num_eval', type=int, default=5, help='how many networks to evaluate on')
    parser.add_argument('--eval_mode', type=str, default='S',help='eval_mode, check utils.py for more info')
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data')
    parser.add_argument('--net_mom', type=float, default=0.9, help='momentum')
    parser.add_argument('--pred_len',type = int, default=12, help='prediction length')
    parser.add_argument('--max_start_epoch', type=int, default=25, help='max epoch we can start at')
    parser.add_argument('--expert_epochs', type=int, default=3, help='how many expert epochs the target params are')
    parser.add_argument('--syn_steps', type=int, default=20, help='how many steps to take on synthetic data')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_syn', type=int, default=256, help='should only use this if you run out of VRAM')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--sim', type=str, default='euclidean', help = 'similarity measure of the initial synthetic data')
    parser.add_argument('--framework',type = str, default='MTT', help='name of framework')
    parser.add_argument('--pix_init', type = str, default='real', help = 'way to initialize the syn data')
    parser.add_argument('--aug',type=str, default='scale_jitter')    
    parser.add_argument('--tau', type=float, default=0.1, help='temperature for contrastive loss')
    parser.add_argument('--lambda_DM', type=float, default=1.0, help='weight for contrastive loss')
    parser.add_argument('--dsa', type=str, default='True', choices=['True', 'False'],
                        help='whether to use differentiable Siamese augmentation.')
    parser.add_argument('--dual', type=int, default=1)
    parser.add_argument('--dsa_strategy', type=str, default='color_crop_cutout_flip_scale_rotate',
                        help='differentiable Siamese augmentation strategy')
    # parser.add_argument('--split_gradually',type=int, default=0)
    parser.add_argument('--inputaug',type=str,default='raw')


    parser.add_argument('--zca', action='store_true', help="do ZCA whitening")


    parser.add_argument('--no_aug', type=bool, default=False, help='this turns off diff aug during distillation')

    parser.add_argument('--texture', action='store_true', help="will distill textures instead")
    parser.add_argument('--canvas_size', type=int, default=2, help='size of synthetic canvas')
    parser.add_argument('--canvas_samples', type=int, default=1, help='number of canvas samples per iteration')



    parser.add_argument('--force_save', action='store_true', help='this will save images for 50ipc')
    parser.add_argument('--subset', type=str, default='imagenette', help='ImageNet subset. This only does anything when --dataset=ImageNet')

    args = parser.parse_args()

    
    
    main(args)


 