import torch.fft as fft
import torcheval.metrics
import random
import time
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import tqdm
import model.Augment as Augment
from torch.utils.data import Dataset
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from torchvision import datasets
from torchvision import transforms
from torch.utils.data.sampler import SubsetRandomSampler

def get_loops(ipc):
    # Get the two hyper-parameters of outer-loop and inner-loop.
    # The following values are empirically good.
    if ipc == 1:
        outer_loop, inner_loop = 1, 1
    elif ipc == 10:
        outer_loop, inner_loop = 10, 50
    elif ipc == 20:
        outer_loop, inner_loop = 20, 25
    elif ipc == 30:
        outer_loop, inner_loop = 30, 20
    elif ipc == 40:
        outer_loop, inner_loop = 40, 15
    elif ipc == 50:
        outer_loop, inner_loop = 50, 10
    else:
        outer_loop, inner_loop = 0, 0
        exit('loop hyper-parameters are not defined for %d ipc'%ipc)
    return outer_loop, inner_loop


def get_cifar10_dataloaders(data_dir,
                           batch_size,
                           args,
                           augment=False,
                           random_seed=7,
                           valid_size=0.1,
                           shuffle=True,
                           num_workers=4,
                           pin_memory=False):
    """
    Utility function for loading and returning train and valid
    multi-process iterators over the CIFAR-10 dataset. A sample
    9x9 grid of the images can be optionally displayed.
    If using CUDA, num_workers should be set to 1 and pin_memory to True.
    Params
    ------
    - data_dir: path directory to the dataset.
    - batch_size: how many samples per batch to load.
    - augment: whether to apply the data augmentation scheme
      mentioned in the paper. Only applied on the train split.
    - random_seed: fix seed for reproducibility.
    - valid_size: percentage split of the training set used for
      the validation set. Should be a float in the range [0, 1].
    - shuffle: whether to shuffle the train/validation indices.
    - show_sample: plot 9x9 sample grid of the dataset.
    - num_workers: number of subprocesses to use when loading the dataset.
    - pin_memory: whether to copy tensors into CUDA pinned memory. Set it to
      True if using GPU.
    Returns
    -------
    - train_loader: training set iterator.
    - valid_loader: validation set iterator.
    """
    error_msg = "[!] valid_size should be in the range [0, 1]."
    assert ((valid_size >= 0) and (valid_size <= 1)), error_msg

    normalize = transforms.Normalize(
        mean=[0.4914, 0.4822, 0.4465],
        std=[0.2023, 0.1994, 0.2010],
    )

    # define transforms
    valid_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
    ])
    if augment:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])
    else:
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            normalize,
        ])

    # load the dataset
    train_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=train_transform,
    )

    valid_dataset = datasets.CIFAR10(
        root=data_dir, train=True,
        download=True, transform=valid_transform,
    )
    
    test_dataset = datasets.CIFAR10(
        root=data_dir, train=False,
        download=True, transform=valid_transform,
    )

    num_train = len(train_dataset)
    indices = list(range(num_train))
    split = int(np.floor(valid_size * num_train))

    if shuffle:
        np.random.seed(random_seed)
        np.random.shuffle(indices)

    train_idx, valid_idx = indices[split:], indices[:split]
    train_sampler = SubsetRandomSampler(train_idx)
    valid_sampler = SubsetRandomSampler(valid_idx)

    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, sampler=train_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset, batch_size=batch_size, sampler=valid_sampler,
        num_workers=num_workers, pin_memory=pin_memory,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=pin_memory,
    )

    
    return (train_loader, valid_loader, test_loader)

def log_verbose(args, msg):
    print(msg)
    with open(os.path.join(args.save_dir, "log.txt"), "a") as f:
        f.write(msg + "\n")


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # if you are using multi-GPU.
    np.random.seed(seed)  # Numpy module.
    random.seed(seed)  # Python random module.
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.set_default_dtype(torch.float32)
    return None


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def random_true_label(size, true_series, syn_Y, alpha,args):
    b, n, l = size
    N,_,L = true_series.shape
    # print("DEBUG",n,l,N,L)
    # true_series: [N, 1, L]
    true_series = torch.tensor(true_series,dtype=torch.float).squeeze(1).to(args.device)
    # true_series: [N, L]
    select_node = torch.randperm(N)[:n]
    select_start = torch.randperm(L - l)[:b]
    select_end = select_start + l
    select_label = [true_series[select_node,select_start[i]:select_end[i]] for i in range(b)]
    select_label = torch.stack(select_label,dim=0)
    
    
    # print("DEBUG select label shape : {}, syn_Y shape : {}".format(select_label.shape, syn_Y.shape))
    
    return select_label * alpha + syn_Y * (1 - alpha)
    
def get_time():
    return str(time.strftime("[%Y-%m-%d %H:%M:%S]", time.localtime()))

def Input_Augmentation(X,inputaug):
    if(inputaug == 'raw'):
        return X
    
    if(inputaug == 'LPF'):
        X_f = torch.fft.rfft(X, dim = -1)
        msk = torch.ones_like(X_f)
        msk[:,:,int(msk.shape[-1] * 0.5):] = 0
        X_f = X_f * msk
        X = torch.fft.irfft(X_f, dim = -1)
        return X
    if(inputaug == 'FTPP'):
        perturbation_factor = torch.pi / 4
        fft_data = fft.fft(X)
        phase_perturbation = torch.randn_like(fft_data) * perturbation_factor
        perturbed_fft_data = torch.abs(fft_data) * torch.exp(1j * (torch.angle(fft_data) + phase_perturbation))
        perturbed_data = fft.ifft(perturbed_fft_data).real
        return perturbed_data
    if(inputaug == 'FTMP'):
        perturbation_factor = 0.01
        fft_data = fft.fft(X)
        amplitude = torch.abs(fft_data)
        for c in range(amplitude.shape[1]):
            target = torch.mean(amplitude[:,c,1:])
            # noise = torch.randn_like(amplitude) * perturbation_factor * target
            noise = torch.randn_like(amplitude[:,c,:]) * perturbation_factor * target
            amplitude[:,c,:] = amplitude[:,c,:] + noise
        perturbed_fft_data = amplitude * torch.exp(1j * torch.angle(fft_data))
        perturbed_data = fft.ifft(perturbed_fft_data).real
        return perturbed_data
    


def epoch(mode, data, net, optimizer, criterion, args, aug, texture=False):
    loss_avg, acc_avg, num_exp = 0,0,0

    num_b, num_sample = 0,0
    net = net.to(args.device)
    data_loader,  model_name = data['data_loader'], data['model']
    if('train' in mode):
        net.train()
    else:
        net.eval()
    
    metrics = {
        'Accuracy':torcheval.metrics.MulticlassAccuracy(),
        'Precision':torcheval.metrics.MulticlassPrecision(num_classes = args.num_classes, average = 'macro'),
        'Recall':torcheval.metrics.MulticlassRecall(num_classes = args.num_classes, average = 'macro'),
        'F1':torcheval.metrics.MulticlassF1Score(num_classes = args.num_classes, average = 'macro'),
        'AUROC':torcheval.metrics.MulticlassAUROC(num_classes = args.num_classes, average = 'macro'),
        'AUPRC':torcheval.metrics.MulticlassAUPRC(num_classes = args.num_classes, average = 'macro')
    }
    for met in metrics.values():
        met.to(args.device)
    # log_verbose(args,"".format(A.shape, A_gnd.shape))
    # duplicate the A matrix for batch size

    for i_batch,(X_true,y_true) in enumerate(data_loader):
        # print("X shape : {}, y shape : {}".format(X.shape, y.shape))
        # X = X_true.to(args.device)
        # y = y_true.to(torch.long).to(args.device)
        for inputaug in args.inputaug_list:
            y = y_true.to(torch.long).to(args.device)
            X = X_true.to(args.device)  
                
            X = Input_Augmentation(X,inputaug)
            # print('used input aug, now is {}'.format(inputaug))
            if(aug in ["weak", "strong"] and aug):
                weak_aug_X, strong_aug_X = Augment.DataTransform(X, args)
            if(args.aug =='weak' and aug):
                X = weak_aug_X
            elif(args.aug =='strong' and aug):
                X = strong_aug_X
            elif(aug != None):
                X = Augment.Aug_data(X,args)
            else:
                X = X
                            # print(X.shape, y.shape)
            # if(model_name=='CNNIN'):            
            #     y_pred = net(X).to(args.device)
            # else:
            #     raise NotImplementedError
            if('train' in mode):
                y_pred,t_emb,f_emb = net(X)
            else:
                with torch.no_grad():
                    y_pred,t_emb,f_emb = net(X)
            
            # print(y_pred.shape, y.shape)
            # y_pred = net(X)
            
            loss = criterion(y_pred, y)
            n_b = y.shape[0]
            loss_avg += loss.item()*n_b
            
            # c_loss = cross_domain_loss(args, t_emb, f_emb, net)
            # loss_avg += c_loss.item()*n_b

            # print(y_pred, y)
            prob = F.softmax(y_pred, dim=1)
            # print(prob)
            # print(MSE, RMSE, MAE, MAPE)
            for k, met in metrics.items():
                # if(k == 'AUROC' or k == 'AUPRC'):
                #     prob = torch.tensor(prob, dtype=torch.float)
                met.update(prob, y)
        
            num_b += n_b

            if('train' in mode):
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
    loss_avg /= num_b
    
    # return loss_avg, metrics, loss.item() / num_b, c_loss.item() / num_b
    return loss_avg, metrics

def epoch_origin(mode, data, net, optimizer, criterion, args, aug, texture=False):
    loss_avg, acc_avg, num_exp = 0,0,0

    num_b, num_sample = 0,0
    net = net.to(args.device)
    data_loader,  model_name = data['data_loader'], data['model']
    if('train' in mode):
        net.train()
    else:
        net.eval()
    
    metrics = {
        'Accuracy':torcheval.metrics.MulticlassAccuracy(),
        'Precision':torcheval.metrics.MulticlassPrecision(num_classes = args.num_classes, average = 'macro'),
        'Recall':torcheval.metrics.MulticlassRecall(num_classes = args.num_classes, average = 'macro'),
        'F1':torcheval.metrics.MulticlassF1Score(num_classes = args.num_classes, average = 'macro'),
        'AUROC':torcheval.metrics.MulticlassAUROC(num_classes = args.num_classes, average = 'macro'),
        'AUPRC':torcheval.metrics.MulticlassAUPRC(num_classes = args.num_classes, average = 'macro')
    }
    for met in metrics.values():
        met.to(args.device)
    # log_verbose(args,"".format(A.shape, A_gnd.shape))
    # duplicate the A matrix for batch size

    for i_batch,(X,y) in enumerate(data_loader):
        # print("X shape : {}, y shape : {}".format(X.shape, y.shape))
        y = y.to(torch.long).to(args.device)
        X = X.to(args.device)  
        
        
        if(aug in ["weak", "strong"] and aug):
            weak_aug_X, strong_aug_X = Augment.DataTransform(X, args)
        if(args.aug =='weak' and aug):
            X = weak_aug_X
        elif(args.aug =='strong' and aug):
            X = strong_aug_X
        elif(aug != None):
            X = Augment.Aug_data(X,args)
        else:
            X = X
        # print(X.shape, y.shape)
        # if(model_name=='CNNIN'):            
        #     y_pred = net(X).to(args.device)
        # else:
        #     raise NotImplementedError
        # print(y_pred.shape, y.shape)
        if('train' in mode):
            y_pred,t_emb,f_emb = net(X)
        else:
            with torch.no_grad():
                y_pred,t_emb,f_emb = net(X)        # y_pred = net(X)
        
        loss = criterion(y_pred, y)
        n_b = y.shape[0]
        loss_avg += loss.item()*n_b
        
        # c_loss = cross_domain_loss(args, t_emb, f_emb, net)
        # loss_avg += c_loss.item()*n_b

        # print(y_pred, y)
        prob = F.softmax(y_pred, dim=1)
        # print(prob)
        # print(MSE, RMSE, MAE, MAPE)
        for k, met in metrics.items():
            # if(k == 'AUROC' or k == 'AUPRC'):
            #     prob = torch.tensor(prob, dtype=torch.float)
            met.update(prob, y)
    
        num_b += n_b

        if('train' in mode):
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    loss_avg /= num_b
    
    # return loss_avg, metrics, loss.item() / num_b, c_loss.item() / num_b
    return loss_avg, metrics


def get_eval_pool(eval_mode, model, model_eval):
    # Multiple Architectures
    if(eval_mode == "M"):
        model_eval_pool = ['GWN']
    # Self
    elif(eval_mode == 'S'):
        model_eval_pool=[model]
    else:
        raise NotImplementedError
    return model_eval_pool


def evaluate_synset(it_eval, net, real_data_dict, args, use_data='val', return_loss=False, aug=True):
    
    net = net.to(args.device)

    lr = float(args.lr_teacher)
    Epoch = int(args.epoch_eval_train)
    optimizer = torch.optim.SGD(net.parameters(),lr = lr, momentum=0.9, weight_decay=0.0005 )
    
    criterion = nn.CrossEntropyLoss().to(args.device)
    
    syn_loader = real_data_dict['syn']
    test_loader = real_data_dict[use_data]
    
    time0 = time.time()
    
    metrics_epoch = {}
    
    for ep in range(Epoch+1):
        
        train_loss, train_metrics = epoch('train_evalsyn', data={'data_loader':syn_loader, 'model' : args.model}, net = net, optimizer=optimizer, criterion=criterion, args = args, aug = aug)
        
        train_str = 'Evaluation SynSet id : {}, Time : {:.5f}\t Epoch: {}/{}\tTrain Loss : {:.6f}\t'.format(it_eval,time.time()-time0, ep,Epoch,train_loss)
        for k in train_metrics.keys():
            # print(k, train_metrics[k].compute())
            train_str += '{}: {:.4f}, '.format(k, train_metrics[k].compute())
            if(k in metrics_epoch.items()):
                metrics_epoch[k].append(train_metrics[k].compute())
                    
        if(ep % 100 == 0 or ep == Epoch):
            log_verbose(args,train_str)

        time0=time.time()
            
        
        if(ep == Epoch):
            test_loss, test_metrics = epoch_origin('test_evalsyn', data={'data_loader':test_loader, 'model' : args.model}, net = net, optimizer=None, criterion=criterion, args = args, aug = False)

            test_str = 'Evaluation SynSet id : {}, Epoch: {}\tTest Loss : {:.6f}\t'.format(it_eval, ep ,test_loss)
            for k in test_metrics.keys():
                # print(k, train_metrics[k].compute())
                test_str += '{}: {:.4f}, '.format(k, test_metrics[k].compute())
            log_verbose(args,test_str)

    if return_loss:
        return net, test_metrics, train_loss, test_loss
    else:
        return net, test_metrics



def get_ts_data(dataset, merge_train_val=False):
    train = torch.load('./data/{}/train.pt'.format(dataset))
    test = torch.load('./data/{}/test.pt'.format(dataset))
    val = torch.load('./data/{}/val.pt'.format(dataset))
    
    if(merge_train_val):
        for k, v in train.items():
            train[k] = torch.cat((train[k], val[k]), dim=0)
    if(dataset == 'epilepsy'):
        for k,v in enumerate(train['labels']):
            if(train['labels'][k]!=0):
                train['labels'][k] = 1
        for k,v in enumerate(val['labels']):
            if(val['labels'][k]!=0):
                val['labels'][k] = 1
        for k,v in enumerate(test['labels']):
            if(test['labels'][k]!=0):
                test['labels'][k] = 1
    return train, val, test
