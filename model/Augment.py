import numpy as np
import torch


def one_hot_encoding(X):
    X = [int(x) for x in X]
    n_values = np.max(X) + 1
    b = np.eye(n_values)[X]
    return b

def Aug_data(sample, args):
    aug_str = args.aug
    aug_list = aug_str.split('_')
    aug_func_dic = {
        'jitter':jitter,
        'scale':scaling,
        'permutation':permutation,
        'fourier':fourier,
        'ifourier':ifourier,
        "None":identical
    }
    if('fourier' in aug_list and 'ifourier' not in aug_list):
        aug_func_dic['fourier'] = fourier_cat
    for aug in aug_list:
        sample = aug_func_dic[aug](sample, args)
    return sample

def identical(sample, args):
    return sample

def fourier(sample, args):
    # sample : [batch, channel, time_step]
    train_fft = torch.fft.rfft(sample, dim = -1)
    # train_fft_rep = torch.view_as_real(train_fft).reshape(train_fft.shape[0],train_fft.shape[1],-1)
    train_fft_rep = train_fft
    return train_fft_rep

def fourier_cat(sample,args):
    train_fft = torch.fft.rfft(sample, dim = -1)
    train_fft_rep = torch.view_as_real(train_fft).reshape(train_fft.shape[0],train_fft.shape[1],-1)
    return train_fft_rep

def ifourier(sample, args):
    ts = torch.fft.irfft(sample)
    return ts

def DataTransform(sample, args):
    """Weak and strong augmentations"""
    weak_aug = scaling(sample, args)
    # weak_aug = permutation(sample, max_segments=args.max_seg)
    # strong_aug = jitter(permutation(sample, max_segments=args.max_seg), args.jitter_ratio)
    strong_aug = jitter(scaling(sample, args), args)
    weak_aug, strong_aug = torch.tensor(weak_aug,dtype=torch.float32).to(args.device), torch.tensor(strong_aug,dtype=torch.float32).to(args.device)

    return weak_aug, strong_aug


def jitter(x, args):
    # https://arxiv.org/pdf/1706.00527.pdf
    # return x + np.random.normal(loc=0., scale=sigma, size=x.shape)
    return x + torch.normal(mean=0., std=args.jitter_ratio, size=x.shape).to(args.device)


def scaling(x, args):
    # https://arxiv.org/pdf/1706.00527.pdf
    # factor = np.random.normal(loc=2., scale=sigma, size=(x.shape[0], x.shape[2]))
    # ai = []
    # for i in range(x.shape[1]):
    #     xi = x[:, i, :]
    #     ai.append(np.multiply(xi, factor[:, :])[:, np.newaxis, :])
    # return np.concatenate((ai), axis=1)
    
    # generate random factor tensor
    factor = torch.normal(mean=1., std=args.jitter_scale_ratio, size=(x.shape[0], x.shape[2])).to(args.device)
    # create an empty list to store the results
    ai = []
    # iterate over the second dimension of x
    for i in range(x.shape[1]):
        # get the ith slice of x
        xi = x[:, i, :]
        # element-wise multiplication of xi and factor
        product = torch.mul(xi, factor[:, :]).unsqueeze(1)
        # append the result to the list
        ai.append(product)

    # concatenate the results along the second axis
    result = torch.cat(ai, dim=1)

    return result
def permutation(x, max_segments=5, seg_mode="random"):
    orig_steps = np.arange(x.shape[2])

    num_segs = np.random.randint(1, max_segments, size=(x.shape[0]))

    ret = np.zeros_like(x)
    for i, pat in enumerate(x):
        if num_segs[i] > 1:
            if seg_mode == "random":
                split_points = np.random.choice(x.shape[2] - 2, num_segs[i] - 1, replace=False)
                split_points.sort()
                splits = np.split(orig_steps, split_points)
            else:
                splits = np.array_split(orig_steps, num_segs[i])
            warp = np.concatenate(np.random.permutation(splits)).ravel()
            ret[i] = pat[0,warp]
        else:
            ret[i] = pat
    return torch.from_numpy(ret)


def DataTransform_TD(sample, args):
    """Simplely use the jittering augmentation. Feel free to add more autmentations you want,
    but we noticed that in TF-C framework, the augmentation has litter impact on the final tranfering performance."""
    aug = jitter(sample, args.jitter_ratio)
    return aug


def DataTransform_TD_bank(sample, args):
    """Augmentation bank that includes four augmentations and randomly select one as the positive sample.
    You may use this one the replace the above DataTransform_TD function."""
    aug_1 = jitter(sample, args.jitter_ratio)
    aug_2 = scaling(sample, args.jitter_scale_ratio)
    aug_3 = permutation(sample, max_segments=args.max_seg)
    aug_4 = masking(sample, keepratio=0.9)

    li = np.random.randint(0, 4, size=[sample.shape[0]])
    li_onehot = one_hot_encoding(li)
    aug_1 = aug_1 * li_onehot[:, 0][:, None, None]  # the rows that are not selected are set as zero.
    aug_2 = aug_2 * li_onehot[:, 0][:, None, None]
    aug_3 = aug_3 * li_onehot[:, 0][:, None, None]
    aug_4 = aug_4 * li_onehot[:, 0][:, None, None]
    aug_T = aug_1 + aug_2 + aug_3 + aug_4
    return aug_T

def DataTransform_FD(sample, args):
    """Weak and strong augmentations in Frequency domain """
    aug_1 = remove_frequency(sample, pertub_ratio=0.1)
    aug_2 = add_frequency(sample, pertub_ratio=0.1)
    aug_F = aug_1 + aug_2
    return aug_F

def remove_frequency(x, pertub_ratio=0.0):
    mask = torch.cuda.FloatTensor(x.shape).uniform_() > pertub_ratio # maskout_ratio are False
    mask = mask.to(x.device)
    return x*mask

def add_frequency(x, pertub_ratio=0.0):

    mask = torch.cuda.FloatTensor(x.shape).uniform_() > (1-pertub_ratio) # only pertub_ratio of all values are True
    mask = mask.to(x.device)
    max_amplitude = x.max()
    random_am = torch.rand(mask.shape)*(max_amplitude*0.1)
    pertub_matrix = mask*random_am
    return x+pertub_matrix


def generate_binomial_mask(B, T, D, p=0.5): # p is the ratio of not zero
    return torch.from_numpy(np.random.binomial(1, p, size=(B, T, D))).to(torch.bool)

def masking(x, keepratio=0.9, mask= 'binomial'):
    global mask_id
    nan_mask = ~x.isnan().any(axis=-1)
    x[~nan_mask] = 0
    # x = self.input_fc(x)  # B x T x Ch

    if mask == 'binomial':
        mask_id = generate_binomial_mask(x.size(0), x.size(1), x.size(2), p=keepratio).to(x.device)
    # elif mask == 'continuous':
    #     mask = generate_continuous_mask(x.size(0), x.size(1)).to(x.device)
    # elif mask == 'all_true':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    # elif mask == 'all_false':
    #     mask = x.new_full((x.size(0), x.size(1)), False, dtype=torch.bool)
    # elif mask == 'mask_last':
    #     mask = x.new_full((x.size(0), x.size(1)), True, dtype=torch.bool)
    #     mask[:, -1] = False

    # mask &= nan_mask
    x[~mask_id] = 0
    return x
