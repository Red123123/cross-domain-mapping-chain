import torch.nn.functional as F
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torch.utils.data.sampler import Sampler
from sklearn import preprocessing
from torch import nn
import numpy as np
import torch
import random

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.xavier_uniform_(m.weight, gain=1)
        if m.bias is not None:
            m.bias.data.zero_()
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight, 1.0, 0.02)
        m.bias.data.zero_()
    elif classname.find('Linear') != -1:
        nn.init.xavier_normal_(m.weight)
        if m.bias is not None:
            m.bias.data = torch.ones(m.bias.data.size())
#
# def weights_init(m):
# 	if isinstance(m, nn.Conv2d):
# 		m.weight.data.normal_(0, 0.02)
# 		if m.bias is not None:
# 			m.bias.data.zero_()
#
# 	elif isinstance(m, nn.Linear):
# 		m.weight.data.normal_(0, 0.02)
# 		if m.bias is not None:
# 			m.bias.data.zero_()
#
# 	elif isinstance(m, nn.Conv1d):
# 		m.weight.data.normal_(0, 0.02)
# 		if m.bias is not None:
# 			m.bias.data.zero_()
#
# 	elif isinstance(m, nn.Conv3d):
# 		m.weight.data.normal_(0, 0.02)
# 		if m.bias is not None:
# 			m.bias.data.zero_()
#
# 	elif isinstance(m, nn.BatchNorm1d):
# 		m.weight.data.normal_(0, 0.02)
# 		if m.bias is not None:
# 			m.bias.data.zero_()
#
# 	elif isinstance(m, nn.BatchNorm2d):
# 		m.weight.data.normal_(0, 0.02)
# 		if m.bias is not None:
# 			m.bias.data.zero_()
#
# 	elif isinstance(m, nn.BatchNorm3d):
# 		m.weight.data.normal_(0, 0.02)
# 		if m.bias is not None:
# 			m.bias.data.zero_()


def padWithZeros(X, margin):
    newX = np.zeros((X.shape[0] + 2 * margin, X.shape[1] + 2* margin, X.shape[2]))
    x_offset = margin
    y_offset = margin
    newX[x_offset:X.shape[0] + x_offset, y_offset:X.shape[1] + y_offset, :] = X
    return newX

def data_preprocessing(X):
    [p1, p2] = X.shape
    MEAN = np.mean(X, axis=0)
    STD = np.std(X, axis=0)
    Ma_MEAN = np.ones([p1, p2]) * MEAN
    Ma_STD = np.ones([p1, p2]) * STD
    X_a = (X - Ma_MEAN) / Ma_STD
    return X_a, MEAN, STD

def data_preprocessing_with_mean(X, MEAN, STD):
    [p1, p2] = X.shape
    Ma_MEAN = np.ones([p1, p2]) * MEAN
    Ma_STD = np.ones([p1, p2]) * STD
    X_a = (X - Ma_MEAN) / Ma_STD
    return X_a

def AA(m):
    a = np.sum(m, axis=1)
    zuseracclist = []
    for aai in range(m.shape[0]):
        zuseracclist.append(m[aai][aai] / a[aai])
    b = np.average(zuseracclist)
    return b

class SpectralSpatialData(Dataset):
	def __init__(self, patch, labels, transform=None):
		# self.patch_data = patch_data
		self.patch = patch
		self.labels = labels
		self.transform = transform

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		one_patch = self.patch[index]
		one_label = self.labels[index]
		return one_patch, one_label


# def reduce_loss(loss, reduction='mean'):
#     return loss.mean() if reduction == 'mean' else loss.sum() if reduction == 'sum' else loss

# def linear_combination(x, y, epsilon):
#     return epsilon*x + (1-epsilon)*y
# class LabelSmoothingCrossEntropy(nn.Module):
#     def __init__(self, epsilon: float = 0.1, reduction='mean'):
#         super().__init__()
#         self.epsilon = epsilon
#         self.reduction = reduction
#
#     def forward(self, preds, target):
#         n = preds.size()[-1]
#         log_preds = F.log_softmax(preds, dim=-1)
#         loss = reduce_loss(-log_preds.sum(dim=-1), self.reduction)
#         nll = F.nll_loss(log_preds, target, reduction=self.reduction)
#         return linear_combination(loss / n, nll, self.epsilon)



class Task(object):

    def __init__(self, data, num_classes, shot_num, query_num):
        self.data = data
        self.num_classes = num_classes
        self.support_num = shot_num
        self.query_num = query_num

        class_folders = sorted(list(data))

        class_list = random.sample(class_folders, self.num_classes)

        labels = np.array(range(len(class_list)))

        labels = dict(zip(class_list, labels))

        samples = dict()

        self.support_datas = []
        self.query_datas = []
        self.support_labels = []
        self.query_labels = []
        for c in class_list:
            temp = self.data[c]  # list
            samples[c] = random.sample(temp, len(temp))
            random.shuffle(samples[c])

            self.support_datas += samples[c][:shot_num]
            self.query_datas += samples[c][shot_num:shot_num + query_num]

            self.support_labels += [labels[c] for i in range(shot_num)]
            self.query_labels += [labels[c] for i in range(query_num)]
            # print(self.support_labels)
            # print(self.query_labels)




class ClassBalancedSampler(Sampler):
    ''' Samples 'num_inst' examples each from 'num_cl' pool of examples of size 'num_per_class' '''
    # 参数：
    #   num_per_class: 每个类的样本数量
    #   num_cl: 类别数量
    #   num_inst：support set或query set中的样本数量
    #   shuffle：样本是否乱序
    def __init__(self, num_per_class, num_cl, num_inst,shuffle=True):
        self.num_per_class = num_per_class
        self.num_cl = num_cl
        self.num_inst = num_inst
        self.shuffle = shuffle

    def __iter__(self):
        # return a single list of indices, assuming that items will be grouped by class
        if self.shuffle:
            batch = [[i+j*self.num_inst for i in torch.randperm(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        else:
            batch = [[i+j*self.num_inst for i in range(self.num_inst)[:self.num_per_class]] for j in range(self.num_cl)]
        batch = [item for sublist in batch for item in sublist]

        if self.shuffle:
            random.shuffle(batch)
        return iter(batch)

    def __len__(self):
        return 1

def get_HBKC_data_loader(task, num_per_class=1, split='train',shuffle = False):
    # 参数:
    #   task: 当前任务
    #   num_per_class:每个类别的样本数量，与split有关
    #   split：‘train'或‘test'代表support和querya
    #   shuffle：样本是否乱序
    # 输出：
    #   loader
    dataset = HBKC_dataset(task,split=split)

    if split == 'train':
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.support_num, shuffle=shuffle) # support set
    else:
        sampler = ClassBalancedSampler(num_per_class, task.num_classes, task.query_num, shuffle=shuffle) # query set

    loader = DataLoader(dataset, batch_size=num_per_class*task.num_classes, sampler=sampler)

    return loader

class FewShotDataset(Dataset):
    def __init__(self, task, split='train'):
        self.task = task
        self.split = split
        self.image_datas = self.task.support_datas if self.split == 'train' else self.task.query_datas
        self.labels = self.task.support_labels if self.split == 'train' else self.task.query_labels

    def __len__(self):
        return len(self.image_datas)

    def __getitem__(self, idx):
        raise NotImplementedError("This is an abstract class. Subclass this class for your particular dataset.")

class HBKC_dataset(FewShotDataset):
    def __init__(self, *args, **kwargs):
        super(HBKC_dataset, self).__init__(*args, **kwargs)

    def __getitem__(self, idx):
        image = self.image_datas[idx]
        label = self.labels[idx]
        return image, label

def radiation_noise(data, alpha_range=(0.9, 1.1), beta=1/25):
    alpha = np.random.uniform(*alpha_range)
    noise = np.random.normal(loc=0., scale=1.0, size=data.shape)

    return alpha * data + beta * noise

def TVLoss(A):
    batchsize = A.shape[0]
    # loss = torch.abs(A[:, 1:] - A[:, :A.shape[1] - 1]).sum()    # pow()
    loss = torch.pow(A[:, 1:] - A[:, :A.shape[1] - 1], 2).sum()    # pow()
    return loss/batchsize

def pre_scale(data_all):
    data = data_all.reshape(np.prod(data_all.shape[:2]), np.prod(data_all.shape[2:]))
    data_scaler = preprocessing.scale(data)
    data_scaler = data_scaler.reshape(data_all.shape[0], data_all.shape[1], data_all.shape[2])
    return data_scaler