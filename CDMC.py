from __future__ import division
import argparse
import itertools
import os
import torch
from sklearn.neighbors import KNeighborsClassifier
import random
import torch.utils.data as Data
from itertools import cycle
import numpy as np
import mymodels
import utils
import time
import math
from scipy.io import loadmat
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, cohen_kappa_score, recall_score, accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument('--DATASET', type=str, default='Houston2018-Houston2013', help='DATASET')
parser.add_argument('--nclasses', type=int, default=7, help='number of classes')
parser.add_argument('--source_dim', type=int, default=48, help='source dim')
parser.add_argument('--target_dim', type=int, default=144, help='target dim')
parser.add_argument('--feature_dim', type=int, default=400, help='feature_dim of D')
parser.add_argument('--ncycle', type=int, default=5, help='number of classes, K value, recommended value 5')
parser.add_argument('--patch_size', type=int, default=7, help='the size of patch')
parser.add_argument('--lambda1_D', type=float, default=1.0, help='alpha, recommended value 1.0')
parser.add_argument('--lambda2_G', type=float, default=1.0, help='beta, recommended value 1.0')
parser.add_argument('--lambda3_G', type=float, default=1.0, help='lambda, recommended value 1.0')
parser.add_argument('--batchSize', type=int, default=150, help='batch size')
parser.add_argument('--nz', type=int, default=128, help='size of the noise vector')
parser.add_argument('--nepochs', type=int, default=1000, help='number of epochs for training')
parser.add_argument('--lr', type=float, default=0.0001, help='learning rate, default=0.0001')
parser.add_argument('--beta1', type=float, default=0.9, help='beta1 for adam. default=0.9')
parser.add_argument('--r', type=float, default=0.5, help='default=0.1, noise intensity, gamma, recommended value 0.1')
opt = parser.parse_args()

SRC_PER_NUM = 150
TGT_NUM_PER_CLASS = 5

cuda = True if torch.cuda.is_available() else False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor


if not os.path.exists('./EX'):
    os.makedirs('./EX')

os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

def euclidean_metric(a, b):
    n = a.shape[0]
    m = b.shape[0]
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    logits = -((a - b)**2).sum(dim=2)
    return logits

def Patch(data, height_index, width_index):
    height_slice = slice(height_index, height_index + opt.patch_size)
    width_slice = slice(width_index, width_index + opt.patch_size)
    patch = data[height_slice, width_slice, :]
    return patch

def set_seed(seed=1):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
seed_list = [8000, 20, 300, 4465, 4845, 60, 768, 8686, 9898, 60]
#            0  1   2    3    4   5  6   7    8   9
mat = loadmat('../../holi/mat/'+ opt.DATASET +'.mat')     # use SOURCE-TARGET dataset

SOURCE = mat['DataCube1']     # SOURCE represents the source scene.
SOURCE_GT = mat['gt1']
SOURCE = utils.pre_scale(SOURCE)
s_old1, s_old2, s_old3 = SOURCE.shape[0], SOURCE.shape[1], SOURCE.shape[2]

TARGET = mat['DataCube2']  # TARGET represents the target scene.
TARGET_GT = mat['gt2']
TARGET = utils.pre_scale(TARGET)
t_old1, t_old2, t_old3 = TARGET.shape[0], TARGET.shape[1], TARGET.shape[2]

SOURCE = utils.padWithZeros(SOURCE, opt.patch_size // 2)
TARGET = utils.padWithZeros(TARGET, opt.patch_size // 2)


r_list =[0.1]               # gamma, recommended value 0.1
lambda1_D_list = [1.0]      # alpha, recommended value 1.0
lambda2_G_list = [1.0]      # beta, recommended  1.0
lambda3_G_list = [1.0]      # lambda, recommended  1.0
ncycle_list = [5]           # K, recommended value 5

for lambda1_D_list_i in range(0, len(lambda1_D_list)):
    for lambda2_G_list_i in range(0, len(lambda2_G_list)):
        for lambda3_G_list_i in range(0, len(lambda3_G_list)):
            for r_list_i in range(0, len(r_list)):
                for ncycle_list_i in range(0, len(ncycle_list)):
                    opt.lambda1_D = lambda1_D_list[lambda1_D_list_i]
                    opt.lambda2_G = lambda2_G_list[lambda2_G_list_i]
                    opt.lambda3_G = lambda3_G_list[lambda3_G_list_i]
                    opt.r = r_list[r_list_i]
                    opt.ncycle = ncycle_list[ncycle_list_i]

                    PATH = './EX/EX(noise'+str(opt.r)+')' + str(opt.lambda1_D) + '-' + str(opt.lambda2_G) + '-' + str(opt.lambda3_G) + '-' + str(opt.ncycle) + '/'
                    if not os.path.exists(PATH):
                        os.makedirs(PATH)

                    for traini in range(0, 10):

                        set_seed(seed_list[traini])

                        Evaluation = {'oa': -1000, 'aa': -1000, 'k': -1000, 'time_all': -1000}

                        if not os.path.exists(PATH + 'EX' + str(traini+1)):
                            os.makedirs(PATH + 'EX' + str(traini+1))

                        number = traini
                        source_train_index = np.load('./data-source/source_train_index_in_GT' + str(number) + '.npy', allow_pickle=True)
                        source_train_label = np.load('./data-source/source_train_label' + str(number) + '.npy', allow_pickle=True)

                        target_train_index = np.load('./data-target/target_train_index_in_GT' + str(number) + '.npy', allow_pickle=True)
                        target_train_label = np.load('./data-target/target_train_label' + str(number) + '.npy', allow_pickle=True)

                        target_test_index = np.load('./data-target/target_test_index_in_GT' + str(number) + '.npy', allow_pickle=True)
                        target_test_label = np.load('./data-target/target_test_label' + str(number) + '.npy', allow_pickle=True)

                        source_train_info = []      # spatial info of source samples for training (patch)

                        target_train_info = []      # spatial info of target samples for training (patch)

                        target_test_info = []      # spatial info of target samples for test (patch)

                        for i in range(0, len(source_train_index)):

                            a = source_train_index[i] // s_old2
                            b = source_train_index[i] % s_old2
                            image_patch = Patch(SOURCE, a, b)
                            source_train_info.append(image_patch.transpose(2, 0, 1))

                        source_train_info = np.asarray(source_train_info)  # src_train_number*bands*7*7, 7: patch size
                        source_train_label = np.asarray(source_train_label)
                        source_train_label = source_train_label - 1
                        source_train_info = torch.from_numpy(source_train_info)
                        source_train_label = torch.from_numpy(source_train_label)

                        source_train_dataset = utils.SpectralSpatialData(source_train_info, source_train_label)
                        source_trainloader = Data.DataLoader(dataset=source_train_dataset, batch_size=opt.batchSize,
                                                             shuffle=True,
                                                             drop_last=False)

                        # target_train
                        for i in range(0, len(target_train_index)):
                            a = target_train_index[i] // t_old2
                            b = target_train_index[i] % t_old2
                            image_patch = Patch(TARGET, a, b)
                            target_train_info.append(image_patch.transpose(2, 0, 1))

                        target_train_info = np.asarray(target_train_info)
                        target_train_label = np.asarray(target_train_label)
                        target_train_label = target_train_label - 1

                        target_train_info_tensor = torch.from_numpy(target_train_info)
                        target_train_label_tensor = torch.from_numpy(target_train_label)
                        target_train_dataset = utils.SpectralSpatialData(target_train_info_tensor, target_train_label_tensor)
                        target_trainloader = Data.DataLoader(dataset=target_train_dataset, batch_size=opt.nclasses*5, shuffle=True, drop_last=False)

                        # target_test
                        for i in range(0, len(target_test_index)):
                            a = target_test_index[i] // t_old2
                            b = target_test_index[i] % t_old2
                            image_patch = Patch(TARGET, a, b)
                            target_test_info.append(image_patch.transpose(2, 0, 1))

                        target_test_info = np.asarray(target_test_info)
                        target_test_label = np.asarray(target_test_label)
                        target_test_label = target_test_label - 1
                        target_test_info = torch.from_numpy(target_test_info)
                        target_test_label = torch.from_numpy(target_test_label)
                        target_test_info = utils.SpectralSpatialData(target_test_info, target_test_label)
                        target_testloader = Data.DataLoader(dataset=target_test_info, batch_size=200, shuffle=False, drop_last=False)


                        netFs = mymodels._netF(opt.feature_dim, opt.source_dim)
                        netFt = mymodels._netF(opt.feature_dim, opt.target_dim)

                        netDs = mymodels._netD(opt.feature_dim, opt.nclasses)
                        netDt = mymodels._netD(opt.feature_dim, opt.nclasses)

                        netGts = mymodels._netG(opt.feature_dim, opt.feature_dim, opt.nz, opt.nclasses)
                        netGst = mymodels._netG(opt.feature_dim, opt.feature_dim, opt.nz, opt.nclasses)

                        criterion_c = torch.nn.CrossEntropyLoss()
                        criterion_s = torch.nn.BCELoss()
                        criterion_r = torch.nn.MSELoss()

                        netFt.apply(utils.weights_init)
                        netFs.apply(utils.weights_init)
                        netGts.apply(utils.weights_init)
                        netGst.apply(utils.weights_init)
                        netDs.apply(utils.weights_init)
                        netDt.apply(utils.weights_init)

                        if cuda:
                            netDs.cuda()
                            netDt.cuda()
                            netGts.cuda()
                            netGst.cuda()
                            netFt.cuda()
                            netFs.cuda()

                            criterion_c.cuda()
                            criterion_s.cuda()
                            criterion_r.cuda()
                        optimizer_netGts_Gst = torch.optim.Adam(
                            itertools.chain(netGts.parameters(), netGst.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.2) #, betas=(opt.beta1, 0.999), weight_decay=0.2

                        optimizer_netD_F = torch.optim.Adam(
                            itertools.chain(netDs.parameters(), netDt.parameters(), netFt.parameters(), netFs.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999), weight_decay=0.2)

                        since = time.time()
                        print('training', ' K:', opt.ncycle, ' No.', traini)
                        for epoch in range(1, opt.nepochs+1):

                            netGst.train()
                            netGts.train()
                            netDs.train()
                            netDt.train()
                            netFs.train()
                            netFt.train()
                            loss_G_epoch = 0
                            loss_D_epoch = 0
                            batch_num = 0
                            for batch_i, data in enumerate(zip(source_trainloader, cycle(target_trainloader))):
                                batch_num = batch_num + 1
                                src_patch, src_labels = data[0][0], data[0][1]
                                tgt_patch, tgt_labels = data[1][0], data[1][1]
                                src_size = src_labels.shape[0]
                                tgt_size = tgt_labels.shape[0]
                                reallabelv_src = FloatTensor(src_size).fill_(1.0).reshape(src_size, 1)
                                fakelabelv_src = FloatTensor(src_size).fill_(0.0).reshape(src_size, 1)
                                reallabelv_tgt = FloatTensor(tgt_size).fill_(1.0).reshape(tgt_size, 1)
                                fakelabelv_tgt = FloatTensor(tgt_size).fill_(0.0).reshape(tgt_size, 1)

                                tgt_labels_onehot = torch.zeros(tgt_size, opt.nclasses).scatter_(1, tgt_labels.view(tgt_size, 1).type(
                                                                                                     torch.int64), 1)
                                src_labels_onehot = torch.zeros(src_size, opt.nclasses).scatter_(1, src_labels.view(src_size, 1).type(
                                                                                                     torch.int64), 1)

                                # if cuda:
                                src_patchv, src_labelsv = src_patch.type(FloatTensor), src_labels.type(LongTensor)
                                tgt_patchv, tgt_labelsv = tgt_patch.type(FloatTensor), tgt_labels.type(LongTensor)
                                tgt_labels_onehotv, src_labels_onehotv = tgt_labels_onehot.type(LongTensor), src_labels_onehot.type(LongTensor)


                                x_gen_src_list = []
                                x_gen_tgt_list = []
                                x_hat_tgt_list = []
                                x_hat_src_list = []
                                optimizer_netD_F.zero_grad()
                                optimizer_netGts_Gst.zero_grad()

                                xs_0 = netFs(src_patchv)
                                xt_0 = netFt(tgt_patchv)

                                tgt_f_copy = torch.ones_like(xt_0).type(FloatTensor)
                                src_f_copy = torch.ones_like(xs_0).type(FloatTensor)
                                tgt_real_c_onehot = torch.ones_like(tgt_labels_onehotv).type(LongTensor)
                                src_real_c_onehot = torch.ones_like(src_labels_onehotv).type(LongTensor)
                                """
                                Discriminator (netD)
                                 """

                                for ncycle_i in range(0, opt.ncycle):
                                    if ncycle_i == 0:
                                        tgt_f_copy.copy_(xt_0)
                                        src_f_copy.copy_(xs_0)

                                        src_real_s, src_real_c = netDs(src_f_copy)
                                        tgt_real_s, tgt_real_c = netDt(tgt_f_copy)

                                        src_real_s_err = criterion_s(src_real_s, reallabelv_src)
                                        src_real_c_err = criterion_c(src_real_c, src_labelsv)
                                        tgt_real_s_err = criterion_s(tgt_real_s, reallabelv_tgt)
                                        tgt_real_c_err = criterion_c(tgt_real_c, tgt_labelsv)

                                        loss_D = src_real_s_err + tgt_real_s_err + opt.lambda1_D * (src_real_c_err + tgt_real_c_err)

                                    x_gen_src = netGts(tgt_f_copy, tgt_labels_onehotv, opt.r)
                                    x_gen_tgt = netGst(src_f_copy, src_labels_onehotv, opt.r)

                                    x_gen_src_s, x_gen_src_c = netDs(x_gen_src)
                                    x_gen_tgt_s, x_gen_tgt_c = netDt(x_gen_tgt)

                                    x_hat_tgt = netGst(x_gen_src, tgt_labels_onehotv, opt.r)
                                    x_hat_src = netGts(x_gen_tgt, src_labels_onehotv, opt.r)

                                    x_gen_src_list.append(x_gen_src)
                                    x_gen_tgt_list.append(x_gen_tgt)
                                    x_hat_tgt_list.append(x_hat_tgt)
                                    x_hat_src_list.append(x_hat_src)

                                    x_hat_src_s, x_hat_src_c = netDs(x_hat_src)
                                    x_hat_tgt_s, x_hat_tgt_c = netDt(x_hat_tgt)

                                    x_gen_src_s_err = criterion_s(x_gen_src_s, fakelabelv_tgt)
                                    x_gen_src_c_err = criterion_c(x_gen_src_c, tgt_labelsv)
                                    x_gen_tgt_s_err = criterion_s(x_gen_tgt_s, fakelabelv_src)
                                    x_gen_tgt_c_err = criterion_c(x_gen_tgt_c, src_labelsv)
                                    x_hat_src_s_err = criterion_s(x_hat_src_s, fakelabelv_src)
                                    x_hat_src_c_err = criterion_c(x_hat_src_c, src_labelsv)
                                    x_hat_tgt_s_err = criterion_s(x_hat_tgt_s, fakelabelv_tgt)
                                    x_hat_tgt_c_err = criterion_c(x_hat_tgt_c, tgt_labelsv)

                                    src_f_copy = x_hat_src
                                    tgt_f_copy = x_hat_tgt

                                    loss_D = loss_D + (1.0) * (x_gen_src_s_err + x_gen_tgt_s_err + x_hat_src_s_err + x_hat_tgt_s_err \
                                                + opt.lambda1_D * (x_gen_src_c_err + x_gen_tgt_c_err + x_hat_src_c_err + x_hat_tgt_c_err))

                                loss_G = criterion_r(torch.zeros(2).type(FloatTensor), torch.zeros(2).type(FloatTensor))
                                """
                                Generators
                                """
                                for ncycle_i in range(0, opt.ncycle):
                                    x_gen_src_s, x_gen_src_c = netDs(x_gen_src_list[ncycle_i])
                                    x_gen_tgt_s, x_gen_tgt_c = netDt(x_gen_tgt_list[ncycle_i])

                                    x_hat_src_s, x_hat_src_c = netDs(x_hat_src_list[ncycle_i])
                                    x_hat_tgt_s, x_hat_tgt_c = netDt(x_hat_tgt_list[ncycle_i])

                                    x_gen_src_s_err = criterion_s(x_gen_src_s, reallabelv_tgt)
                                    x_gen_src_c_err = criterion_c(x_gen_src_c, tgt_labelsv)
                                    x_gen_tgt_s_err = criterion_s(x_gen_tgt_s, reallabelv_src)
                                    x_gen_tgt_c_err = criterion_c(x_gen_tgt_c, src_labelsv)
                                    x_hat_src_s_err = criterion_s(x_hat_src_s, reallabelv_src)
                                    x_hat_src_c_err = criterion_c(x_hat_src_c, src_labelsv)
                                    x_hat_tgt_s_err = criterion_s(x_hat_tgt_s, reallabelv_tgt)
                                    x_hat_tgt_c_err = criterion_c(x_hat_tgt_c, tgt_labelsv)

                                    reconv_src_err = criterion_r(x_hat_src_list[ncycle_i], xs_0)
                                    reconv_tgt_err = criterion_r(x_hat_tgt_list[ncycle_i], xt_0)

                                    loss_G = loss_G + (1.0) * ((opt.lambda3_G) * (reconv_src_err + reconv_tgt_err) \
                                                + (x_gen_tgt_s_err + x_hat_tgt_s_err + x_gen_src_s_err + x_hat_src_s_err) \
                                                + opt.lambda2_G * (x_gen_src_c_err + x_gen_tgt_c_err + x_hat_src_c_err + x_hat_tgt_c_err))

                                loss_G.backward(retain_graph=True)
                                loss_D.backward()
                                optimizer_netGts_Gst.step()
                                optimizer_netD_F.step()

                            if epoch % opt.nepochs == 0:    # save model
                                netFt.eval()
                                netFs.eval()
                                netDs.eval()
                                netDt.eval()
                                netGst.eval()
                                netGts.eval()
                                checkpoint = {
                                    'epoch': epoch,
                                    'netGts': netGts.state_dict(),
                                    'netGst': netGst.state_dict(),
                                    'netDs': netDs.state_dict(),
                                    'netDt': netDt.state_dict(),
                                    'netFt':netFt.state_dict(),
                                    'netFs': netFs.state_dict(),
                                }
                                if not os.path.exists(PATH + 'EX' + str(traini+1) + '/checkpoint'):
                                    os.makedirs(PATH + 'EX' + str(traini+1) + '/checkpoint')
                                torch.save(checkpoint, PATH + 'EX' + str(traini+1) + '/checkpoint/checkpoint'+str(epoch))

                            if (epoch % opt.nepochs) == 0:     # test

                                netDs.eval()
                                netDt.eval()
                                netFs.eval()
                                netFt.eval()
                                total = 0
                                correct = 0
                                predlabel = torch.Tensor(np.array([])).type(LongTensor)
                                realtestlabel = torch.Tensor(np.array([])).type(LongTensor)

                                for testi, testdatas in enumerate(target_testloader):

                                    test_patch, testlabels = testdatas

                                    if cuda:
                                        test_patchv, testlabels = test_patch.type(FloatTensor), testlabels.type(LongTensor)

                                    test_f = netFt(test_patchv)
                                    _, test_out_c = netDt(test_f)
                                    _, predicted = torch.max(test_out_c.data, 1)
                                    predlabel = torch.cat((predlabel, predicted), -1)
                                    realtestlabel = torch.cat((realtestlabel, testlabels), -1)
                                    total += testlabels.size(0)
                                    correct += ((predicted == testlabels.cuda()).sum())

                                test_acc = float(correct) / total

                                predlabel = predlabel + 1
                                realtestlabel = realtestlabel + 1
                                predlabel = predlabel.data.cpu().numpy()
                                realtestlabel = realtestlabel.data.cpu().numpy()
                                C = confusion_matrix(realtestlabel, predlabel)
                                kappa = cohen_kappa_score(realtestlabel, predlabel, labels=None, weights=None,
                                                          sample_weight=None)
                                aa = utils.AA(C)
                                Evaluation['oa'] = 100 * test_acc
                                Evaluation['aa'] = 100 * aa
                                Evaluation['k'] = 100 * kappa

                                print('noise('+str(opt.r)+')'+ str(opt.lambda1_D) + '-' + str(opt.lambda2_G) + '-' + str(opt.lambda3_G) + '-' + str(
                                    opt.ncycle), traini, 'OA', 100 * test_acc, 'AA', 100 * aa, 'k', 100 * kappa)


                        time_last = time.time() - since
                        del netDs, netDt, netGts, netGst, netFs, netFt
                        print('Training complete in {:.0f}m {:.0f}s'.format(time_last // 60, time_last % 60))

                        Evaluation['time'] = time_last
                        torch.save(Evaluation, PATH + 'EX' + str(traini + 1) + '/evaluation')









