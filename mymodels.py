
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np

cuda = True if torch.cuda.is_available() else False
# cuda = False
FloatTensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor
LongTensor = torch.cuda.LongTensor if cuda else torch.LongTensor



"""

"""

class _netG(nn.Module):
    def __init__(self, input_dim, out_dim, nz, nclasses):
        super(_netG, self).__init__()

        self.input_dim = input_dim
        self.out_dim = out_dim
        self.nz = nz
        self.nclasses = nclasses

        self.linear_1 = nn.Sequential(


            nn.Linear(self.nz + self.input_dim + self.nclasses, 256),
            nn.Tanh()
        )
        self.linear_2 = nn.Sequential(

            nn.Linear(256, 256),
            nn.LeakyReLU(0.2)
        )
        self.linear_3 = nn.Sequential(

            nn.Linear(256, 256),

        )
        self.linear_4 = nn.Sequential(

            nn.Linear(256, self.out_dim),
        )



    def forward(self, input, one_hot, r=None):
        batchSize = input.size()[0]
        input = input.reshape(-1, self.input_dim)
        if r==None:
            noise = FloatTensor(batchSize, self.nz).normal_(0, 1)
        else:
            input_mean = torch.mean(torch.abs(input)).cpu().data.numpy()
            # print(input_mean)
            noise = FloatTensor(batchSize, self.nz).normal_(0, r * input_mean)
        input = torch.cat((one_hot, input, noise), 1).reshape(-1, self.nz + self.input_dim + self.nclasses)
        # input = input + noise
        output = self.linear_1(input)
        output = self.linear_2(output)
        output = self.linear_3(output)
        output = self.linear_4(output)

        return output.reshape(-1, self.out_dim)



class _netD(nn.Module):
    def __init__(self, dimension, nclasses):
        super(_netD, self).__init__()

        self.dimension = dimension
        self.nclasses = nclasses

        self.linear_1 = nn.Sequential(
            nn.Linear(self.dimension, 1024),
            nn.ReLU(),
        )
        self.linear_2 = nn.Sequential(
            nn.Linear(1024, 512),
            nn.ReLU(),

        )
        self.linear_3 = nn.Sequential(
            nn.Linear(512, 180),
            nn.ReLU(),
            nn.Dropout(0.5),
        )
        self.classifier_s = nn.Sequential(
            nn.Linear(180, 1),
            nn.Sigmoid()
        )
        self.classifier_c = nn.Sequential(
            nn.Linear(180, self.nclasses),
        )

    def forward(self, input):

        output = self.linear_1(input)
        output = self.linear_2(output)
        output = self.linear_3(output)
        output_s = self.classifier_s(output)
        output_c = self.classifier_c(output)
        return output_s, output_c

class _netF(nn.Module):
    def __init__(self, FEATURE_DIM, INPUT_SPEC_DIM):
        super(_netF, self).__init__()
        self.INPUT_SPEC_DIM = INPUT_SPEC_DIM
        self.FEATURE_DIM = FEATURE_DIM

        self.conv_1 = nn.Sequential(
            nn.Conv2d(INPUT_SPEC_DIM, 70, 1, 1, 0),
            nn.ReLU()
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(70, 50, 1, 1, 0),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.conv_3 = nn.Sequential(
            nn.Conv3d(1, 32, (1, 1, 3), (1, 1, 2), (0, 0, 1)),
            nn.ReLU(),

        )
        self.conv_4 = nn.Sequential(
            nn.Conv3d(32, 16, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.ReLU(),
            nn.Dropout(0.25)
        )
        self.conv_5 = nn.Sequential(
            nn.Conv3d(16, 8, (3, 3, 3), (2, 2, 1), (1, 1, 1)),
            nn.ReLU(),

        )
        self.conv_6 = nn.Sequential(
            nn.Conv3d(8, 1, (3, 3, 3), (1, 1, 1), (1, 1, 1)),
            nn.ReLU(),
        )

    def forward(self, x, ):  # x
        batchSize = x.shape[0]
        x = self.conv_1(x)
        x = self.conv_2(x)
        x = x.permute(0, 2, 3, 1)
        x = x.unsqueeze(1)
        x = self.conv_3(x)
        x = self.conv_4(x)
        x = self.conv_5(x)
        x = self.conv_6(x)

        x = x.reshape(batchSize, self.FEATURE_DIM)

        return x




