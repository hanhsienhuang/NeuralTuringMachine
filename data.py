import torch
import torch.nn.functional as F
from torch.autograd import Variable
import random


class generator_copy(object):
    def __init__(self, width, lengths):
        self.width = width
        self.lengths = lengths

    def get_data(self, batchsize):
        """
        return input, target
        input size:  (length*2) * batchsize * width
        target size: (length*2) * batchsize * width
        """
        length = random.randint(*self.lengths)
        data  = Variable((torch.rand(length, batchsize, self.width+1)>0.5).float(), requires_grad=False).cuda()
        data[:, :, -1] = 0
        data[-1, :, -1] = 1
        zeros = Variable(torch.zeros(length, batchsize, self.width+1), requires_grad=False).cuda()
        input = torch.cat([data, zeros], dim = 0)
        output = data[..., :-1]
        return input, output
