import NTM
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import data
import time

insize = 8
outsize = insize
copy_len = [1,20]
shifting_size = 11
model = NTM.NTM( in_size = insize+1,
        out_size = outsize,
        memory_size = [20*2, insize*2],
        shifting_size = shifting_size,
        contr_class = NTM.LSTMChain, 
        #contr_class = NTM.FeedForward, # use this line for feed-forward controller
        contr_dict = {"hid_layers": [200]})
gen = data.generator_copy(insize, copy_len)

optim = torch.optim.RMSprop(model.parameters(), lr=0.0001, momentum = 0.9)
def loss_func(output, target):
    return torch.mean(-target* F.logsigmoid(output) - (1-target) * F.logsigmoid(-output))
    
    
iterations = 10000
batch_size = 200

def train():
    pre_time = time.time()
    for i in range(iterations):
        input, target = gen.get_data( batch_size)
        model.zero_grad()
        output = model(input)
        loss = loss_func(output[target.size()[0]:], target)
        loss.backward()
        optim.step()
        if i%100 == 0:
            now = time.time()
            print("iter: ", i, " loss: ", loss.item(), "time:", now - pre_time)
            pre_time = now

train()
