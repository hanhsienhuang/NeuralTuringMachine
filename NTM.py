import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.nn.functional as F
import math
import numpy as np


def matmul_blw_bw(m, v):
    return torch.sum(m*v.unsqueeze(1).expand_as(m), dim = -1).squeeze(-1) # return dim = b*l

def mat_expand(u, v):
    b = u.size()[0]
    l = u.size()[1]
    w = v.size()[1]
    return u.unsqueeze(-1).expand(b,l,w) * v.unsqueeze(1).expand(b,l,w)

class FeedForward(nn.Module):
    def __init__(self, in_size, out_size, hid_layers, activation = torch.tanh):
        super(FeedForward, self).__init__()
        layers = [in_size] + hid_layers + [out_size]
        self.linears = nn.ModuleList([ nn.Linear( layers[i], layers[i+1]) for i in range(len(layers)-1) ])
        self.layers = layers
        self.in_size = in_size
        self.out_size = out_size
        self.activation = activation
        self.cuda()

    def initialize(self, batch_size):
        pass

    def forward(self, input):
        pre_output = input
        for i in range(len(self.layers)-2):
            pre_output = self.linears[i](pre_output)
            pre_output = self.activation(pre_output)
        pre_output = self.linears[-1](pre_output)
        return pre_output

class LSTMChain(nn.Module):
    def __init__(self, in_size, out_size, hid_layers):
        super(LSTMChain, self).__init__()
        layers = [in_size] + hid_layers 
        self.LSTMcells = nn.ModuleList([ nn.LSTMCell( layers[i], layers[i+1]) for i in range(len(hid_layers)) ])
        self.Linear = nn.Linear(layers[-1], out_size)
        self.hid_layers = hid_layers
        self.in_size = in_size
        self.out_size = out_size
        def zeros_variable( size):
            return Variable(torch.zeros(size), requires_grad = False).cuda()
        self.states_init = [( zeros_variable([1, size]), zeros_variable([1, size]) ) for size in self.hid_layers]

        # move to gpu
        self.cuda()

    def initialize(self, batch_size):
        def expand_batch( param):
            return param.expand((batch_size,)+param.size()[1:])
        self.states = [(expand_batch(h0), expand_batch(c0)) for (h0, c0) in self.states_init]

    def forward(self, input):
        pre_output = input
        for i in range(len(self.hid_layers)):
            state = self.LSTMcells[i]( pre_output, self.states[i])
            pre_output = state[0]
            self.states[i] = state
        output = self.Linear(pre_output)
        return output

class NTM(nn.Module):
    def __init__(self, in_size, out_size, memory_size, shifting_size, contr_class, contr_dict = {}):
        super(NTM, self).__init__()
        # parameters
        self.in_size = in_size
        self.out_size = out_size
        self.shifting_size = shifting_size
        self.mem_length, self.mem_width = memory_size
        self.read_params_size  = self.mem_width + 3 + shifting_size
        self.write_params_size = self.mem_width + 3 + shifting_size + 2*self.mem_width

        # variables
        def zeros_variable(size, ones = False):
            var = Variable(torch.zeros(size), requires_grad = False)
            if ones:
                var.data[0] = 1.0
            return var.cuda()
        self.r_init = zeros_variable( [self.mem_width])
        self.output_init = zeros_variable([self.out_size])
        self.read_w_init = zeros_variable([self.mem_length], ones = True)
        self.write_w_init = self.read_w_init.clone()
        self.memory_init = zeros_variable([self.mem_length, self.mem_width])

        # construct controller
        contr_in_size = in_size + out_size + self.mem_width 
        contr_out_size = out_size + self.read_params_size + self.write_params_size
        self.controller = contr_class(contr_in_size, contr_out_size, **contr_dict)

        # move variables to gpu
        self.cuda()

    def initialize(self, batch_size):
        def expand_batch( param):
            return param.unsqueeze(0).expand((batch_size,)+param.size())
        self.r = expand_batch( self.r_init)
        self.output = expand_batch( self.output_init)
        self.read_w = expand_batch( self.read_w_init)
        self.write_w = expand_batch( self.write_w_init)
        self.memory = expand_batch( self.memory_init)

    def forward(self, inputs, reset=True):
        outputs = []
        batch_size = inputs.size()[1]
        if reset:
            self.initialize(batch_size)
            self.controller.initialize(batch_size)
        for t in range(inputs.size()[0]):
            input_t = inputs[t]
            contr_input  = torch.cat([input_t, self.output, self.r], dim = 1)
            contr_output = self.controller(contr_input)
            self.output  = contr_output[:, :self.out_size]
            read_params  = contr_output[:, self.out_size: -self.write_params_size]
            write_params = contr_output[:, -self.write_params_size:]
            self.r       = self.read_head(read_params)
            self.write_head(write_params)
            outputs.append( self.output)
        outputs = torch.stack(outputs)
        return outputs

    def read_head(self, read_params):
        w = self.get_w( read_params, self.read_w)
        self.read_w = w
        return matmul_blw_bw(self.memory.transpose(1,2), w)

    def write_head(self, write_params):
        w = self.get_w( write_params[:, :-2*self.mem_width], self.write_w)
        e = torch.sigmoid( write_params[:, -2*self.mem_width: -self.mem_width])
        a = write_params[:, -self.mem_width:]
        self.memory = self.memory *( 1 - mat_expand(w, e))
        self.memory += mat_expand(w, a)
        self.write_w = w

    def get_w(self, input, w):
        # input = cat( [k, beta, g, gamma, s])
        k     = input[:, :self.mem_width]  # dim = batchsize
        beta  = F.softplus(input[:, self.mem_width]).unsqueeze(-1)      # dim = batchsize * 1
        g     = torch.sigmoid( input[:, self.mem_width+1]).unsqueeze(-1)    # dim = batchsize * 1
        gamma = F.softplus(input[:, self.mem_width+2] ).unsqueeze(-1)+1  # dim = batchsize * 1
        s     = F.softmax( input[:, -self.shifting_size:], dim=-1)  # dim = batchsize * shift_size

        # content addressing
        epsilon = 1e-14
        norm_k   = torch.sqrt( torch.sum( torch.pow(k, 2), dim = -1, keepdim=True) + epsilon) # dim = batchsize * 1
        norm_mem = torch.sqrt( torch.sum( torch.pow(self.memory, 2), dim = -1) + epsilon).squeeze(-1) # dim = batchsize * mem_length
        K        = matmul_blw_bw(self.memory, k) / (norm_mem *norm_k.expand_as(norm_mem)) # dim = batchsize * mem_length
        w_c = F.softmax(K*beta.expand_as(K), dim=-1)  # dim = batchsize * mem_length

        # interpolation
        g = g.expand_as(w_c)
        w_g = w_c*g + (1-g)*w  #dim = batchsize * mem_length

        # shifting
        middle = self.shifting_size // 2
        left = middle
        right = self.shifting_size - left -1
        cat_wg = torch.cat( (w_g[:, -left:], w_g, w_g[:, :right]), dim =1)   # dim = batchsize * (mem_length + shift_size -1)
        unfolded_wg = torch.stack( [cat_wg[:, i:i+self.shifting_size] for i in range(self.mem_length)], dim = 1) 
        # dim = batchsize * mem_length * shift_size
        w_tilde = matmul_blw_bw(unfolded_wg, s)    # dim = batchsize * mem_length

        # sharpening
        w = torch.pow(w_tilde, gamma.expand_as(w_tilde))
        w = w / torch.sum(w, dim = 1, keepdim=True).expand_as(w)
        return w

