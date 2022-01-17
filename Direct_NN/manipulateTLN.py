import numpy as np
from TLN_env.TLN.TLN import Tln
import sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class ManipulateTLN():
    def __init__(self, inputFile):
        self.TLN = Tln(inputFile)
    
    def step(self, action, output_ref_values):
        # action = list(action)
        self.TLN.set_weights(action[0:len(self.TLN.edges)])
        # self.TLN.print_weights()
        # self.TLN.set_thresholds([0]*len(self.TLN.nodes))
        # outputs = torch.empty(0, dtype = torch.double)
        outputs = torch.empty(len(output_ref_values), dtype = torch.double)
        for i in range(int(math.pow(2, len(self.TLN.pis)))):
            input_values = "{0:b}".format(i).zfill(len(self.TLN.pis))
            input_values = torch.tensor(list(map(int, list(input_values))), dtype = torch.double)
            self.TLN.propagate(input_values)

            #MSELoss
            outputs[i*len(self.TLN.pos):(i + 1)*len(self.TLN.pos)] = torch.tensor(self.TLN.collect_outputs(), dtype = torch.double)
            # SE.extend(self.TLN.collect_outputs())
            # SE.append(s)

        #CrossEntropy
        # outputs = torch.tensor([SE, output_values], dtype = torch.double)
        # loss = nn.CrossEntropyLoss()
        # outputs.requires_grad = True
        # return loss(outputs, torch.tensor([0, 1], dtype = torch.long))

        #MSELoss
        # outputs = torch.tensor(SE, dtype = torch.double)
        outputs.requires_grad = True
        # print("outputs: ", outputs)
        target = torch.tensor(output_ref_values, dtype = torch.double)
        target.requires_grad = True
        # print("target: ", target)
        # MSE = torch.from_numpy(MSELoss)
        # MSE.requires_grad = True
        return nn.MSELoss()(outputs, target)
