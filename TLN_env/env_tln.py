import numpy as np
from TLN_env.TLN.TLN import Tln
import sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class Tln_env():

    def __init__(self, inputFile, batch_size):
        self.TLN = Tln(inputFile)
        self.batch_size = batch_size
    
    def step(self, action, outputs_ref_values):
        # action = list(action)
        self.TLN.set_weights(action[0:len(self.TLN.edges)])
        # self.TLN.print_weights()
        # self.TLN.set_thresholds([0]*len(self.TLN.nodes))
        # outputs = torch.empty(0, dtype = torch.float)
        outputs = torch.empty(len(outputs_ref_values), dtype = torch.float)
        for k in range(self.batch_size):
            for i in range(int(math.pow(2, len(self.TLN.pis)))):
                input_values = "{0:b}".format(i).zfill(len(self.TLN.pis))
                input_values = torch.tensor(list(map(int, list(input_values))), dtype = torch.float)
                input_values.requires_grad = True
                self.TLN.propagate(input_values)

                #MSELoss
                for j in range(len(self.TLN.pos)):
                    outputs[k*int(len(outputs_ref_values)/self.batch_size) + i*len(self.TLN.pos) + j] = self.TLN.collect_outputs()[j]

        outputs_ref_values.requires_grad = True
        # print(outputs)
        # print()
        # print("outputs: ", outputs)
        # print("outputs_ref_values: ", outputs_ref_values)
        return nn.MSELoss()(outputs, outputs_ref_values)

if __name__ == "__main__":
    if sys.argv[1] == 'read':
        input_file = sys.argv[2]
        model = Tln_env(input_file)
        model.step([1, -0.5, 1], [0])
        print("done")
