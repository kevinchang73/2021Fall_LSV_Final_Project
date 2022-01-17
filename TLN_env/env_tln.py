import numpy as np
from TLN_env.TLN.TLN import Tln
import sys
import math
import torch
import torch.nn as nn
from torch.autograd import Variable


class Tln_env():

    def __init__(self, inputFile):
        self.TLN = Tln(inputFile)
    
    def step(self, action, outputs_ref_values):
        # action = list(action)
        self.TLN.set_weights(action[0:len(self.TLN.edges)])
        # self.TLN.print_weights()
        # self.TLN.set_thresholds([0]*len(self.TLN.nodes))
        # outputs = torch.empty(0, dtype = torch.float)
        outputs = torch.empty(len(outputs_ref_values), dtype = torch.float)
        for i in range(int(math.pow(2, len(self.TLN.pis)))):
            input_values = "{0:b}".format(i).zfill(len(self.TLN.pis))
            input_values = torch.tensor(list(map(int, list(input_values))), dtype = torch.float)
            input_values.requires_grad = True
            self.TLN.propagate(input_values)

            #MSELoss
            for j in range(len(self.TLN.pos)):
                outputs[i*len(self.TLN.pos) + j] = self.TLN.collect_outputs()[j]
                # outputs[i*len(self.TLN.pos):(i + 1)*len(self.TLN.pos)] = self.TLN.collect_outputs()
            # SE.extend(self.TLN.collect_outputs())
            # SE.append(s)
        #CrossEntropy
        # outputs = torch.tensor([SE, output_values], dtype = torch.float)
        # loss = nn.CrossEntropyLoss()
        # outputs.requires_grad = True
        # return loss(outputs, torch.tensor([0, 1], dtype = torch.long))

        #MSELoss
        # outputs = torch.tensor(SE, dtype = torch.float)
        # print("outputs: ", outputs)
        # target = torch.tensor(output_ref_values, dtype = torch.float)
        outputs_ref_values.requires_grad = True
        # print("target: ", target)
        # MSE = torch.from_numpy(MSELoss)
        # MSE.requires_grad = True
        print()
        print("outputs: ", outputs)
        print("outputs_ref_values: ", outputs_ref_values)
        return nn.MSELoss()(outputs, outputs_ref_values)

if __name__ == "__main__":
    if sys.argv[1] == 'read':
        input_file = sys.argv[2]
        model = Tln_env(input_file)
        model.step([1, -0.5, 1], [0])
        print("done")
