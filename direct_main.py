import numpy as np
import sys
from torch.utils.data import DataLoader
from Direct_NN.manipulateTLN import *
from Direct_NN.data import *

# Read input file
# Usage: python3 direct_main.py caseX.tln caseX.funct
tln_file = sys.argv[1]
func_file = sys.argv[2]
manTLN = ManipulateTLN(tln_file)
func_inp = open(func_file, 'r')
functions = func_inp.readlines()[1:]
functions = [list(map(int, l.strip().split(" "))) for l in functions]

# Dataset
# All the data is for training
num_data = 2**len(manTLN.TLN.pis)
train_x = np.empty([len(manTLN.TLN.pis), num_data])
train_y = np.empty([len(manTLN.TLN.pos), num_data])
print(train_x)
print(train_y)
print('Size of training data = {}'.format(train_x.shape))
print('Size of training label = {}'.format(train_y.shape))
train_set = TLNDataset(train_x, train_y)
BATCH_SIZE = 8
train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)

# Create Model
