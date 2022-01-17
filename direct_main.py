import numpy as np
import sys
from Direct_NN.manipulateTLN import *

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
print('Size of training data = {}'.format(train_x.shape))
print('Size of training label = {}'.format(train_y.shape))