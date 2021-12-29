#Two level TLN w11, w12, w13, w14, w21, w22, t11, t12, t2
#TLN_mat is 2-dim array: [[2, 2], [2]] in above example
#TLN-weight: [[[w11, w12], [w13, w14]], [[w21, w22]]]
#TLN-weight: [[t11, t12], [t2]]
import numpy as np
class Node():

class TLN :
    def __init__(self, TLN_mat):
        self.TLN_mat = TLN_mat
        self.level = len(TLN_mat)
        self.TLN_weight_threshold = []
        self.init_TLN()

    def init_TLN(self):

    
    def reset(self):


    def reward(self, weigit_threshold):
        #assign weight and threshold to the TLN
        #reward = accuracy(output, theo_output)
        return reward

if __name__ == "__main__":
    TLN_model = TLN([[2, 2], [2]])
    print(TLN_model.TLN_weight)
    print(TLN_model.TLN_threshold)
