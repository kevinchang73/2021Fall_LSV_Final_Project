#Two level TLN w11, w12, w13, w14, w21, w22, t11, t12, t2
#TLN_mat is 2-dim array: [[2, 2], [2]] in above example
#TLN-weight: [[[w11, w12], [w13, w14]], [[w21, w22]]]
#TLN-weight: [[t11, t12], [t2]]
#import numpy as np
import sys

class Node:
    def __init__(self, id, isPI, isPO):
        self.id = id
        self.isPI = isPI
        self.isPO = isPO
        self.threshold = 0.0
        self.outs = dict() #output node id to weight
        self.ins = dict() #input node id to weight

class Tln:
    def __init__(self, tln_file):
        self.nodes = []
        self.level = int(0)
        self.weights = []
        self.thresholds = []
        self.read_and_init_tln(tln_file)

    def read_and_init_tln(self, tln_file):
        f = open(tln_file, 'r')
        line = f.readline()
        # nodes
        while True:
            line = f.readline()
            if len(line.split(' ')) > 1:
                id = line.split(' ')[0]
                isPI = bool(int(line.split(' ')[1]))
                isPO = bool(int(line.split(' ')[2].strip('\n')))
                node = Node(id, isPI, isPO)
                self.nodes.append(node)
                print(id, isPI, isPO)
            else:
                break
        print(len(self.nodes))
        # edges
        while True:
            line = f.readline()
            if len(line.split(' -> ')) > 1:
                id1 = int(line.split(' -> ')[0])
                id2 = int(line.split(' -> ')[1].strip('\n'))
                print(id1, id2)
                u = self.nodes[id1-1]
                v = self.nodes[id2-1]
                u.outs[id2] = 0.0
                v.ins[id1] = 0.0
            else:
                break
        # level
        line = f.readline()
        self.level = int(line)
        f.close()
        self.write_tln('inputs/test.tln')
    
    def write_tln(self, out_file):
        f = open(out_file, 'w')
        out_str = []
        out_str.append('nodeID isPI isPO\n')
        for node in self.nodes:
            out_str.append(str(node.id) + ' ' + str(int(node.isPI)) + ' ' + str(int(node.isPO)) + '\n')
        out_str.append('edges\n')
        for node in self.nodes:
            for key in node.outs:
                out_str.append(str(node.id) + ' -> ' + str(key) + '\n')
        out_str.append('levels\n')
        out_str.append(str(self.level) + '\n')
        f.writelines(out_str)
        f.close()
    
    #def reset(self):


    #def reward(self, weigit_threshold):
        #assign weight and threshold to the TLN
        #reward = accuracy(output, theo_output)
        #return reward

if __name__ == "__main__":
    # Usage: python3 TLN.py [read/write] filename
    if sys.argv[1] == 'read':
        input_file = sys.argv[2]
        myTln = Tln(input_file)

