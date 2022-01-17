import sys
import random
import math
import torch

class Node:
    def __init__(self, id, isPI, isPO):
        self.id = id
        self.isPI = isPI
        self.isPO = isPO
        self.threshold = torch.tensor(0.0, dtype = torch.float)
        self.threshold.requires_grad = True
        self.outs = [] #edges
        self.ins = [] #edges
        self.value = torch.tensor(0.0, dtype = torch.float)
    def calc_value(self):
        weight_x_value = torch.empty(len(self.ins), dtype = torch.float)
        i = 0
        for edge in self.ins:
            assert(torch.is_tensor(edge.weight))
            # print(edge.weight.is_leaf)
            assert(torch.is_tensor(edge.value))

            weight_x_value[i] = torch.mul(edge.weight, edge.value)
            i += 1
            # sum += edge.weight * edge.value
        assert(torch.is_tensor(self.threshold))
        weight_x_value_sum = weight_x_value.sum()
        self.value = 1/(1 + torch.exp(1000*(self.threshold - weight_x_value_sum)))
        print(self.value.requires_grad)
        print(self.value)
        # if sum >= self.threshold:
        #     self.value = torch.tensor(1.0, dtype = torch.float)
        # else:
        #     self.value = torch.tensor(0.0, dtype = torch.float)
        assert(torch.is_tensor(self.value))
        for edge in self.outs:
            edge.value = self.value
    def set_edge_value(self):
        for edge in self.outs:
            assert(torch.is_tensor(self.value))
            edge.value = self.value
            assert(torch.is_tensor(edge.value))

class Edge:
    def __init__(self, u, v):
        self.u = u
        self.v = v
        self.weight = 0.0
        self.value = bool(0)

class Tln:
    def __init__(self, tlnFile):
        self.nodes = []
        self.edges = []
        self.level = int(0)
        self.pis = []
        self.pos = []
        self.read_and_init_tln(tlnFile)

    def read_and_init_tln(self, tlnFile):
        f = open(tlnFile, 'r')
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
                if isPI:
                    self.pis.append(node)
                if isPO:
                    self.pos.append(node)
            else:
                break
        # edges
        while True:
            line = f.readline()
            if len(line.split(' -> ')) > 1:
                id1 = int(line.split(' -> ')[0])
                id2 = int(line.split(' -> ')[1].strip('\n'))
                u = self.nodes[id1-1]
                v = self.nodes[id2-1]
                edge = Edge(u, v)
                self.edges.append(edge)
                u.outs.append(edge)
                v.ins.append(edge)
            else:
                break
        # level
        line = f.readline()
        self.level = int(line)
        f.close()
        self.write_tln('inputs/test.tln')
    def write_tln(self, outFile):
        f = open(outFile, 'w')
        out_str = []
        out_str.append('nodeID isPI isPO\n')
        for node in self.nodes:
            out_str.append(str(node.id) + ' ' + str(int(node.isPI)) + ' ' + str(int(node.isPO)) + '\n')
        out_str.append('edges\n')
        for edge in self.edges:
            out_str.append(str(edge.u.id) + ' -> ' + str(edge.v.id) + '\n')
        out_str.append('levels\n')
        out_str.append(str(self.level) + '\n')
        f.writelines(out_str)
        f.close()
    
    def set_weights(self, weights): # weights is in the same order with self.edges
        assert len(weights) == len(self.edges)
        for i in range(0, len(weights)):
            self.edges[i].weight = weights[i]
            assert(torch.is_tensor(self.edges[i].weight))
    def set_thresholds(self, thresholds): # thresholds is in the same order with self.nodes
        assert len(thresholds) == len(self.nodes)
        for i in range(0, len(thresholds)):
            self.nodes[i].threshold = thresholds[i]
    def propagate(self, values):
        assert len(values) == len(self.pis)
        for i in range(0, len(self.pis)):
            assert(torch.is_tensor(values[i]))
            self.pis[i].value = values[i]
            self.pis[i].set_edge_value()
            assert(self.pis[i].value.requires_grad)
        for node in self.nodes:
            if not node.isPI:
                node.calc_value()
    def evaluate(self, values) -> int: # Modify error calculation metric here
        assert len(values) == len(self.pos)
        error = 0
        for i in range(0, len(self.pos)):
            if self.pos[i].value != values[i]:
                error += 1
        return error
    
    def collect_functions(self, funcFile, randomThreshold = False, num = 30000):
        functs = set()
        count = 0
        stop_count = 0
        while len(functs) < num:
            self.random_weights()
            if randomThreshold:
                self.random_thresholds()
            #self.print_weights()
            #self.print_thresholds()
            funct = []
            for inp in range(2**len(self.pis)):
                temp = bin(inp)[2:].zfill(len(self.pis))
                values = [int(n) for n in temp]
                self.propagate(values)
                outputs = self.collect_outputs()
                funct = funct + outputs
            size_orig = len(functs)
            functs.add(tuple(funct))
            if len(functs) == size_orig:
                stop_count += 1
            else:
                stop_count = 0
            count += 1
            print('Number of functions = {}, stop count = {}'.format(len(functs), stop_count))
            if stop_count > 1000:
                break
        #print('Iterations = ' + str(count))
        out_str = []
        for inp in range(2**len(self.pis)):
            out_str.append(bin(inp)[2:].zfill(len(self.pis)) + ' ')
        out_str.append('\n')
        for funct in sorted(functs):
            for x in funct:
                out_str.append(str(int(x)) + ' ')
            out_str.append('\n')
        f = open(funcFile, 'w')
        f.writelines(out_str)
    def random_weights(self):
        for edge in self.edges:
            edge.weight = random.uniform(-1,1)
    def random_thresholds(self):
        for node in self.nodes:
            if not node.isPI:
                node.threshold = random.uniform(-1,1)
    def collect_outputs(self) -> list:
        values = []
        for node in self.pos:
            assert(torch.is_tensor(node.value))
            values.append(node.value)
        return values

    # print
    def print_weights(self):
        print('Weights:')
        for edge in self.edges:
            print(edge.weight)
    def print_thresholds(self):
        print('Thresholds:')
        for node in self.nodes:
            print(node.threshold)

class TlnGenerator:
    def __init__(self, level, numNodes, prob = 1):
        assert level == len(numNodes)
        self.level = level
        self.numNodes = numNodes # number of node in each level
        self.prob = prob
    def write_tln(self, tlnFile, fixedInput = -1):
        f = open(tlnFile, 'w')
        out_str = []
        out_str.append('nodeID isPI isPO\n')
        idCount = 0
        for i in range(self.level):
            if i == 0:
                for j in range(self.numNodes[i]):
                    idCount += 1
                    out_str.append(str(idCount) + ' 1 0\n')
            elif i == self.level - 1:
                for j in range(self.numNodes[i]):
                    idCount += 1
                    out_str.append(str(idCount) + ' 0 1\n')
            else:
                for j in range(self.numNodes[i]):
                    idCount += 1
                    out_str.append(str(idCount) + ' 0 0\n')
        out_str.append('edges\n')
        if fixedInput < 0:
            print('Randomly connect!')
            idCount = 1
            for i in range(self.level-1):
                for u in range(idCount, idCount+self.numNodes[i]):
                    for v in range(idCount+self.numNodes[i], idCount+self.numNodes[i]+self.numNodes[i+1]):
                        temp = random.random()
                        if temp < self.prob:
                            out_str.append(str(u) + ' -> ' + str(v) + '\n')
                idCount += self.numNodes[i]
        else: # Only adjacent layers may be connected
            print('Fix input!')
            idCount = self.numNodes[0] # number of PIs
            for i in range(1, self.level):
                assert self.numNodes[i-1] <= 2*self.numNodes[i]
                while True:
                    connected = set()
                    out_edges = []
                    for v in range(idCount+1, idCount+1+self.numNodes[i]):
                        inp1 = random.randrange(idCount-self.numNodes[i-1]+1, idCount+1)
                        inp2 = -1
                        while True:
                            inp2 = random.randrange(idCount-self.numNodes[i-1]+1, idCount+1)
                            if inp1 != inp2:
                                break
                        out_edges.append(str(inp1) + ' -> ' + str(v) + '\n')
                        out_edges.append(str(inp2) + ' -> ' + str(v) + '\n')
                        connected.add(inp1)
                        connected.add(inp2)
                    if len(connected) == self.numNodes[i-1]:
                        out_str.extend(out_edges)
                        break
                idCount += self.numNodes[i]

        out_str.append('levels\n')
        out_str.append(str(self.level) + '\n')
        f.writelines(out_str)
        f.close()

if __name__ == "__main__":
    if sys.argv[1] == 'read':
    # Usage: python3 TLN.py read filename
        input_file = sys.argv[2]
        myTln = Tln(input_file)
        # example for case 0, which is a single TLG
        '''
        weights = [1, -0.5]
        thresholds = [0, 0, 1] # need to include PIs, though their thresholds is irrelavent
        inputs = [1, 1]
        outputs = [0]
        myTln.set_weights(weights)
        myTln.set_thresholds(thresholds)
        myTln.propagate(inputs)
        errorNum = myTln.evaluate(outputs)
        print('Number of errors = ' + str(errorNum))
        '''
        myTln.print_thresholds()
        myTln.collect_functions(input_file.split('.')[0]+'.funct2')
    elif sys.argv[1] == 'write':
    # Usage: python3 TLN.py write filename <prob | fixedInput> <level> <num of nodes in each level (seperated by space)>
        output_file = sys.argv[2]
        prob = float(sys.argv[3])
        level = int(sys.argv[4])
        assert len(sys.argv) == level + 5
        numNodes = []
        for i in range(5,level+5):
            numNodes.append(int(sys.argv[i]))
        if prob > 1:
            fixedInput = int(prob)
            myTlnGen = TlnGenerator(level, numNodes)
            myTlnGen.write_tln(output_file, fixedInput)
        else:
            myTlnGen = TlnGenerator(level, numNodes, prob)
            myTlnGen.write_tln(output_file)
