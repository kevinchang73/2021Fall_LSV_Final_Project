import sys

class Node:
    def __init__(self, id, isPI, isPO):
        self.id = id
        self.isPI = isPI
        self.isPO = isPO
        self.threshold = 0.0
        self.outs = [] #edges
        self.ins = [] #edges
        self.value = bool(0)
    def calc_value(self):
        sum = 0
        for edge in self.ins:
            sum += edge.weight * edge.value
        if sum >= self.threshold:
            self.value = True
        else:
            self.value = False
        for edge in self.outs:
            edge.value = self.value
    def set_edge_value(self):
        for edge in self.outs:
            edge.value = self.value

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
    def set_thresholds(self, thresholds): # thresholds is in the same order with self.nodes
        assert len(thresholds) == len(self.nodes)
        for i in range(0, len(thresholds)):
            self.nodes[i].threshold = thresholds[i]
    def propagate(self, values):
        assert len(values) == len(self.pis)
        for i in range(0, len(self.pis)):
            self.pis[i].value = values[i]
            self.pis[i].set_edge_value()
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
    
    #def reward(self, weigit_threshold):
        #assign weight and threshold to the TLN
        #reward = accuracy(output, theo_output)
        #return reward

if __name__ == "__main__":
    # Usage: python3 TLN.py [read/write] filename
    if sys.argv[1] == 'read':
        input_file = sys.argv[2]
        myTln = Tln(input_file)
        # example for case 0, which is a single TLG
        weights = [1, -0.5]
        thresholds = [0, 0, 1] # need to include PIs, though their thresholds is irrelavent
        inputs = [1, 1]
        outputs = [0]
        myTln.set_weights(weights)
        myTln.set_thresholds(thresholds)
        myTln.propagate(inputs)
        errorNum = myTln.evaluate(outputs)
        print("Number of errors = " + str(errorNum))

