# Learning Weights and Thresholds of Threshold Logic Networks

Threshold logic network is more viable nowadays due to its compactness and strong bind to neural network applications. However, the problem of weights and thresholds determination still remains open. In this work, we introduce machine learning and propose two approaches, the function-based approach and the network-based approach, to solve the problem. Experimental results show that our method achieves near 80% accuracy in the function-based approach and 70% to 90% accuracy in the network-based approach.

Please visit https://kevinchang73.github.io/projects/ for detailed information.

## Input file formats
`nodeID isPI isPO` \
`<nodes>` \
`edges` \
`<edges> (u.id -> v.id)` \
`level` \
`<level>`

**Note**
1. Node id must be 1,2,3,...,n
2. 1,2,3,...,n must be a valid topological order; that is, u < v for all edge (u,v).
3. PIs should also be included in the node lists, though they are not actually TLGs.

**Compile Excute the program**
`python main.py read inputs/case1` or `python main.py read inputs/case2` \
Adjust the learning rate in `agent.py: 34` \
Adjust number of epochs and batch size in `main.py: 17.18` \
Adjust coefficient in `TLN_env/TLN/TLN.py: 38` \
Adjust number of functions in `main.py: 37` (Uncomment the line and adjust the number) \
