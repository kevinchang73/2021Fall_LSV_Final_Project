# 2021Fall_LSV_Final_Project
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
