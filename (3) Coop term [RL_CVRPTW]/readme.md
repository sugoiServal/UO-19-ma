Unofficial implementation of the 2021 CAIAC paper "Hierarchical Reinforcement Learning for Vehicle Routing Problems with Time Windows"

## Abstract
Vehicle routing problem with time windows (VRPTW) is a practical and complex vehicle routing problem (VRP) which is faced by thousands of companies in logistics and transportation. Usually, VRP is solved by traditional heuristic algorithms. Recently, deep learning models under the reinforcement learning (RL) framework have been proposed to solve variants of VRP. In our study, we propose to use the hierarchical RL to find an optimal policy for generating optimal routes in VRPTW. The hierarchical RL structure includes a low level which generates feasible solutions and a high level which further searches for an optimal solution. Experimental results show that the proposed hierarchical RL model outperforms the non-hierarchical RL model and the heuristic algorithms Google OR-Tools. The proposed model also shows generalization capability in three different scenarios: varied time window constraints, from small-scale to large-scale problems, and generalization across different datasets. The flexible framework of hierarchical RL can also be applied to solve other complex VRPs with multiple objectives.

## Usage
### Requirements
- python3
- torch
- numpy
- matplotlib
### files
- `configs.py` - epxerimental variable setting
- `env.py` - Reinforcement learning environment Setup for VRPTW 
- `gpn.py` - Graph Pointer Networks
- `VRPTW_datagen_mk3.py` - used to generate artificial VRPTW problems (ie, the data)
- `train_low` - training either low or high hierarchy

### train model
```bash
python train_low.py
```
