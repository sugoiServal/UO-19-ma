# Exploration on metaheuristics for solving VRPs

## Abstract
Capacitated vehicle routing problem (CVRP) is
an extension to the classic NP problem Travel
salesman problem and can be solved by either exact method, heuristics methods or metaheurisitcs
methods. We explore the nature of the CVRP and
conduct a literature review focusing on the rich
family of metaheuristics as well as how they are
adapted to CVRP. Finally, we implemented an
Ant Colony solver and compared it with Google
OR-Tools and 3 other heuristics solvers.



## Usage

### Package Install:
```bash
python -m pip install --upgrade --user ortools
pip install orderedset
```

### run ortools or aco experiments
```bash
python3 run.py --solver <aco/ ortool>
```
### run heuristics algorithms experiments
```bash
cd ./VeRyPy/
python3 run_heuristics.py
```

### plot results
```bash
cd ../
python3 plot_result.py
```
