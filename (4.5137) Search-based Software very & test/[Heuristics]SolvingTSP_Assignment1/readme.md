# Travelling Salesman Problem (TSP) Heuristics Solver


# Usage:
### Requirements
-  Python3
-  Numpy
-  Matplotlib

### `python TSP_solver.py`
by default, the program solve the a280.txt instance under the current work directory with hill climbing, and print the fitness on the screen.

### options:
- specify a metaheuristic, you can either use `'HC' (hill climbing)`, or `'SA'(simulated annealing)`

```bash
$ python TSP_solver.py --metaheuristic <option>
```

- specify another file to test, given the absolute path to the file. The implementation only support 2D Euclidean type in TSPLIB

```bash
$ python TSP_solver.py --file <option>
```

- you can modify the solver inside the script to save solution to .csv or plot the route.