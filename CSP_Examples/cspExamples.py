import sys
import cspProblem
from operator import lt,ne,eq,gt

# Hide warnings in the matplotlib sections
import warnings
warnings.filterwarnings("ignore")
import matplotlib.pyplot as plt
import numpy as np

def ne_(val):
    """not equal value"""
    # nev = lambda x: x != val # alternative definition
    # nev = partial(neq,val) # another alternative definition
    def nev(x):
        return val != x
    nev.__name__ = f"{val} != " # name of the function
    return nev


def is_(val):
    """is a value"""
    # isv = lambda x: x == val # alternative definition
    # isv = partial(eq,val) # another alternative definition
    def isv(x):
        return val == x
    isv.__name__ = f"{val} == "
    return isv


def example1():
    X = cspProblem.Variable('X', {1,2,3})
    Y = cspProblem.Variable('Y', {1,2,3})
    Z = cspProblem.Variable('Z', {1,2,3})
    csp0 = cspProblem.CSP("csp0", {X,Y,Z}, [cspProblem.Constraint([X,Y],lt), cspProblem.Constraint([Y,Z],lt)])

    A = cspProblem.Variable('A', {1,2,3,4}, position=(0.2,0.9))
    B = cspProblem.Variable('B', {1,2,3,4}, position=(0.8,0.9))
    C = cspProblem.Variable('C', {1,2,3,4}, position=(1,0.4))
    C0 = cspProblem.Constraint([A,B], lt, "A < B", position=(0.4,0.3))
    C1 = cspProblem.Constraint([B], ne_(2), "B != 2", position=(1,0.9))
    C2 = cspProblem.Constraint([B,C], lt, "B < C", position=(0.6,0.1))
    csp1 = cspProblem.CSP("csp1", {A, B, C}, [C0, C1, C2])
    csp1.show()
#csp1s = CSP("csp1s", {A, B, C}, [C0, C2]) # A<B, B<C


def example2():
    A = cspProblem.Variable('A', {1,2,3,4}, position=(0.2,0.9))
    B = cspProblem.Variable('B', {1,2,3,4}, position=(0.8,0.9))
    C = cspProblem.Variable('C', {1,2,3,4}, position=(1,0.4))
    D = cspProblem.Variable('D', {1, 2, 3, 4}, position=(0, 0.4))
    E = cspProblem.Variable('E', {1, 2, 3, 4}, position=(0.5, 0))

    csp2 = cspProblem.CSP("csp2", {A, B, C, D, E},
        [cspProblem.Constraint([B], ne_(3), "B != 3", position=(1, 0.9)),
        cspProblem.Constraint([C], ne_(2), "C != 2", position=(1, 0.2)),
        cspProblem.Constraint([A, B], ne, "A != B"),
        cspProblem.Constraint([B, C], ne, "A != C"),
        cspProblem.Constraint([C, D], lt, "C < D"),
        cspProblem.Constraint([A, D], eq, "A = D"),
        cspProblem.Constraint([E, A], lt, "E < A"),
        cspProblem.Constraint([E, B], lt, "E < B"),
        cspProblem.Constraint([E, C], lt, "E < C"),
        cspProblem.Constraint([E, D], lt, "E < D"),
        cspProblem.Constraint([B, D], ne, "B != D")])

    csp2.show()


def example3():
    A = cspProblem.Variable('A', {1,2,3,4}, position=(0.2,0.9))
    B = cspProblem.Variable('B', {1,2,3,4}, position=(0.8,0.9))
    C = cspProblem.Variable('C', {1,2,3,4}, position=(1,0.4))
    D = cspProblem.Variable('D', {1, 2, 3, 4}, position=(0, 0.4))
    E = cspProblem.Variable('E', {1, 2, 3, 4}, position=(0.5, 0))
    csp3 = cspProblem.CSP("csp3", {A, B, C, D, E},
               [cspProblem.Constraint([A, B], ne, "A != B"), cspProblem.Constraint([A, D], lt, "A < D"),
                cspProblem.Constraint([A, E], lambda a, e: (a - e) % 2 == 1, "A-E is odd"),
                cspProblem.Constraint([B, E], lt, "B < E"),
                cspProblem.Constraint([D, C], lt, "D < C"),
                cspProblem.Constraint([C, E], ne, "C != E"),
                cspProblem.Constraint([D, E], ne, "D != E")])

    csp3.show()

from csp import *

def usingAC3():
    neighbors = parse_neighbors('A: B; B: ')
    domains = {'A': [0, 1, 2, 3, 4], 'B': [0, 1, 2, 3, 4]}
    constraints = lambda X, x, Y, y: x % 2 == 0 and (x + y) == 4 and y % 2 != 0
    removals = []

    csp = CSP(variables=None, domains=domains, neighbors=neighbors, constraints=constraints)

    result, checks = AC3(csp, removals=removals)
    print("res"
          "ult for AC3 consistency implementation is ", result, " and number of checks ", checks)


def plotChessboard(solution, title=None) ->None:
    size = len(solution)
    chessboard = np.zeros((size, size))

    chessboard[1::2, 0::2] = 1
    chessboard[0::2, 1::2] = 1

    plt.ioff()
    plt.imshow(chessboard, cmap='binary')
    if title is not None:
        plt.title(title)

    for (j, i) in solution.items():
        plt.text(i, j, '♕', fontsize=20, ha='center', va='center', color='black' if (i - j) % 2 == 0 else 'yellow')

    plt.show()


def nQueens_hillClimbing():
    for _ in range(100):
        eight_queens = NQueensCSP(num_queens)
        solution = min_conflicts(eight_queens)
        if solution is not None:
            #print("solution found")
            #eight_queens.display(solution)
            return solution
        else:
            print("eightQueen: no solution found!")
            return None

def nQueens_backtracking_mrv_forwardChecking():
    for _ in range(100):
        eight_queens = NQueensCSP(num_queens)
        solution = backtracking_search(eight_queens, select_unassigned_variable=mrv, inference=forward_checking)
        if solution is not None:
            #print("solution found")
            #eight_queens.display(solution)
            return solution
        else:
            print("eightQueen: no solution found!")
            return None



def nQueens_backtracking_mrv_mac():
    for _ in range(100):
        eight_queens = NQueensCSP(num_queens)
        solution = backtracking_search(eight_queens, select_unassigned_variable=mrv, inference=mac)
        if solution is not None:
            #print("solution found")
            #eight_queens.display(solution)
            return solution
        else:
            print("eightQueen: no solution found!")
            return None

num_queens = 5
import time
if __name__ == "__main__":
    #example1()
    #pass
    #example3()
    #usingAC3()


    if len(sys.argv) > 1:
        num_queens = int(sys.argv[1])

    print("num_queens = ", num_queens)

    sols1 = []
    sols2 = []
    sols3 = []
    print("Hillclimbing method:")
    st = time.time()
    for _ in range(10):
        solution = nQueens_hillClimbing()
        #plotChessboard(solution)
        sols1.append(solution)
    elapsed_time1 = (time.time() - st) * 1000   # elapsed time in millisecond
    plotChessboard(solution, 'HillClimbing_minConflict')
    print("backtracking_mrv_forwardChecking:")
    st = time.time()
    for _ in range(10):
        solution = nQueens_backtracking_mrv_forwardChecking()
        sols2.append(solution)
    elapsed_time2 = (time.time() - st) * 1000   # elapsed time in millisecond
    plotChessboard(solution, 'BackTracking_mrv_forwardChecking')


    print("backtracking_mrv_mac:")
    st = time.time()
    for _ in range(10):
        solution = nQueens_backtracking_mrv_mac()
        sols3.append(solution)
    elapsed_time3 = (time.time() - st) * 1000   # elapsed time in millisecond
    plotChessboard(solution, 'BackTracking_mrv_mac')

    print("Execution times for 10 runs: \nHillClimbing_minConflict: ", elapsed_time1, "\n MRV_ForwardChecking:", elapsed_time2, "\n MRV_MAC:", elapsed_time3)





