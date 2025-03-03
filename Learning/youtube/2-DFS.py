# DEPTH FIRST SEARCH

# Key points
# uninformed search technique
# LIFO stack
# deepest nodes are searched first
# incomplete search algorithm
# non-optimal solution
# time complexity: O(V+E)
#       V = number of vertices
#       E = number of edges
#       in AI, O(b^d) 

# recursion
def walk(tree): # pass the root
    if tree is not None: # in python, NULL is None
        print(tree) 
        walk(tree.left)
        walk(tree.right)

# iterative
def walk2(tree, stack):
    stack.append(tree)
    while len(stack) > 0:
        node = stack.pop()
        if node is not None:
            print(node)
            stack.append(node.right)
            stack.append(node.left)

graph= { # not a tree, it's a graph. 
    'a' : ['b', 'c', 'd'], # b c d are connected to a 
    'b' : ['a', 'e', 'd'],
    'c' : ['a', 'd'],
    'd' : ['a', 'b', 'c', 'e'],
    'e' : ['b', 'd']
}

def print_graph(graph):
    print("[", end = "")
    for i in graph:
        print(i + ", ", end = "")
    print("]")

def dfs(start, graph):
    stack =[]
    visited =[]
    stack.append(start)
    while len(stack) > 0:
        print_graph(stack)
        print_graph(visited)
        node = stack.pop(len(stack)-1)
        visited.append(node)
        print(node)
        for neighbour in graph[node]:
            if neighbour not in visited and neighbour not in stack:
                stack.append(neighbour)

visited2 = set()
def dfs2(node, visited, graph):
    if node not in graph:
        return
    if node not in visited:
        print(node)
        visited.add(node)
        for i in graph[node]:
            dfs2(i, visited, graph)



dfs('a', graph) 
dfs2('a', visited2, graph)
