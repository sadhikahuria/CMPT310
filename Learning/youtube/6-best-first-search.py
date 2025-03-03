"""
Alorithm: Best First Search
- implemented using priority queue based on the heuristic value
- pop the node with the lowest heuristic value
- if node is goal node, return the path
- otherwise, expand the node and add the children to the queue


"""
graph = {
    'a' : [['b', 11], ['c', 14] , ['d', 7]],
    'b' : [['a', 11], ['e', 15]],
    'c' : [['a', 14], ['e', 8], ['f', 10]],
    'd' : [['a', 7], ['f', 25]],
    'e' : [['b', 15], ['c', 8], ['h', 9]],
    'f' : [['c', 10], ['d', 25], ['g', 12]],
    'g' : [['f', 12], ['h', 10]]
}

heuristic = {
    'a' : 40,
    'b' : 32,
    'c' : 25,
    'd' : 35,
    'e' : 19,
    'f' : 17,
    'g' : 0,
    'h' : 10
}

def best_first_search(graph, start, goal):
    queue = []
    queue.append(start)
    path=[]
    while queue:
        node = queue.pop(0)
        path.append(node)
        if node == goal:
            break
        for neighbour in graph[node]:
            # append based on heuristic value
            queue.append(neighbour[0])
        queue.sort(key = lambda x: heuristic[x])
    for node in path:
        print(node, end=' -> ')


best_first_search(graph, 'a', 'g') # a -> c -> f -> g ->
