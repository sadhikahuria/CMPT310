# BREATH FIRST SEARCH

# Key points
# uninformed search technique
# FIFO queue
# shallowest nodes are searched first
# complete search algorithm
# optimal solution
# time complexity: O(V+E)
#       V = number of vertices
#       E = number of edges
# space complexity: O(V)
# in AI, O(b^d) where 
#       b is branching factor (average number of children of a node)
#       d is depth of shallowest solution


graph = {
    '5' : ['3', '7'], # 5 is the parent of 3 and 7
    '3' : ['2', '4'],
    '7' : ['8'],
    '2' : [],
    '4' : ['8'],
    '8' : []
}

visited = []
queue =[]

def bfs( visited, graph, node):
    queue.append(node)
    visited.append(node)

    while queue: # while queue is not empty
        m = queue.pop(0) # pop for the left side, could use pop(0) as well
        print(m) # if i do print(m, end = " "), it will print in a single line

        for neighbour in graph[m]:
        # python will find all the neighbours of m, so if I write graph['5'], it will automatically find '5' in the queue, and look at the neighbours. 
            if neighbour not in visited:
                queue.append(neighbour)
                visited.append(neighbour)

    
bfs(visited, graph, '5')
# print(graph['5']) 
