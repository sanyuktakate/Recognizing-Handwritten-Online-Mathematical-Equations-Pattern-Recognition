'''
@author: Sanyukta Kate, Pratik Bongale
'''

import random
import math

## Reference: http://www.cs.tau.ac.il/~zwick/grad-algo-13/directed-mst.pdf

class priority_queue:
    """ Class to reprsent a priority queue for a vertex(v) storing input edges to v """

    __slots__ = ('vertex', 'queue', 'graph')

    def __init__(self, *args):
        if len(args) == 2:
            self.vertex = args[0]
            self.graph = args[1]
            self.queue = list()
        else:
            self.vertex = None
            self.queue = list()
            self.graph = None

    def enqueue(self, ele):
        '''
        enqueue as per min priority queue
        :param ele: 2-tuple describing an edge between two vertices
        :return: None
        '''
        self.queue.append(ele)
        self.queue.sort(key=lambda e:self.graph[e[0]][e[1]])
        # self.queue = sorted(self.queue, key=lambda e:self.graph[e[0]][e[1]])

    def dequeue(self):
        return self.queue.pop(0)

    def is_empty(self):
        return len(self.queue) == 0

    def meld(self, pq_other):
        while not pq_other.is_empty():
            self.enqueue( pq_other.dequeue() )

    def __str__(self):
        return str(self.queue)


class Graph:

    __slots__ = ('graph', 'pq', 'in_edge', 'const', 'prev', 'parent', 'children', 'root', 'R')

    def __init__(self, *args):

        if len(args) == 2:
            self.graph = args[0]
            self.root = args[1]
            self.pq = None
            self.in_edge = None
            self.const = None
            self.prev = None
            self.parent = None
            self.children = None
            self.R = None

    def contract(self):
        self.initialize()

        vertices = list(self.graph.keys() - self.root)
        r = random.randint(0, len(vertices)-1)

        a = vertices[r]     # select an arbitrary non-root vertex
        k = 1

        while not self.pq[a].is_empty():
            (u, v) = self.pq[a].dequeue()
            b = self.find(u)     # find super-vertex that currently contains u

            if a != b :     # check for a self loop(a and b belong to the same super-vertex)
                self.in_edge[a] = (u, v)    # the best incoming edge for a (in the original graph)
                self.prev[a] = b

                if self.in_edge[u] is None: # extend path
                    a = b
                else:   # cycle formed
                    c = 'c' + str(k)    # create a new vertex
                    k += 1
                    self.init_vertex(c)

                    while self.parent[a] is None:
                        self.parent[a] = c

                        e = self.in_edge[a]    # best edge to vertex a
                        self.const[a] = -self.graph[e[0]][e[1]]
                        self.children[c].append(a)
                        self.pq[c].meld(self.pq[a])
                        a = self.prev[a]
                    a = c

    def expand(self, root):
        self.R = []
        self.dismantle(root)

        while self.R:
            c = self.R.pop(0)
            ie = self.in_edge[c]
            u = ie[0]
            v = ie[1]
            self.in_edge[v] = (u, v)
            self.dismantle(v)

        mst = []
        for e in self.graph.keys() - {root}:
            mst.append(self.in_edge[e])

        return mst

    def initialize(self):

        for u in self.graph:    # for each vertex
            self.init_vertex(u)

        for u in self.graph:
            for v in self.graph[u]:
                self.pq[v].enqueue((u,v))

    def init_vertex(self, u):

        if self.in_edge is None:
            self.in_edge = { u : None }
        else:
            self.in_edge[u] = None

        if self.const is None:
            self.const = {u: 0}
        else:
            self.const[u] = 0

        if self.prev is None:
            self.prev = { u : None }
        else:
            self.prev[u] = None

        if self.parent is None:
            self.parent = { u : None }
        else:
            self.parent[u] = None

        if self.children is None:
            self.children = { u : [] }
        else:
            self.children[u] = []

        if self.pq is None:
            self.pq = { u : priority_queue(u, self.graph) }
        else:
            self.pq[u] = priority_queue(u, self.graph)

    def find(self, u):

        while self.parent[u] is not None:
            u = self.parent[u]

        return u

    def weight(self, u, v):
        w = self.graph[u][v]

        while self.parent[v] is not None:
            w = w + self.const[v]
            v = self.parent[v]

        return w

    def dismantle(self, u):
        while self.parent[u] is not None:
            for v in self.children[self.parent[u]]:
                if v != u:
                    self.parent[v] = None
                    if self.children[v]:
                        self.R.append(v)

            u = self.parent[u]

def get_mst(G, root):

    # print(G)
    g = Graph(G, root)
    g.contract()
    return g.expand(root)

def is_sc(graph):

    # select a vertex v
    # check if dfs visits all other vertices from v
    # rg = reverse graph
    # check if the same vertex can still reach all vertices

    vertices = list(graph.keys())
    r = random.randint(0, len(vertices) - 1)
    v = vertices[r]  # select an arbitrary vertex

    visited = set()     # set of visited vertices
    isc_helper(graph, v, visited)

    if graph.keys() - visited:      # if every vertex of graph is not in visited
        return False

    rev_graph = get_reverse_graph(graph)
    visited = set()
    isc_helper(rev_graph, v, visited)

    if rev_graph.keys() - visited:      # if every vertex of graph is visited, difference will be empty set
        return False

    return True


def isc_helper(graph, v, visited):

    visited.add(v)
    for nbor in graph[v]:
        if nbor not in visited:
            isc_helper(graph, nbor, visited)

    return visited

def get_reverse_graph(graph):

    rev_g = {}

    for node in graph:
        for nbor in graph[node]:
            if nbor in rev_g:
                rev_g[nbor].add(node)       # flip/reverse the edge
            else:
                rev_g[nbor] =  { node }

    return rev_g

def negate_wts(g):
    for u in g:
        for v in g[u]:
            g[u][v] = -g[u][v]

def wt_sum(g, edges):
    sum = 0

    for (u,v) in edges:
        sum += g[u][v]

    return sum

def test_edmonds(G, root):
    ## Testing Edmonds algorithm ##
    in_edges = get_mst(G, root)
    print(in_edges)
    print(wt_sum(G, in_edges))

def test_priority_queue(G):
    ## testing priority queue ##
    pq = priority_queue('b', G)
    pq.enqueue(('h', 'b'))
    pq.enqueue(('a', 'b'))
    pq.enqueue(('g', 'b'))
    print(pq)
    print(pq.dequeue())

    pq = priority_queue('c', G)
    pq.enqueue(('b', 'c'))
    pq.enqueue(('e', 'c'))
    print(pq)
    print(pq.dequeue())

def test_isc(G):
    ## Testing strongly connected graph ##
    print(is_sc(G))

if __name__ == '__main__':

    G1 = {
            'a':{'b':5, 'h':11},
            'b':{'c':3, 'f':13, 'a':40},
            'c':{'d':12, 'f':9},
            'd':{'e':1},
            'e':{'c':4},
            'f':{'e':8, 'g':7},
            'g':{'b':2, 'h':10},
            'h':{'b':6}
         }

    G2 = {
            'r': {'a':5},
            'a': {'b':10},
            'b': {'r':11}
        }


    G3 = {
        'a': {'b': 5, 'h': 11},
        'b': {'c': 3, 'f': 13, 'a': 1},
        'c': {'d': 12, 'f': 9},
        'd': {'e': 1},
        'e': {'c': 4},
        'f': {'e': 8, 'g': 7},
        'g': {'b': 2, 'h': 10},
        'h': {'b': 6}

    }

    G3['dmy'] = {}
    d_out_cost = -math.inf      # the cose of egdes goind out from dummy node

    G3['dmy']['a'] = d_out_cost
    G3['dmy']['b'] = d_out_cost
    G3['dmy']['c'] = d_out_cost
    G3['dmy']['d'] = d_out_cost
    G3['dmy']['e'] = d_out_cost
    G3['dmy']['f'] = d_out_cost
    G3['dmy']['g'] = d_out_cost
    G3['dmy']['h'] = d_out_cost

    G3['a']['dmy'] = -math.inf

    G4 = {
        'root':{'v1':5, 'v2':1, 'v3':1},
        'v1': {'v2':11, 'v3':4},
        'v2': {'v1':10, 'v3':5},
        'v3': {'v1':9, 'v2':8},
    }

    G4['dmy'] = {}
    val = -1
    G4['dmy']['root'] = val
    G4['dmy']['v1'] = val
    G4['dmy']['v2'] = val
    G4['dmy']['v3'] = val

    G4['v1']['root'] = -1
    G4['root']['dmy'] = -1


    G5 = {
        '5': {'6':0.99, '7':0.99},
        '6': {'5':0.31, '7':0.98},
        '7': {'5':0.70, '6':0.39}
    }

    # G5['dmy'] = {}
    # val = 0
    # G5['dmy']['5'] = val
    # G5['dmy']['6'] = val
    # G5['dmy']['7'] = val
    #
    # G5['5']['dmy'] = val

    negate_wts(G5)
    test_isc(G5)
    test_edmonds(G5, '5')
    test_edmonds(G5, '6')
    test_edmonds(G5, '7')

