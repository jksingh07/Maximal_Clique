## SOLVING MAX CLIQUE PROBLEM USING BRANCH AND BOUND METHOD

# This is an implementation of the branch and bound method to solve the maximum clique problem. 
# The algorithm reads a graph file in the DIMACS format and tries to find the maximum clique in the graph.

'''
Find maxumum clique in given dimacs-format graph
'''
import os
import sys
import threading
from contextlib import contextmanager
import _thread
import time
from openpyxl import Workbook
import networkx as nx




# The TimeoutException is a custom exception class that is used to raise an exception 
# if the algorithm takes more than a specified amount of time to execute.
class TimeoutException(Exception):
    pass

# The timer function is a context manager that is used to set a time limit on the execution of the algorithm.
@contextmanager
def timer(seconds):
    timer = threading.Timer(seconds, lambda: _thread.interrupt_main())
    timer.start()
    try:
        yield
    except KeyboardInterrupt:
        raise TimeoutException()
    finally:
        timer.cancel()

# The capture_time_consumed is a decorator function that is used to measure the time of the function execution.
def capture_time_consumed(f):
    '''
    Measures time of function execution
    '''
    def wrap(*args):
        time1 = time.time()
        ret = f(*args)
        time2 = time.time()
        print('\n{0} function took {1:.3f} ms'.format(
            f.__name__, (time2 - time1) * 1000.0))
        return (ret, '{0:.3f} ms'.format((time2 - time1) * 1000.0))
    return wrap

# The arguments function is used to parse the command line arguments passed to the script.
def arguments():
    import argparse
    parser = argparse.ArgumentParser(
        description='Compute maximum clique for a graph')
    parser.add_argument('--path', type=str, required=True,
                        help='Path to dimacs-format graph file')
    parser.add_argument('--time', type=int, default=60,
                        help='Time limit in seconds')
    parser.add_argument('--test', type=str, default=None, required=False)
    return parser.parse_args()

# The dimacs_graph function reads the graph file in the DIMACS format and returns a graph object.
def dimacs_graph(file_path):
    '''
        Parse .col file and return graph object
    '''
    edges = []
    with open(file_path, 'r') as file:
        for line in file:
            if line.startswith('c'):  # graph description
                print(*line.split()[1:])
            # first line: p name num_of_vertices num_of_edges
            elif line.startswith('p'):
                p, name, vertices_num, edges_num = line.split()
                print('{0} {1} {2}'.format(name, vertices_num, edges_num))
            elif line.startswith('e'):
                _, v1, v2 = line.split()
                edges.append((v1, v2))
            else:
                continue
        return nx.Graph(edges)


# The bronk_algorithm function is an implementation of the Bron–Kerbosch algorithm for finding all maximal cliques in a graph.
def bronk_algorithm(graph, P, R=set(), X=set()):
    '''
    Implementation of Bron–Kerbosch algorithm for finding all maximal cliques in graph
    '''
    if not any((P, X)):
        yield R
    for node in P.copy():
        for r in bronk_algorithm(graph, P.intersection(graph.neighbors(node)),
                       R=R.union(node), X=X.intersection(graph.neighbors(node))):
            yield r
        P.remove(node)
        X.add(node)

# The greedy_clique_heuristic_method function is a greedy search for cliques by iterating through nodes with the highest degree and filtering only their neighbors.
def greedy_clique_heuristic_method(graph):
    '''
    Greedy search for clique iterating by nodes 
    with highest degree and filter only neighbors 
    '''
    K = set()
    nodes = [node[0] for node in sorted(nx.degree(graph),
                                        key=lambda x: x[1], reverse=True)]
    while len(nodes) != 0:
        neigh = list(graph.neighbors(nodes[0]))
        K.add(nodes[0])
        nodes.remove(nodes[0])
        nodes = list(filter(lambda x: x in neigh, nodes))
    return K

# The greedy_coloring_heuristic_method function is a greedy graph coloring heuristic with the degree order rule.
def greedy_coloring_heuristic_method(graph):
    '''
    Greedy graph coloring heuristic with degree order rule
    '''
    color_num = iter(range(0, len(graph)))
    color_map = {}
    used_colors = set()
    nodes = [node[0] for node in sorted(nx.degree(graph),
                                        key=lambda x: x[1], reverse=True)]
    color_map[nodes.pop(0)] = next(color_num)  # color node with color code
    used_colors = {i for i in color_map.values()}
    while len(nodes) != 0:
        node = nodes.pop(0)
        neighbors_colors = {color_map[neighbor] for neighbor in
                            list(filter(lambda x: x in color_map, graph.neighbors(node)))}
        if len(neighbors_colors) == len(used_colors):
            color = next(color_num)
            used_colors.add(color)
            color_map[node] = color
        else:
            color_map[node] = next(iter(used_colors - neighbors_colors))
    return len(used_colors)

# The branching_method function is the branching method procedure.
def branching_method(graph, cur_max_clique_len):
    '''
    branching_method procedure
    '''
    g1, g2 = graph.copy(), graph.copy()
    max_node_degree = len(graph) - 1
    nodes_by_degree = [node for node in sorted(nx.degree(graph),  # All graph nodes sorted by degree (node, degree)
                                               key=lambda x: x[1], reverse=True)]
    # Nodes with (current clique size < degree < max possible degree)
    partial_connected_nodes = list(filter(
        lambda x: x[1] != max_node_degree and x[1] <= max_node_degree, nodes_by_degree))
    # graph without partial connected node with highest degree
    g1.remove_node(partial_connected_nodes[0][0])
    # graph without nodes which is not connected with partial connected node with highest degree
    g2.remove_nodes_from(
        graph.nodes() -
        graph.neighbors(
            partial_connected_nodes[0][0]) - {partial_connected_nodes[0][0]}
    )
    return g1, g2


'''
The branch_bound_maximum_clique function is the main function that implements the branch and bound algorithm. 
It uses the Bron–Kerbosch algorithm and the greedy clique and graph coloring heuristics to find the maximum clique in the graph.
The function starts with the graph, removes nodes that are not in a maximal clique, and creates two subgraphs. 
Then, it recursively applies the procedure to each subgraph until it finds the maximum clique. 
The function returns the maximum clique and the chromatic number of the graph.
'''
def branch_bound_maximum_clique(graph):
    max_clique = greedy_clique_heuristic_method(graph)
    chromatic_number = greedy_coloring_heuristic_method(graph)
    if len(max_clique) == chromatic_number:
        return max_clique
    else:
        g1, g2 = branching_method(graph, len(max_clique))
        return max(branch_bound_maximum_clique(g1), branch_bound_maximum_clique(g2), key=lambda x: len(x))
    
# def branch_bound_maximum_clique(graph):
#     max_clique = greedy_clique_heuristic_method(graph)
#     q = [graph]
#     while len(q) != 0:
#         graph = q.pop(0)
#         for g in branching_method(graph, len(max_clique)):
#             with timer(2):
#                 try:
#                     cliques = list(bronk_algorithm(g, set(g.nodes())))
#                 except TimeoutException:
#                     continue
#             for c in cliques:
#                 if len(c) > len(max_clique):
#                     max_clique = c
#                     q.append(g.subgraph(c))
#     return max_clique



# returns the max clique using branch and bound method.
@capture_time_consumed
def fetch_max_clique(graph):
    return branch_bound_maximum_clique(graph)

# return the file size in a sorted order
def fetch_FilesSize(dirpath):
    return sorted((os.path.join(basedir, filename)
                   for basedir, dirs, files in os.walk(dirpath) for filename in files),
                  key=os.path.getsize)

# It will iterate through all clq files given in the path directory and perform branch and bound method on all to find maximal cliques
def run_test_cases(args):
    import pandas as pd
    test_results = pd.DataFrame(
        columns=['filename', 'nodes', 'edges', 'clique', 'clique length', 'time'])
    files = fetch_FilesSize(args.test)
    try:
        for f in files:
            graph = dimacs_graph(f)
            try:
                with timer(args.time):
                    max_cliques = fetch_max_clique(graph)
                    test_results = test_results.append({'filename': f, 'nodes': graph.number_of_nodes(),
                                                        'edges': graph.number_of_edges(), 'clique': str(max_cliques[0]),
                                                        'clique length': len(max_cliques[0]), 'time': max_cliques[1]},
                                                       ignore_index=True)
                    test_results.to_excel('test_results.xlsx')
            except TimeoutException:
                test_results = test_results.append({'filename': f, 'nodes': graph.number_of_nodes(),
                                                    'edges': graph.number_of_edges(), 'clique': 0,
                                                    'clique length': 0, 'time': 'TIMEOUT'},
                                                   ignore_index=True)
    finally:
        test_results.to_excel('test_results.xlsx')


def main():
    args = arguments()

    if args.test:
        run_test_cases(args)
    else:
        graph = dimacs_graph(args.path)
        try:
            with timer(args.time):
                max_cliques = fetch_max_clique(graph)
                print('\nMaximum clique', max_cliques, '\nlen:', len(max_cliques))
        except TimeoutException:
            print("Timed out!")
            sys.exit(0)


if __name__ == '__main__':
    main()
