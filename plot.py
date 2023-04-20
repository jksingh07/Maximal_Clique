import networkx as nx
import matplotlib.pyplot as plt

# read the input file
filename = './samples/le450_5a.col'
with open(filename, 'r') as f:
    lines = f.readlines()

# parse the input file
G = nx.Graph()
for line in lines:
    if line.startswith('e'):
        nodes = line.split()[1:]
        G.add_edge(int(nodes[0]), int(nodes[1]))

# plot the graph
nx.draw(G, with_labels=True)
plt.show()

