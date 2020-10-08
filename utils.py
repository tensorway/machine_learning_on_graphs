import torch_geometric
import matplotlib.pyplot as plt
import numpy as np
from torch_geometric.datasets import WikiCS
import random
import torch

def random_graph(n, m):
    edges = torch.randint(0, n, torch.Size((2, m)))
    x = torch.range(0, n-1).unsqueeze(-1)
    return Data(edge_index=edges, x=x)

def create_edges(tensor, offset, mod):
    b = (tensor + offset)%mod
    return torch.cat((tensor, b), dim=0).long()
def small_world_graph(n, m):
    beg = torch.range(0, n-1).unsqueeze(0)
    a1 = create_edges(beg, -2, n)
    a2 = create_edges(beg, -1, n)
    a3 = create_edges(beg, +1, n)
    a4 = create_edges(beg, +2, n)
    a5 = random_graph(n, m).edge_index
    
    edges = torch.cat((a1, a2, a3, a4, a5), dim=1)
    x = torch.range(0, n-1).unsqueeze(-1)
    return Data(edge_index=edges, x=x)
def plt_log_degree(deg, log=True, range_max=100, range_min=0):
    outd = deg
    y, _ = np.histogram(outd.numpy(),  bins=np.array(list(range(range_min, range_max))))
    x = np.array(list(range(range_min, range_max-1)))
    if log:
        x, y = np.log(x), np.log(y+1e-1)
    plt.plot(x, y)
def make_adj_matrix(graph):
    mat = torch.zeros(graph.num_nodes, graph.num_nodes, dtype=torch.bool)
    print(mat.shape)
    for a, b in zip(graph.edge_index[0], graph.edge_index[1]):
        if a != b and mat[a, b]==0 and mat[b, a]==0:
            mat[a, b] = 1
            mat[b, a] = 1
    return mat


def count_out_degree(data, aslist=False):
    cnt = torch.zeros(data.x.shape[0], )
    for edge in data.edge_index[0]:
        cnt[edge] += 1
    if aslist:
        l = [int(t.item()) for t in cnt]
        return l
    return cnt
def count_in_degree(data, aslist=False):
    cnt = torch.zeros(data.x.shape[0], )
    for edge in data.edge_index[1]:
        cnt[edge] += 1
    if aslist:
        l = [int(t.item()) for t in cnt]
        return l
    return cnt

def prob_clustering_coeff(graph, adj_mat, n=10000):
    nhit = 0.001
    n2 = n
    while n2 > 0:
        ii = random.randint(0, len(adj_mat)-1)
        frst = torch.tensor(list(range(len(adj_mat))))
        idcs = torch.masked_select(frst, adj_mat[ii])
        perm = torch.randperm(idcs.shape[0])
        idx = perm[:2]
        samples = idcs[idx]
        if len(samples) >= 2:
            nhit += adj_mat[samples[0], samples[1]].float()
            n2 -= 1
        print(100-n2/n*100, "%", end='\r')
    return nhit/n




def create_pointer_graph(graph):
    l = [([], []) for i in range(graph.num_nodes)]
    for fromx, tox in zip(graph.edge_index[0], graph.edge_index[1]):
        l[fromx][0].append(tox.item())
        l[tox][1].append(fromx.item())
    return l
def get_in_out_ego_edges(ego_node, pointer_graph):
    #counts in edges twice and only out out edges no out in edges 
    inedges, outedges = 0, 0
    ego_net_dict = {}
    ego_net_dict[ego_node] = 1
    from itertools import chain
    for node in chain(pointer_graph[ego_node][0], pointer_graph[ego_node][1]):
        inedges += 1
        ego_net_dict[node] = 1

    for node in chain(pointer_graph[ego_node][0], pointer_graph[ego_node][1]):
        for node2 in chain(pointer_graph[node][0], pointer_graph[node][1]):
            if node2 in ego_net_dict:
                inedges += 1
            else:
                outedges += 1
    return inedges, outedges
def agregate_sum_mean(pointer_graph, data, niter):
    for i in range(niter):
        datalist = []
        for node in range(data.num_nodes):
            l = pointer_graph[node][1]
            idx = torch.tensor(l, dtype=torch.long)
            vec = data.x[idx]
            sum, mean = vec.sum(dim=0), vec.mean(dim=0)
            datalist.append(torch.cat((data.x[node], mean, sum)).unsqueeze(0))
            print((i+1)/niter*(node+1)/data.num_nodes, "%", end='\r')
        # print(datalist)
        data.x = torch.cat(datalist, dim=0)

