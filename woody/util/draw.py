#
# Copyright (C) 2015-2017 Fabian Gieseke <fabian.gieseke@di.ku.dk>
# License: GPL v2
#

import os
import numpy

def draw_single_tree(tree, 
                     node_stats=None, 
                     ax=None, 
                     figsize=(200,20),
                     fname="tree.pdf", 
                     with_labels=True, 
                     arrows=False, 
                     edge_width=1.0,
                     font_size=7,
                     alpha=0.5,
                     edges_alpha=1.0,
                     node_size=1000):
    
    try:
        import networkx as nx
        from networkx.drawing.nx_agraph import graphviz_layout
        import matplotlib.pyplot as plt
    except Exception as e:
        raise Exception("Module 'networkx' is required to export the tree structure: %s" % str(e))
    
    d = os.path.dirname(fname)
    if len(d) > 0:
        if not os.path.exists(d):
            os.makedirs(d)     
                        
    if ax is None:
        fig = plt.figure(figsize=figsize)
        ax = fig.add_subplot(111)
                        
    pos = graphviz_layout(tree, prog='dot')
    
    if node_stats is not None:
        lmin = numpy.array([node_stats[i] for i in node_stats.keys()]).min()
        lmax = numpy.array([node_stats[i] for i in node_stats.keys()]).max()
    
    internal_nodes = {'labels':{}, 'sizes':[], 'node_list':[]}
    leaves = {'labels':{}, 'sizes':[], 'node_list':[]}
        
    for i in xrange(len(tree.nodes())):
        if tree.node[i]['is_leaf'] == True:
            leaves['node_list'].append(i)
            if node_stats is not None:
                leaves['labels'][i] = "#" + str(node_stats[i]) + "(" + str(tree.node[i]['leaf_criterion']) + ")"
                leaves['sizes'].append((0.00001 + ((float(node_stats[i] - lmin) / (lmax-lmin)))) *  node_size)
            else:
                leaves['sizes'].append(node_size)
        else:
            internal_nodes['node_list'].append(i)
            internal_nodes['labels'][i] = "" #str(i)
            internal_nodes['sizes'].append(node_size)
    
    # internal nodes
    nx.draw_networkx_nodes(tree, 
            pos,
            nodelist=internal_nodes['node_list'],
            ax=ax, 
            node_color='#0000FF', 
            with_labels=False,
            linewidths=edge_width,
            alpha=alpha,   
            node_size=internal_nodes['sizes'])
    if with_labels == True:
        nx.draw_networkx_labels(tree, pos, internal_nodes['labels'], font_size=7)

    # leaves
    nx.draw_networkx_nodes(tree, 
            pos,
            nodelist=leaves['node_list'],
            ax=ax, 
            node_color='#FF0000',
            with_labels=False,
            linewidths=edge_width,
            alpha=alpha,              
            node_size=leaves['sizes'])
    if with_labels == True:
        nx.draw_networkx_labels(tree, pos, leaves['labels'], font_size=font_size)
    
    # draw edges
    nx.draw_networkx_edges(tree, pos, edge_color='#000000', width=edge_width, alpha=edges_alpha)
    
    plt.axis('off')
    plt.savefig(fname, bbox_inches='tight')
    plt.close()
    plt.close()
    