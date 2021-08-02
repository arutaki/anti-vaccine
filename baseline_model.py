import os
import numpy as np
import pandas as pd
import networkx as nx
import matplotlib.pyplot as plt

from collections import Counter
from tqdm import  tqdm

## Functions
def calc_support(subs, G):
    subs_pol = [G.nodes[sub]['polarity'] for sub in subs]
    count = Counter(subs_pol)
    return count['p'] / sum(count.values())

def calc_prob_r(l):
    return 1 / (1 + np.exp(-10 * (l - 0.5)))

## Graph Initialization

# Setup the graphs and the data
nodes_df = pd.read_csv('dataset/node_table.csv')
nodes_df = nodes_df.drop(nodes_df.columns[0], axis=1)
nodes_df["banner_cdc"] = nodes_df["banner_cdc"].fillna(False)
selectors = list(nodes_df["selector"])
id_selector_dict = {i : selectors[i] for i in range(len(selectors))}
selector_id_dict = {val : key for key, val in id_selector_dict.items()}
G = nx.DiGraph()
edges_df = pd.read_csv('dataset/edge_table.csv')
G.add_nodes_from(range(len(selectors)))
edges_list = [(selector_id_dict[edges_df.source[i]], 
                    selector_id_dict[edges_df.target[i]]) for i in range(len(edges_df))]
for edge in edges_list:
    G.add_edge(edge[0], edge[1])
init_props = {selector_id_dict[nodes_df.selector[i]] : {"id" : i, "polarity" : nodes_df.polarity[i][0],
                    "banner" : nodes_df.banner_cdc[i]} for i in range(len(nodes_df))}
id_polarity_dict = {i : nodes_df.polarity[i][0] for i in range(len(nodes_df))}
nx.set_node_attributes(G, init_props)

# Changing all neutral nodes without neighbors to pro-vaccine
for i in range(len(G)):
    neighbor = list(G[i])
    if G.nodes[i]['polarity'] == 'n' and not neighbor:
        G.nodes[i]['polarity'] = 'p'

# Changing all neutral nodes with neighbors to majority of its non-neutral neighbors
new_polar = {}
for i in range(len(G.nodes())):
    neighbor = list(G[i])
    if G.nodes[i]['polarity'] == 'n' and neighbor:
        neighbor_pol = []
        for j in neighbor:
            neighbor_pol.append(G.nodes[j]['polarity'])
        count = Counter(neighbor_pol)
        if count['a'] > count['p']:
            new_polar[i] = 'a'
        else:
            new_polar[i] = 'p'
    
for i in new_polar.keys():
    G.nodes[i]['polarity'] = new_polar[i]

## Setting up parameters and placeholders

# Parameters
pol_l = ['a', 'p']
q_l = {'low' : 0.6, 'med' : 0.75, 'high' : 0.9}
l_l = [0.05 * i for i in range(21)]

# Creating column for the dataframe
cols = [str(5 * i) for i in range(21)]
cols = ['spec'] + cols 

## Simulation

for repeat in tqdm(range(20)):
    # Create dataframe
    df = pd.DataFrame(columns=cols)

    # The simulation starts
    for pol in pol_l:
        for q_key in q_l.keys():
            q = q_l[q_key]
            spec = f"{pol}_{q_key}"
            entry = [spec]
            for l in l_l:
                temp = []

                # Simulating for each entry point
                for i in range(1326):
                    
                    # Generating user data
                    user = {'pol' : pol, 'prob_r' : calc_prob_r(l), 'prob_a' : q, 'subs' : [i]}
                    
                    # Simulating for each time step
                    for time in range(50):
                        if not G[i]:
                            break
                        sub = np.random.choice(user['subs'])
                        if not G[sub]:
                            continue
                        sub_rec = np.random.choice(list(G[sub]))
                        if sub_rec in user['subs']:
                            continue
                        
                        ## Receive
                        if np.random.rand() < user['prob_r']:
                            receive = True
                        else:
                            receive = False
                        
                        ## Accept
                        if receive:
                            if G.nodes[sub_rec]['polarity'] == user['pol']:
                                if np.random.rand() < user['prob_a']:
                                    user['subs'].append(sub_rec)
                            else:
                                if np.random.rand() < 1 - user['prob_a']:
                                    user['subs'].append(sub_rec)
                        else:
                            if np.random.rand() < 0.5:
                                user['subs'].append(sub_rec)
                    
                    ## Sample
                    sup = calc_support(user['subs'], G)
                    temp.append(sup)
                
                ## Results for all entry points
                avg_sup = np.average(temp)
                entry.append(avg_sup)
            
            ## Updating the dataframe
            df.loc[len(df)] = entry

    ## Post-processing
    
    dest = f'/Users/arutaki/Documents/research/anti-vaccine/anti-anti-vaccine/base/baseline_model-{repeat}.csv'
    df.to_csv(dest, index=False)