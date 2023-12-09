import os
import random
import numpy as np
import eng_to_ipa as ipa

from local.utils import read_file
from local.utils import read_json
from local.utils import read_pickle
from local.utils import write_json

from fuzzywuzzy import fuzz
from fuzzywuzzy import process

from tqdm import tqdm

TOKEN_PATH = "./data/en_token_list/bpe_unigram5000/tokens.txt"

# a = "Hello"
# b = "How are you"

# a_pho = ipa.convert(a.lower())
# b_pho = ipa.convert(b.lower())

# sim = fuzz.ratio(a_pho, b_pho)
# print(sim)

def distance(a, b):
    return 1 - (fuzz.ratio(a, b) / 100)

def pair_wise_distance(list_a, list_b):
    dist_table = np.zeros([len(list_a), len(list_b)])
    for i in range(len(list_a)):
        for j in range(len(list_b)):
            dist_table[i][j] = distance(list_a[i], list_b[j])
    return dist_table

def kmeans_one_step(
    centers, 
    cluster_len, 
    cluster_nodes, 
    masked_nodes, 
    dist_table, 
    argmax_dist
):
    # top n step
    for i in range(len(centers)):
        for j in range(dist_table.shape[-1]):
            element = argmax_dist[centers[i], j]
            if len(cluster_nodes[i]) >= cluster_len:
                break
            if masked_nodes[element]:
                continue
            cluster_nodes[i].append(element)
            masked_nodes[element] = True
    for j in range(len(masked_nodes)):
        if not masked_nodes[j]:
            cluster_nodes[-1].append(j)
    # avg step
    _centers = []
    for i in range(len(cluster_nodes)):
        scores = []
        for j in range(len(cluster_nodes[i])):
            element = cluster_nodes[i][j]
            row_idx = [element for _ in range(len(cluster_nodes[i]))]
            col_idx = cluster_nodes[i]
            scores.append(np.sum(dist_table[row_idx, col_idx]))
        _centers.append(cluster_nodes[i][np.argmin(scores)])
    return _centers, cluster_nodes

def kmeans(pho_tokens, k):
    # initial choosing k samples
    cluster_len   = len(pho_tokens) // k
    
    indexis = list(range(len(pho_tokens)))
    centers = random.choices(indexis, k=k)
    
    dist_table  = pair_wise_distance(pho_tokens, pho_tokens)
    argmax_dist = np.argsort(dist_table, axis=1) 

    print('start clustering!')
    for _ in tqdm(range(300)):
        masked_nodes  = [False if p not in centers else True for p in range(len(pho_tokens))]
        cluster_nodes = [[centers[i]] for i in range(k)]
        centers, cluster_nodes = kmeans_one_step(
            centers, 
            cluster_len, 
            cluster_nodes, 
            masked_nodes, 
            dist_table, 
            argmax_dist
        )
    return centers, cluster_nodes

if __name__ == '__main__':
    # tokens     = [d[0] for d in read_file(TOKEN_PATH, sp='\t')[2:1002]]
    tokens     = [d[0] for d in read_file(TOKEN_PATH, sp='\t')[2:]]
    pho_tokens = read_json('./pho.json') 
    # pho_tokens = [ipa.convert(t.replace('▁', '').lower()) for t in tqdm(tokens)]
    
    # output_path = './pho.json'
    # write_json(output_path, pho_tokens)

    # pho_tokens = [t.replace('▁', '').lower() for t in tqdm(tokens)]
    k = 5
    centers, cluster_nodes = kmeans(pho_tokens, k)

    result = []
    for i in range(len(centers)):
        center_token   = tokens[centers[i]]
        cluster_tokens = [tokens[j] for j in cluster_nodes[i]]
        result.append({
            'center': center_token,
            'nodes' : cluster_tokens
        })
    
    output_path = f'./pho_kmean_{k}.json'
    write_json(output_path, result)