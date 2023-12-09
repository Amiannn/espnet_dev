import os
import torch
import pandas as pd

from sklearn.cluster import KMeans

from local.utils import read_file
from local.utils import read_json
from local.utils import read_pickle
from local.utils import write_json

from sklearn.manifold import TSNE

import matplotlib.pyplot as plt
import seaborn as sns

import plotly.express as px

PATH       = "./exp/pyf98/librispeech_100_transducer_conformer/valid.loss.ave_10best.pth"
TOKEN_PATH = "./data/en_token_list/bpe_unigram5000/tokens.txt"

CLUSTER_PATH = "./pho_kmean_5.json"

cluster = read_json(CLUSTER_PATH)

ckpt    = torch.load(PATH)
tokens  = [d[0] for d in read_file(TOKEN_PATH, sp='\t')]
s_token = []

# for i in range(len(tokens)):
#     n_class = 0
#     for j in range(len(cluster)):
#         if tokens[i] in cluster[j]['nodes']:
#             n_class = j + 1
#     s_token.append(str(n_class))

lin_out_weight = ckpt['joint_network.lin_out.weight'].numpy()
lin_out_bias   = ckpt['joint_network.lin_out.bias'].numpy()
X = lin_out_weight

kmeans  = KMeans(n_clusters=10, random_state=0, n_init="auto").fit(X)
s_token = list(kmeans.labels_) 

X_embedded = TSNE(n_components=2, verbose=1, perplexity=40, n_iter=2000).fit_transform(X)

df_subset = pd.DataFrame({
    'x': X_embedded[:, 0],
    'y': X_embedded[:, 1],
    'tokens' : tokens,
    's_token': s_token
})


import plotly.graph_objects as go # or plotly.express as px
fig = px.scatter(
    df_subset,
    x="x", 
    y="y", 
    color="s_token", 
    symbol="tokens"
)

# or any Plotly Express function e.g. px.bar(...)
# fig.add_trace( ... )
# fig.update_layout( ... )

from dash import Dash, dcc, html

app = Dash()
app.layout = html.Div([
    dcc.Graph(figure=fig)
])

app.run_server(debug=True, use_reloader=False, host='0.0.0.0') 