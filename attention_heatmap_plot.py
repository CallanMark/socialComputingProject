import torch
import os
import networkx as nx
import matplotlib.pyplot as plt
from torch_geometric.utils import to_networkx
from torch_geometric.datasets import UPFD
from modeling import GATForGraphClassification
from modeling import EnsembleGraphClassifier
import numpy as np
# === CONFIG ===
model_path = "./models/Ensemble_politifact_study_20250427_233703"        # Your trained model path
dataset_dir = "./data/UPFD"
dataset_name = "politifact"
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

num_real_to_plot = 3
num_fake_to_plot = 3
save_dir = "./attention_graphs"
os.makedirs(save_dir, exist_ok=True)

# === Load Model ===
def load_model():
    ensemble_model, _ = EnsembleGraphClassifier.load_from_checkpoint(model_path, device=device)
    gat_model = ensemble_model.models[0]  # GAT is always the first model in your ensemble
    gat_model.eval()
    return gat_model

# === Load Dataset ===
def load_dataset():
    return UPFD(root=dataset_dir, name=dataset_name, split="test", feature="bert")

def plot_graph_attention(graph, attentions, title, save_path=None):
    G = to_networkx(graph, to_undirected=False)
    degrees = dict(G.degree())
    
    # Root node (news node with highest out-degree)
    root_node = max(degrees, key=degrees.get)
    
    # Build position dictionary
    pos = {}
    pos[root_node] = (0.0, 1.0)  # root at top

    layer1 = []  # users directly connected to news
    layer2 = []  # users connected via other users

    for node in G.nodes:
        if node == root_node:
            continue
        if G.has_edge(root_node, node):
            layer1.append(node)
        else:
            layer2.append(node)
    
    # Arrange first layer evenly
    if layer1:
        x_positions = np.linspace(-1.5, 1.5, len(layer1))
        for i, node in enumerate(layer1):
            pos[node] = (x_positions[i], 0.7)

    # Arrange second layer evenly
    if layer2:
        x_positions2 = np.linspace(-2.0, 2.0, len(layer2))
        for i, node in enumerate(layer2):
            pos[node] = (x_positions2[i], 0.4)

    # Edge colors
    edge_list = list(G.edges())
    edge_colors = []
    for i, (u, v) in enumerate(edge_list):
        if i < len(attentions):
            edge_colors.append(attentions[i])
        else:
            edge_colors.append(0.0)
    edge_colors = np.array(edge_colors)

    cmap_used = plt.cm.plasma
    norm = plt.Normalize(vmin=np.min(edge_colors), vmax=np.max(edge_colors))

    plt.figure(figsize=(10, 8))

    # Draw edges
    nx.draw_networkx_edges(
        G, pos,
        edge_color=edge_colors,
        edge_cmap=cmap_used,
        edge_vmin=0.0,
        edge_vmax=1.0,
        width=2.5,
        alpha=0.9
    )

    # Draw nodes
    node_colors = []
    node_sizes = []
    for node in G.nodes:
        if node == root_node:
            node_colors.append('gold')
            node_sizes.append(1000)
        else:
            node_colors.append('skyblue')
            node_sizes.append(400)

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes)

    plt.title(title)
    plt.colorbar(plt.cm.ScalarMappable(cmap=cmap_used, norm=norm), label="Attention Weight")
    plt.axis('off')

    if save_path:
        plt.savefig(save_path)
        plt.close()
        print(f"Saved tree graph: {save_path}")
    else:
        plt.show()




# === Main Execution ===
def main():
    model = load_model()
    dataset = load_dataset()

    real_count = 0
    fake_count = 0

    for idx, graph in enumerate(dataset):
        graph = graph.to(device)
        
        # Forward manually with attention
        with torch.no_grad():
            out, attn_info = model.gat.convs[0](graph.x, graph.edge_index, return_attention_weights=True)
            edge_index, attn_weights = attn_info
        
        attentions = attn_weights.squeeze().cpu().numpy()

        label = int(graph.y.item())

        if label == 0 and real_count < num_real_to_plot:
            title = f"Real News Graph {idx}"
            path = os.path.join(save_dir, f"real_{idx}.png")
            plot_graph_attention(graph, attentions, title, path)
            real_count += 1

        if label == 1 and fake_count < num_fake_to_plot:
            title = f"Fake News Graph {idx}"
            path = os.path.join(save_dir, f"fake_{idx}.png")
            plot_graph_attention(graph, attentions, title, path)
            fake_count += 1

        if real_count >= num_real_to_plot and fake_count >= num_fake_to_plot:
            break


if __name__ == "__main__":
    main()
