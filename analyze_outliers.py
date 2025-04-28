import torch
import os
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from scipy.spatial.distance import euclidean
from torch_geometric.datasets import UPFD
from torch_geometric.utils import to_networkx

# === CONFIG ===
embedding_dir = "./saved_embeddings"
dataset_dir = "./data/UPFD"
dataset_name = "politifact"
model_choice = "combined_with_gat"  # or combined_without_gat
save_dir = "./separate_outlier_graphs"
os.makedirs(save_dir, exist_ok=True)

num_outliers = 10
num_graphs_to_plot = 5

# === Load embeddings and labels ===
def load_embeddings_and_labels():
    gat = torch.load(os.path.join(embedding_dir, "gat_embeddings.pt"))
    rgcn = torch.load(os.path.join(embedding_dir, "rgcn_embeddings.pt"))
    han = torch.load(os.path.join(embedding_dir, "han_embeddings.pt"))
    labels = torch.load(os.path.join(embedding_dir, "test_labels.pt"))

    if model_choice == "combined_with_gat":
        embeddings = torch.cat([gat, rgcn, han], dim=1)
    else:
        embeddings = torch.cat([rgcn, han], dim=1)

    return embeddings.numpy(), labels.numpy()

# === Load dataset ===
def load_dataset():
    return UPFD(root=dataset_dir, name=dataset_name, split='test', feature='bert')

# === Compute centroid for given class ===
def compute_centroid(embeddings, labels, target_class):
    class_embeds = embeddings[labels == target_class]
    return np.mean(class_embeds, axis=0)

# === Find top outliers within a class ===
def find_outliers(embeddings, labels, centroid, target_class, k):
    distances = []
    for i, emb in enumerate(embeddings):
        if labels[i] == target_class:
            dist = euclidean(emb, centroid)
            distances.append((i, dist))
    distances.sort(key=lambda x: x[1], reverse=True)
    return [idx for idx, dist in distances[:k]]

# === Find normal graphs (closest to centroid) ===
def find_normals(embeddings, labels, centroid, target_class, k):
    distances = []
    for i, emb in enumerate(embeddings):
        if labels[i] == target_class:
            dist = euclidean(emb, centroid)
            distances.append((i, dist))
    distances.sort(key=lambda x: x[1])  # Closest first
    return [idx for idx, dist in distances[:k]]

# === Compute graph structure features ===
def compute_graph_features(graph):
    G = to_networkx(graph, to_undirected=True)
    return {
        'nodes': G.number_of_nodes(),
        'edges': G.number_of_edges(),
        'density': nx.density(G)
    }

# === Plot graph ===
def plot_graph(graph, title, filename):
    G = to_networkx(graph, to_undirected=True)
    plt.figure(figsize=(6, 5))
    pos = nx.spring_layout(G, seed=42)
    nx.draw(G, pos, node_color='skyblue', edge_color='gray',
            with_labels=False, node_size=300, font_size=8)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Saved: {filename}")

# === Summarize feature statistics ===
def summarize_features(feature_list, group_name):
    nodes = np.array([feat['nodes'] for feat in feature_list])
    edges = np.array([feat['edges'] for feat in feature_list])
    densities = np.array([feat['density'] for feat in feature_list])

    print(f"\n===== {group_name} =====")
    print(f"Avg nodes   : {np.mean(nodes):.2f}")
    print(f"Avg edges   : {np.mean(edges):.2f}")
    print(f"Avg density : {np.mean(densities):.4f}")

# === Main execution ===
def main():
    embeddings, labels = load_embeddings_and_labels()
    dataset = load_dataset()

    # Compute centroids
    centroid_real = compute_centroid(embeddings, labels, target_class=0)
    centroid_fake = compute_centroid(embeddings, labels, target_class=1)

    # Find outliers and normals
    real_outliers = find_outliers(embeddings, labels, centroid_real, target_class=0, k=num_outliers)
    fake_outliers = find_outliers(embeddings, labels, centroid_fake, target_class=1, k=num_outliers)

    real_normals = find_normals(embeddings, labels, centroid_real, target_class=0, k=num_graphs_to_plot)
    fake_normals = find_normals(embeddings, labels, centroid_fake, target_class=1, k=num_graphs_to_plot)

    # === Extract features ===

    # For outliers
    real_outlier_features = [compute_graph_features(dataset[i]) for i in real_outliers]
    fake_outlier_features = [compute_graph_features(dataset[i]) for i in fake_outliers]

    # For normals: all other graphs of same class
    real_normal_features = []
    fake_normal_features = []

    for i in range(len(dataset)):
        if labels[i] == 0 and i not in real_outliers:
            real_normal_features.append(compute_graph_features(dataset[i]))
        elif labels[i] == 1 and i not in fake_outliers:
            fake_normal_features.append(compute_graph_features(dataset[i]))

    # Summarize features
    summarize_features(real_outlier_features, "Real News Outliers")
    summarize_features(real_normal_features, "Real News Normals")

    summarize_features(fake_outlier_features, "Fake News Outliers")
    summarize_features(fake_normal_features, "Fake News Normals")

    # Plot graphs
    print("\n=== Plotting Real News Outlier Graphs ===")
    for idx in real_outliers[:num_graphs_to_plot]:
        title = f"Real News Outlier {idx}"
        filename = os.path.join(save_dir, f"real_outlier_{idx}.png")
        plot_graph(dataset[idx], title, filename)

    print("\n=== Plotting Real News Normal Graphs ===")
    for idx in real_normals:
        title = f"Real News Normal {idx}"
        filename = os.path.join(save_dir, f"real_normal_{idx}.png")
        plot_graph(dataset[idx], title, filename)

    print("\n=== Plotting Fake News Outlier Graphs ===")
    for idx in fake_outliers[:num_graphs_to_plot]:
        title = f"Fake News Outlier {idx}"
        filename = os.path.join(save_dir, f"fake_outlier_{idx}.png")
        plot_graph(dataset[idx], title, filename)

    print("\n=== Plotting Fake News Normal Graphs ===")
    for idx in fake_normals:
        title = f"Fake News Normal {idx}"
        filename = os.path.join(save_dir, f"fake_normal_{idx}.png")
        plot_graph(dataset[idx], title, filename)

if __name__ == "__main__":
    main()
