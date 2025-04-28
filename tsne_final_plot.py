import torch
import os
import argparse
import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
from matplotlib.patches import Patch

def load_embeddings(embedding_dir):
    embeddings = {}
    labels = torch.load(os.path.join(embedding_dir, "test_labels.pt"))

    for model in ["gat", "rgcn", "han"]:
        path = os.path.join(embedding_dir, f"{model}_embeddings.pt")
        if os.path.exists(path):
            embeddings[model.upper()] = torch.load(path)
        else:
            print(f"Warning: {model}_embeddings.pt not found!")

    return embeddings, labels

def plot_tsne(X, y, title, save_path=None):
    plt.figure(figsize=(10, 8))
    
    # Class names and colors
    class_names = {0: "Real News", 1: "Fake News"}
    colors = ['blue' if label == 0 else 'red' for label in y]

    scatter = plt.scatter(X[:, 0], X[:, 1], c=colors, alpha=0.7)
    
    legend_elements = [
        Patch(facecolor='blue', label='Real News'),
        Patch(facecolor='red', label='Fake News')
    ]
    plt.legend(handles=legend_elements, title="News Type")

    plt.title(title)
    plt.xlabel("t-SNE Component 1 (latent dimension)")
    plt.ylabel("t-SNE Component 2 (latent dimension)")
    plt.grid(True)

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

def run_tsne_all(embeddings, labels, output_dir):
    y = labels.numpy()

    # Individual model embeddings
    for model_name, emb in embeddings.items():
        X = emb.numpy()

        tsne = TSNE(n_components=2, perplexity=30, random_state=42)
        X_reduced = tsne.fit_transform(X)

        plot_tsne(X_reduced, y, title=f"{model_name} Embeddings (t-SNE)",
                  save_path=os.path.join(output_dir, f"{model_name.lower()}_tsne.png"))

    # Combined GAT + RGCN + HAN
    combined_all = torch.cat([embeddings['GAT'], embeddings['RGCN'], embeddings['HAN']], dim=1)
    X_all = combined_all.numpy()

    tsne_all = TSNE(n_components=2, perplexity=30, random_state=42)
    X_all_reduced = tsne_all.fit_transform(X_all)

    plot_tsne(X_all_reduced, y, title="Combined (GAT + RGCN + HAN) Embeddings (t-SNE)",
              save_path=os.path.join(output_dir, "combined_with_gat_tsne.png"))

    # Combined RGCN + HAN only (no GAT)
    combined_without_gat = torch.cat([embeddings['RGCN'], embeddings['HAN']], dim=1)
    X_without_gat = combined_without_gat.numpy()

    tsne_without_gat = TSNE(n_components=2, perplexity=30, random_state=42)
    X_without_gat_reduced = tsne_without_gat.fit_transform(X_without_gat)

    plot_tsne(X_without_gat_reduced, y, title="Combined (RGCN + HAN Only) Embeddings (t-SNE)",
              save_path=os.path.join(output_dir, "combined_without_gat_tsne.png"))

def main():
    parser = argparse.ArgumentParser(description="Run t-SNE for GAT, RGCN, HAN embeddings and combined versions")
    parser.add_argument("--embedding_dir", type=str, required=True, help="Folder with *_embeddings.pt and test_labels.pt")
    parser.add_argument("--output_dir", type=str, default="./tsne_outputs", help="Where to save t-SNE plots")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    embeddings, labels = load_embeddings(args.embedding_dir)
    run_tsne_all(embeddings, labels, args.output_dir)

if __name__ == "__main__":
    main()
