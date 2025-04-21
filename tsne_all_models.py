import torch
import argparse
import os
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
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

def plot_tsne_2d(X, y, title, save_path=None):
    plt.figure(figsize=(10, 8))
    
    # Assign class names
    class_names = {0: "Real News", 1: "Fake News"}
    colors = ['blue', 'red']
    y_named = [class_names[label] for label in y]

    scatter = plt.scatter(X[:, 0], X[:, 1], c=y, cmap="coolwarm", alpha=0.7)
    
    # Legend using class names
    legend_elements = [Patch(facecolor=colors[i], label=class_names[i]) for i in range(2)]
    plt.legend(handles=legend_elements, title="News Type")

    plt.title(f"{title} — 2D t-SNE Embedding")
    plt.xlabel("t-SNE Component 1\n(Latent feature dimension learned from embeddings)")
    plt.ylabel("t-SNE Component 2\n(Latent feature dimension learned from embeddings)")
    plt.grid(True)
    
    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

def plot_tsne_3d(X, y, title, save_path=None):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection="3d")

    class_names = {0: "Real News", 1: "Fake News"}
    colors = ['blue', 'red']
    y_named = [class_names[label] for label in y]

    scatter = ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=y, cmap="coolwarm", alpha=0.7)
    ax.set_title(f"{title} — 3D t-SNE Embedding")
    ax.set_xlabel("t-SNE Component 1\n(Latent axis)")
    ax.set_ylabel("t-SNE Component 2\n(Latent axis)")
    ax.set_zlabel("t-SNE Component 3\n(Latent axis)")

    legend_elements = [Patch(facecolor=colors[i], label=class_names[i]) for i in range(2)]
    ax.legend(handles=legend_elements, title="News Type")

    if save_path:
        plt.savefig(save_path)
        print(f"Saved: {save_path}")
    else:
        plt.show()

def run_tsne(embeddings, labels, output_dir):
    y = labels.numpy()

    for model_name, emb in embeddings.items():
        X = emb.numpy()

        # 2D t-SNE
        tsne_2d = TSNE(n_components=2, perplexity=30, random_state=42)
        X_2d = tsne_2d.fit_transform(X)
        plot_tsne_2d(X_2d, y, title=model_name,
                     save_path=os.path.join(output_dir, f"{model_name}_tsne_2d.png"))

        # 3D t-SNE
        tsne_3d = TSNE(n_components=3, perplexity=30, random_state=42)
        X_3d = tsne_3d.fit_transform(X)
        plot_tsne_3d(X_3d, y, title=model_name,
                     save_path=os.path.join(output_dir, f"{model_name}_tsne_3d.png"))

def main():
    parser = argparse.ArgumentParser(description="Run t-SNE for all model embeddings with clear labels")
    parser.add_argument("--embedding_dir", type=str, required=True,
                        help="Directory with *_embeddings.pt and test_labels.pt")
    parser.add_argument("--output_dir", type=str, default="./tsne_outputs",
                        help="Where to save the plots")

    args = parser.parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    embeddings, labels = load_embeddings(args.embedding_dir)
    run_tsne(embeddings, labels, args.output_dir)

if __name__ == "__main__":
    main()
