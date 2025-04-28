import torch
import os
from torch_geometric.datasets import UPFD
from torch_geometric.loader import DataLoader
from modeling import EnsembleGraphClassifier

# === Paths ===
checkpoint_path = "./models/Ensemble_politifact_study_20250427_233703"
save_embeddings_dir = "./saved_embeddings"
os.makedirs(save_embeddings_dir, exist_ok=True)

# === Set Device ===
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

# === Load Model ===
model, config = EnsembleGraphClassifier.load_from_checkpoint(checkpoint_path)
model.eval()
model = model.to(device)

# === Load Test Dataset ===
test_dataset = UPFD(root="./data/UPFD", name="politifact", split="test", feature="bert")
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)

all_labels = []
# === Prepare Storage ===
all_embeddings = {'GAT': [], 'RGCN': [], 'HAN': []}
key_mapping = {
    'GATForGraphClassification': 'GAT',
    'RGCNForGraphClassification': 'RGCN',
    'HANForGraphClassification': 'HAN'
}

# === Extract Embeddings ===
with torch.no_grad():
    for batch in test_loader:
        batch = batch.to(device)
        embeds = model.get_embedding(batch)  # Dict: {model_class_name: embedding}
        
        for key, value in embeds.items():
            # Normalize key
            short_key = key_mapping.get(key, None)
            if short_key is None:
                print(f"Warning: unknown key {key} found!")
                continue
            
            all_embeddings[short_key].append(value.cpu())

        # Labels
        if hasattr(batch, 'y'):
            all_labels.append(batch.y.cpu())
        else:
            all_labels.append(batch['source'].y.cpu())

# === Save Embeddings ===
for key, tensors in all_embeddings.items():
    final_embed = torch.cat(tensors, dim=0)
    save_path = os.path.join(save_embeddings_dir, f"{key.lower()}_embeddings.pt")
    torch.save(final_embed, save_path)
    print(f"Saved {key} embeddings: {save_path}")

# === Save Labels ===
final_labels = torch.cat(all_labels, dim=0)
label_path = os.path.join(save_embeddings_dir, "test_labels.pt")
torch.save(final_labels, label_path)
print(f"Saved test labels: {label_path}")
