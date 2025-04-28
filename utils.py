import os
import json

import torch
from torch_geometric.data import HeteroData, Data, Batch
from typing import List, Union


from modeling import (
    EnsembleGraphClassifier, 
    GATForGraphClassification, 
    HANForGraphClassification, 
    RGCNForGraphClassification
)

def convert_single_graph(homogeneous_graph: Data, source_node_idx: int = 0, add_source_self_loop: bool = True) -> HeteroData:
    """
    Convert a single homogeneous graph to a heterogeneous graph with two node types:
    - 'source': News source node
    - 'user': All other nodes
    
    And two edge types:
    - ('source', 'to', 'user'): Edges from source to users
    - ('user', 'to', 'user'): Edges between users
    - ('source', 'to', 'source'): Self-loop for source node (optional)
    
    Args:
        homogeneous_graph: A PyTorch Geometric Data object
        source_node_idx: Index of the source node in the graph, default is 0
        add_source_self_loop: Whether to add a self-loop to the source node, default is False
        
    Returns:
        A HeteroData object
    """
    hetero_graph = HeteroData()
    
    # Get total number of nodes
    num_nodes = homogeneous_graph.num_nodes
    
    # Extract features for source node
    source_features = homogeneous_graph.x[source_node_idx:source_node_idx+1]
    
    # Extract features for user nodes (all nodes except source)
    user_indices = torch.cat([
        torch.arange(0, source_node_idx), 
        torch.arange(source_node_idx + 1, num_nodes)
    ])
    user_features = homogeneous_graph.x[user_indices]
    
    # Add node features to the heterogeneous graph
    hetero_graph['source'].x = source_features
    hetero_graph['user'].x = user_features
    
    # Create a mapping from original node indices to new node indices
    node_mapping = {}
    node_mapping[source_node_idx] = ('source', 0)  # Source node maps to index 0 in 'source' type
    
    # Map all other nodes to 'user' type
    user_counter = 0
    for i in range(num_nodes):
        if i != source_node_idx:
            node_mapping[i] = ('user', user_counter)
            user_counter += 1
    
    # Process edges
    edge_index = homogeneous_graph.edge_index
    
    # Source-to-user edges and User-to-user edges
    source_to_user_edges = []
    user_to_user_edges = []
    
    for i in range(edge_index.shape[1]):
        src, dst = edge_index[0, i].item(), edge_index[1, i].item()
        
        src_type, src_idx = node_mapping[src]
        dst_type, dst_idx = node_mapping[dst]
        
        if src_type == 'source' and dst_type == 'user':
            # Source to user edge
            source_to_user_edges.append((src_idx, dst_idx))
        elif src_type == 'user' and dst_type == 'user':
            # User to user edge
            user_to_user_edges.append((src_idx, dst_idx))
        # We ignore user-to-source edges as mentioned in the requirements
    
    # Add edges to the heterogeneous graph
    if source_to_user_edges:
        src_indices, dst_indices = zip(*source_to_user_edges)
        hetero_graph['source', 'to', 'user'].edge_index = torch.tensor(
            [src_indices, dst_indices], dtype=torch.long
        )
        
    
    if user_to_user_edges:
        src_indices, dst_indices = zip(*user_to_user_edges)
        hetero_graph['user', 'to', 'user'].edge_index = torch.tensor(
            [src_indices, dst_indices], dtype=torch.long
        )
    
    # Add self-loop to source node if requested
    if add_source_self_loop:
        hetero_graph['source', 'to', 'source'].edge_index = torch.tensor(
            [[0], [0]], dtype=torch.long
        )
    
    # Copy graph-level targets if they exist
    if hasattr(homogeneous_graph, 'y'):
        hetero_graph['source'].y = homogeneous_graph.y
    
    return hetero_graph

def convert_to_heterogeneous(homogeneous_dataset, source_node_idx=0, add_source_self_loop=True):
    """
    Convert a homogeneous UPFD dataset to a heterogeneous dataset.
    
    Args:
        homogeneous_dataset: A PyTorch Geometric UPFD dataset
        source_node_idx: Index of the source node in each graph, default is 0
        add_source_self_loop: Whether to add a self-loop to the source node, default is False
        
    Returns:
        A list of HeteroData objects
    """
    # Simply apply convert_single_graph to each graph in the dataset
    hetero_dataset = [
        convert_single_graph(graph, source_node_idx, add_source_self_loop) 
        for graph in homogeneous_dataset
    ]
    
    return hetero_dataset

def get_edge_type(edge_index, source_indices=[0]):
    """
    Generate edge type tensor based on source node indices.
    
    This function creates a tensor of edge types by assigning different types to edges
    based on whether the source node is in the specified list of source indices.
    
    Args:
        edge_index (torch.Tensor): The edge index tensor of shape [2, num_edges]
            where edge_index[0] contains source nodes and edge_index[1] contains 
            target nodes.
        source_indices (list, optional): List of node indices to be considered as 
            source nodes. Edges originating from these nodes will be assigned type 0,
            while all other edges will be assigned type 1. Defaults to [0].
    
    Returns:
        torch.Tensor: A tensor of shape [num_edges] containing the edge types.
            Edges from nodes in source_indices have type 0, others have type 1.
    
    Example:
        >>> edge_index = torch.tensor([[0, 1, 2, 0], [1, 2, 3, 3]])
        >>> edge_type = get_edge_type(edge_index, source_indices=[0])
        >>> print(edge_type)
        tensor([0, 1, 1, 0])
    """
    edge_type = []
    for src, tgt in edge_index.t().tolist():
        if src in source_indices:
            edge_type.append(0)
        else:
            edge_type.append(1)
    return torch.tensor(edge_type)

def to_hetero_batch(batch, add_source_self_loop=True):
    device = batch.x.device
    data_list = batch.to_data_list()
    data_list = convert_to_heterogeneous(data_list, 0, add_source_self_loop)
    batch = Batch.from_data_list(data_list)
    batch.batch = {
        'source': batch['source'].batch,
        'user': batch['user'].batch
    }
    batch.to(device)
    return batch


def load_model_from_folder(folder_path, device=None):
    """
    Load a trained model from a folder containing best_model.pt and final_results.json.
    
    Args:
        folder_path (str): Path to the folder containing the model files
        device (str, optional): Device to load the model on ('cpu', 'cuda', etc.). 
                              If None, automatically determines the device.
    
    Returns:
        model: The loaded model ready for inference
        config: Dictionary containing model configuration and training results
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(device)
    
    print(f"Using device: {device}")
    
    # Define paths for model file and results file
    model_path = os.path.join(folder_path, "best_model.pt")
    results_file = os.path.join(folder_path, "final_results.json")
    
    # Check if files exist
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if not os.path.exists(results_file):
        raise FileNotFoundError(f"Results file not found: {results_file}")
    
    # Load model configuration and results
    with open(results_file, 'r') as f:
        config = json.load(f)
    
    # Extract parameters for model creation
    params = config["best_params"]
    
    # Get model hyperparameters
    hidden_dim = params["hidden_dim"]
    num_layers = params["num_layers"]
    dropout = params["dropout"]
    heads = params["heads"]
    pooling = params["pooling"]
    ensemble_method = params["ensemble_method"]
    
    # Assume BERT feature dimension of 768
    in_channels = 768
    
    # Load the checkpoint first to examine its structure
    checkpoint = torch.load(model_path, map_location=device)
    
    # Print the keys in the state dict to debug
    if 'model_state_dict' in checkpoint:
        state_dict = checkpoint['model_state_dict']
    else:
        state_dict = checkpoint
    
    print("State dict contains keys like:", list(state_dict.keys())[:5], "... and more")
    
    # Create the ensemble model with the appropriate submodels
    print(f"Creating model with parameters: {params}")
    model = EnsembleGraphClassifier(
        models=[
            GATForGraphClassification(
                in_channels=in_channels,
                hidden_channels=hidden_dim,
                num_classes=2,
                num_layers=num_layers,
                dropout=dropout,
                pooling=pooling,
                heads=heads,
                v2=False,
                concat=True
            ),
            HANForGraphClassification(
                in_channels=in_channels,
                hidden_channels=hidden_dim,
                out_channels=hidden_dim,
                num_classes=2,
                heads=heads,
                metadata=(['source', 'user'], [('source', 'to', 'user'), ('user', 'to', 'user'), ('source', 'to', 'source')]),
                dropout=dropout,
                num_layers=num_layers,
                lin_input_dim=hidden_dim,
            ),
            RGCNForGraphClassification(
                in_channels=in_channels,
                hidden_channels=hidden_dim,
                num_classes=2,
                num_relations=2,
                num_bases=None,
                num_layers=num_layers,
                dropout=dropout,
                pooling=pooling
            )
        ],
        ensemble_method=ensemble_method,
        num_classes=2,
        hidden_dim=hidden_dim,
        dropout=dropout
    )
    
    # Manually check for missing and unexpected keys
    model_keys = set(model.state_dict().keys())
    state_dict_keys = set(state_dict.keys())
    
    missing_keys = model_keys - state_dict_keys
    unexpected_keys = state_dict_keys - model_keys
    
    if missing_keys:
        print(f"Missing keys in state dict: {missing_keys}")
    
    if unexpected_keys:
        print(f"Unexpected keys in state dict: {unexpected_keys}")
    
    # Try loading with strict=False to ignore missing or unexpected keys
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    else:
        model.load_state_dict(checkpoint, strict=False)
    
    print("Model loaded with strict=False (ignoring missing/unexpected keys)")
    
    # Move model to the appropriate device and set to evaluation mode
    model = model.to(device)
    model.eval()
    
    print(f"Model loaded with test accuracy: {config.get('test_accuracy', 'N/A')}")
    return model, config