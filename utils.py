import torch
from torch_geometric.data import HeteroData, Data
from typing import List, Union

def convert_single_graph(homogeneous_graph: Data, source_node_idx: int = 0, add_source_self_loop: bool = False) -> HeteroData:
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

def convert_to_heterogeneous(homogeneous_dataset, source_node_idx=0, add_source_self_loop=False):
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