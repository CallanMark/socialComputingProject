import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from typing import Callable, Dict, List, Optional, Tuple, Union, Final

from torch_geometric.nn import GATConv, GATv2Conv, HANConv, RGCNConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor


class GATForGraphClassification(BasicGNN):
    def __init__(
      self,
      in_channels: int,
      hidden_channels: int,
      num_classes: int,
      num_layers: int,
      dropout: float = 0.0,
      pooling: str = 'mean',
      **kwargs,
    ):
      self.num_classes = num_classes
      
      super().__init__(
        in_channels=in_channels,
        out_channels=None,
        hidden_channels=hidden_channels,
        num_layers=num_layers,
        dropout=dropout,
        **kwargs,
      )
      
      self.pooling = pooling
      
      if pooling == 'add':
        self.pool = global_add_pool
      elif pooling == 'mean':
        self.pool = global_mean_pool
      elif pooling == 'max':
        self.pool = global_max_pool
      else:
        raise ValueError(f"Pooling type {pooling} not supported.")
      
      self.classifier = Linear(self.out_channels, self.num_classes)
      
    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:

        v2 = kwargs.pop('v2', False)
        heads = kwargs.pop('heads', 1)
        concat = kwargs.pop('concat', True)

        # Do not use concatenation in case the layer `GATConv` layer maps to
        # the desired output channels (out_channels != None and jk != None):
        if getattr(self, '_is_conv_to_out', False):
            concat = False

        if concat and out_channels % heads != 0:
            raise ValueError(f"Ensure that the number of output channels of "
                             f"'GATConv' (got '{out_channels}') is divisible "
                             f"by the number of heads (got '{heads}')")

        if concat:
            out_channels = out_channels // heads

        Conv = GATConv if not v2 else GATv2Conv
        return Conv(in_channels, out_channels, heads=heads, concat=concat,
                    dropout=self.dropout.p, **kwargs)
      
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        """
        Forward pass for graph classification.
        
        Args:
            x: Node features [num_nodes, in_channels]
            edge_index: Graph connectivity [2, num_edges]
            batch: Batch vector [num_nodes] mapping each node to its graph
            edge_attr: Edge features [num_edges, edge_dim] (optional)
        
        Returns:
            Graph classification predictions [batch_size, out_channels_final]
        """
        # Get node embeddings using the GNN layers from the parent class
        x = self.convs[0](x, edge_index, edge_attr=edge_attr)
        x = self.act(x)
        
        for i, conv in enumerate(self.convs[1:]):
            x = self.dropout(x)
            x = conv(x, edge_index, edge_attr=edge_attr)
            if i < len(self.convs) - 2:
                x = self.act(x)
        
            
        # Pool node features to graph-level representation
        if batch is None:
            # If no batch is provided, assume a single graph
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Apply pooling to get graph-level representation
        x = self.pool(x, batch)
        
        # Apply final classification layer
        x = self.dropout(x)
        x = self.classifier(x)
        
        return x

class HAN(torch.nn.Module):
    def __init__(self, in_channels: Union[int, Dict[str, int]],
                 out_channels: int, hidden_channels=128, heads=8, metadata=None, dropout=0.6):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                                dropout=dropout, metadata=metadata)
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        x = self.han_conv(x_dict, edge_index_dict)
        x = self.lin(x)
        return x

class HANForGraphClassification(torch.nn.Module):
    def __init__(self, in_channels, out_channels, hidden_channels=128, heads=8, metadata=None,
                 dropout=0.6, num_classes=2):
        super().__init__()
        self.han_conv = HANConv(in_channels, hidden_channels, heads=heads,
                               dropout=dropout, metadata=metadata)
        
        # Linear layer will be initialized during forward pass once we know the input dimension
        self.lin = None
        self.out_channels = out_channels
        self.classifier = torch.nn.Linear(out_channels, num_classes)
    
    def forward(self, x_dict, edge_index_dict, batch=None):
        """
        Forward pass for heterogeneous graph classification.
        
        Args:
            x_dict: Dictionary of node features for each node type
            edge_index_dict: Dictionary of edge indices for each edge type
            batch: Batch vector (optional, only needed for batched inference)
        
        Returns:
            Graph classification predictions
        """
        # Get node embeddings from HANConv
        node_embeddings_dict = self.han_conv(x_dict, edge_index_dict)
        
        # Average pooling for each node type
        pooled_embeddings = []
        
        for node_type, embeddings in node_embeddings_dict.items():
            if embeddings is not None:
                # Average pooling for nodes of the same type
                pooled = torch.mean(embeddings, dim=0)
                pooled_embeddings.append(pooled)
        
        if not pooled_embeddings:
            raise ValueError("No node embeddings were produced by the model")
        
        # Concatenate all pooled embeddings from different node types
        x = torch.cat(pooled_embeddings)
        
        # Initialize the linear layer if not done yet
        if self.lin is None:
            lin_input_dim = x.size(0)
            self.lin = torch.nn.Linear(lin_input_dim, self.out_channels).to(x.device)
        
        # Apply linear layer and classifier
        x = self.lin(x)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        
        return x
    

class RGCNForGraphClassification(torch.nn.Module):
    """
    Graph classification model based on Relational Graph Convolutional Networks (RGCN).
    
    This model uses RGCN layers to process graphs with multiple relation types,
    followed by global pooling to create graph-level representations for classification.
    
    Args:
        in_channels (int): Number of input features for each node.
        hidden_channels (int): Number of hidden features.
        out_channels (int): Number of output classes.
        num_relations (int): Number of different relation/edge types.
        num_bases (int, optional): If set, this layer will use the basis-decomposition
            regularization scheme where num_bases denotes the number of bases to use.
            This helps reduce the number of parameters. Defaults to None.
    
    Example:
        >>> model = RGCNForGraphClassification(
        ...     in_channels=768,
        ...     hidden_channels=128,
        ...     out_channels=2,
        ...     num_relations=2,
        ...     num_bases=4
        ... )
    """
    def __init__(self, in_channels, hidden_channels, out_channels, num_relations, num_bases=None):
        super(RGCNForGraphClassification, self).__init__()
        
        # First RGCN convolution layer
        self.conv1 = RGCNConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            num_relations=num_relations,
            num_bases=num_bases
        )
        
        # Second RGCN convolution layer
        self.conv2 = RGCNConv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            num_relations=num_relations,
            num_bases=num_bases
        )
        
        # Output layer
        self.classifier = Linear(hidden_channels, out_channels)
    
    def forward(self, x, edge_index, edge_type, batch):
        """
        Forward pass for graph classification.
        
        Args:
            x (torch.Tensor): Node feature matrix [num_nodes, in_channels].
            edge_index (torch.Tensor): Edge indices [2, num_edges].
            edge_type (torch.Tensor): Edge type/relation indices [num_edges].
            batch (torch.Tensor): Batch vector [num_nodes] mapping each node to its graph.
            
        Returns:
            torch.Tensor: Graph classification predictions [batch_size, out_channels].
        """
        # First layer with ReLU activation
        x = self.conv1(x, edge_index, edge_type)
        x = F.relu(x)
        x = F.dropout(x, p=0.5, training=self.training)
        
        # Second layer
        x = self.conv2(x, edge_index, edge_type)
        x = F.relu(x)
        
        # Global pooling (from node-level to graph-level representation)
        x = global_mean_pool(x, batch)
        
        # Apply final classifier
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.classifier(x)
        
        return x