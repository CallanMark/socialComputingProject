import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from typing import Callable, Dict, List, Optional, Tuple, Union, Final

from torch_geometric.nn import GATConv, GATv2Conv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor

class GATForGraphClassification(BasicGNN):
    def __init__(
      self,
      in_channels: int,
      hidden_channels: int,
      out_channels: int,
      num_layers: int,
      heads: int,
      dropout: float = 0.0,
      pooling: str = 'mean',
      **kwargs,
    ):
      self.out_channels_final = out_channels
      
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
      
      self.classifier = Linear(self.out_channels, self.out_channels_final)
      
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