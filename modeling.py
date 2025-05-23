import torch
import torch.nn.functional as F
from torch.nn import Linear, ModuleList, ReLU, Sequential
from typing import Callable, Dict, List, Optional, Tuple, Union, Final

from torch_geometric.nn import GATConv, GATv2Conv, HANConv, RGCNConv, global_add_pool, global_mean_pool, global_max_pool
from torch_geometric.nn.models.basic_gnn import BasicGNN
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.data import HeteroData, Batch

from utils import get_edge_type, convert_to_heterogeneous, to_hetero_batch
#######################
# GAT (Graph Attention Network) Models
#######################

class GAT(BasicGNN):
    """
    Base GAT model that outputs node embeddings.
    """
    supports_edge_weight: Final[bool] = False
    supports_edge_attr: Final[bool] = True
    
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        out_channels: int,
        num_layers: int,
        dropout: float = 0.0,
        v2: bool = False,
        heads: int = 8,
        concat: bool = True,
        **kwargs,
    ):
        # Store these attributes before the parent class constructor
        self.v2 = v2
        self.heads = heads
        self.concat = concat
        self.out_dim = out_channels
        
        super().__init__(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            num_layers=num_layers,
            dropout=dropout,
            **kwargs,
        )
      
    def init_conv(self, in_channels: Union[int, Tuple[int, int]],
                  out_channels: int, **kwargs) -> MessagePassing:
        
        # Use the stored attributes
        v2 = kwargs.pop('v2', self.v2)
        heads = kwargs.pop('heads', self.heads)
        concat = kwargs.pop('concat', self.concat)

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


class GATForGraphClassification(torch.nn.Module):
    """
    Graph classification model based on Graph Attention Networks.
    """
    def __init__(
        self,
        in_channels: int,
        hidden_channels: int,
        num_classes: int,
        num_layers: int,
        dropout: float = 0.0,
        pooling: str = 'mean',
        v2: bool = False,
        heads: int = 8,
        concat: bool = True,
        **kwargs,
    ):
        super().__init__()
        
        # Create the base GAT model for node embeddings
        self.gat = GAT(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,
            num_layers=num_layers,
            dropout=dropout,
            v2=v2,
            heads=heads,
            concat=concat,
            **kwargs,
        )
        
        # Set up the pooling function
        if pooling == 'add':
            self.pool = global_add_pool
        elif pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Pooling type {pooling} not supported.")
        
        # Classification layer
        self.classifier = Linear(hidden_channels, num_classes)
        self.dropout = dropout
        
        # Store the embedding dimension for ensemble methods
        self.output_dim = hidden_channels
      
    def forward(self, x, edge_index, batch=None, edge_attr=None):
        """
        Forward pass for graph classification.
        """
        # Get graph embeddings
        embeddings = self.get_embedding(x, edge_index, batch, edge_attr)
        
        # Apply final classification layer
        x = F.dropout(embeddings, p=self.dropout, training=self.training)
        x = self.classifier(x)
        
        return x
    
    def get_embedding(self, x, edge_index, batch=None, edge_attr=None):
        """
        Get graph-level embeddings for use in classification or ensemble methods.
        """
        # Get node embeddings from the base GAT model
        x = self.gat(x, edge_index, edge_attr=edge_attr)
        
        # Pool node features to graph-level representation
        if batch is None:
            # If no batch is provided, assume a single graph
            batch = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        
        # Apply pooling to get graph-level representation
        x = self.pool(x, batch)
        
        return x


#######################
# HAN (Heterogeneous Graph Attention Network) Models
#######################

class HAN(torch.nn.Module):
    """
    Base Heterogeneous Graph Attention Network (HAN) model that outputs node embeddings.
    """
    def __init__(self, 
                 in_channels: Union[int, Dict[str, int]],
                 hidden_channels: int,
                 out_channels: int, 
                 heads: int = 8, 
                 metadata: Optional[Tuple] = None, 
                 dropout: float = 0.6,
                 num_layers: int = 1):
        super().__init__()
        self.num_layers = num_layers
        self.out_dim = out_channels
        
        # HANConv does not support multiple layers natively
        self.han_conv = HANConv(
            in_channels=in_channels, 
            out_channels=hidden_channels, 
            heads=heads,
            dropout=dropout, 
            metadata=metadata
        )
        
        # For multi-layer networks, we create additional transformation layers
        self.transforms = ModuleList()
        if num_layers > 1:
            # First transformation after HANConv
            self.transforms.append(torch.nn.Linear(hidden_channels, hidden_channels))
            
            # Additional transformation layers if requested
            for _ in range(num_layers - 2):
                self.transforms.append(torch.nn.Linear(hidden_channels, hidden_channels))
        
        # Final transformation to output dimension
        self.lin = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_dict, edge_index_dict):
        """
        Forward pass that returns node embeddings for each node type.
        """
        # Get node embeddings from HANConv
        x = self.han_conv(x_dict, edge_index_dict)
        
        # Apply additional transformation layers if specified
        if self.num_layers > 1:
            for i, transform in enumerate(self.transforms):
                # Apply transformation to each node type's embeddings
                for node_type in x.keys():
                    if x[node_type] is not None:
                        x[node_type] = transform(x[node_type])
                        x[node_type] = F.relu(x[node_type])
                        x[node_type] = F.dropout(x[node_type], p=0.5, training=self.training)
        
        # Apply final linear transformation to each node type's embeddings
        for node_type in x.keys():
            if x[node_type] is not None:
                x[node_type] = self.lin(x[node_type])
        
        return x


class HANForGraphClassification(torch.nn.Module):
    """
    Graph classification model based on Heterogeneous Graph Attention Networks.
    """
    def __init__(self, 
                 in_channels: Union[int, Dict[str, int]],
                 hidden_channels: int,
                 out_channels: int, 
                 num_classes: int = 2,
                 heads: int = 8, 
                 metadata: Optional[Tuple] = None, 
                 dropout: float = 0.6,
                 lin_input_dim: Optional[int] = None,
                 num_layers: int = 1,
                 ):
        super().__init__()
        
        # Create the base HAN model for node embeddings
        # The issue is here - you initialize 'self.han', but try to use 'self.han_conv' in forward
        # Either rename this to self.han_conv or fix the forward method
        self.han_conv = HANConv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            heads=heads,
            dropout=dropout,
            metadata=metadata
        )
        
        # Linear layer for dimensionality reduction after pooling
        # Will be initialized during forward pass once we know the input dimension
        self.lin = None
        self.out_channels = out_channels
        
        # Classification layer
        self.classifier = torch.nn.Linear(out_channels, num_classes)
        self.dropout = dropout
        
        # Store the embedding dimension for ensemble methods
        self.output_dim = out_channels

        if lin_input_dim:
            self.lin = torch.nn.Linear(hidden_channels, out_channels)
    def forward(self, x_dict, edge_index_dict, batch=None):
        """
        Forward pass for heterogeneous graph classification.
        """
        # Get node embeddings from HANConv (this line needs to match your initialization)
        x = self.get_embedding(x_dict, edge_index_dict, batch)
        
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.classifier(x)
        
        return x
    
    def get_embedding(self, x_dict, edge_index_dict, batch=None):
        """
        Get graph-level embeddings for use in classification or ensemble methods.
        """
        # Get node embeddings from the base HAN model
        node_embeddings_dict = self.han_conv(x_dict, edge_index_dict)
        
        # Average pooling for each node type
        pooled_embeddings = {}
        if batch is None:
            # If no batch is provided, assume a single graph
            batch = {node_type: torch.zeros(embeddings.size(0), dtype=torch.long, device=embeddings.device)
                     for node_type, embeddings in node_embeddings_dict.items() if embeddings is not None}
        for node_type, embeddings in node_embeddings_dict.items():
            if embeddings is not None:
                # Average pooling for nodes of the same type
                pooled = global_mean_pool(embeddings, batch[node_type])
                pooled_embeddings.setdefault(node_type, []).append(pooled)
        
        if not pooled_embeddings:
            raise ValueError("No node embeddings were produced by the model")
        
        embeddings_by_batch = []
        for source, user in zip(pooled_embeddings['source'], pooled_embeddings['user']):
            embeddings_by_batch += (source + user) / 2
        # Concatenate all pooled embeddings from different node types
        # for i,  embed in enumerate(embeddings_by_batch):
            # print(f"embed at {i}: {embed.shape}")
        x = torch.stack(embeddings_by_batch, dim=0)
        # Initialize the linear layer if not done yet
        if self.lin is None:
            lin_input_dim = x.size(1)
            self.lin = torch.nn.Linear(lin_input_dim, self.out_channels).to(x.device)
        
        # Apply linear layer
        x = self.lin(x)
        x = F.leaky_relu(x)
        
        return x



#######################
# RGCN (Relational Graph Convolutional Network) Models
#######################

class RGCN(torch.nn.Module):
    """
    Base Relational Graph Convolutional Network (RGCN) model that outputs node embeddings.
    """
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 out_channels: int,
                 num_relations: int, 
                 num_bases: Optional[int] = None,
                 num_layers: int = 2,
                 dropout: float = 0.5):
        super().__init__()
        
        self.num_layers = num_layers
        self.dropout = dropout
        self.out_dim = out_channels
        
        # Create RGCN layers
        self.convs = ModuleList()
        
        # First layer
        self.convs.append(
            RGCNConv(
                in_channels=in_channels,
                out_channels=hidden_channels,
                num_relations=num_relations,
                num_bases=num_bases
            )
        )
        
        # Middle layers (if any)
        for _ in range(num_layers - 2):
            self.convs.append(
                RGCNConv(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                    num_relations=num_relations,
                    num_bases=num_bases
                )
            )
        
        # Last layer
        if num_layers > 1:
            self.convs.append(
                RGCNConv(
                    in_channels=hidden_channels,
                    out_channels=out_channels,
                    num_relations=num_relations,
                    num_bases=num_bases
                )
            )
    
    def forward(self, x, edge_index, edge_type):
        """
        Forward pass that returns node embeddings.
        """
        # Apply RGCN layers
        for i, conv in enumerate(self.convs):
            x = conv(x, edge_index, edge_type)
            if i < len(self.convs) - 1:  # Apply activation to all but the last layer
                x = F.relu(x)
                x = F.dropout(x, p=self.dropout, training=self.training)
                
        return x


class RGCNForGraphClassification(torch.nn.Module):
    """
    Graph classification model based on Relational Graph Convolutional Networks.
    """
    def __init__(self, 
                 in_channels: int, 
                 hidden_channels: int, 
                 num_classes: int,
                 num_relations: int, 
                 num_bases: Optional[int] = None,
                 num_layers: int = 2,
                 dropout: float = 0.5,
                 pooling: str = 'mean'):
        super().__init__()
        
        # Create the base RGCN model for node embeddings
        self.rgcn = RGCN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=hidden_channels,  # Use same dimension for simplicity
            num_relations=num_relations,
            num_bases=num_bases,
            num_layers=num_layers,
            dropout=dropout
        )
        
        # Set up the pooling function
        if pooling == 'add':
            self.pool = global_add_pool
        elif pooling == 'mean':
            self.pool = global_mean_pool
        elif pooling == 'max':
            self.pool = global_max_pool
        else:
            raise ValueError(f"Pooling type {pooling} not supported.")
        
        # Classification layer
        self.classifier = Linear(hidden_channels, num_classes)
        
        # Dropout
        self.dropout = dropout
        
        # Store the embedding dimension for ensemble methods
        self.output_dim = hidden_channels
    
    def forward(self, x, edge_index, edge_type, batch):
        """
        Forward pass for graph classification.
        """
        # Get graph embeddings
        embeddings = self.get_embedding(x, edge_index, edge_type, batch)
        
        # Apply final classifier
        x = F.dropout(embeddings, p=self.dropout, training=self.training)
        x = self.classifier(x)
        
        return x
    
    def get_embedding(self, x, edge_index, edge_type, batch):
        """
        Get graph-level embeddings for use in classification or ensemble methods.
        """
        # Get node embeddings from the base RGCN model
        x = self.rgcn(x, edge_index, edge_type)
        
        # Global pooling (from node-level to graph-level representation)
        x = self.pool(x, batch)
        
        return x


#######################
# Ensemble Models
#######################

class EnsembleGraphClassifier(torch.nn.Module):
    """
    Ensemble model that combines multiple graph neural networks for graph classification.
    """
    def __init__(self, 
                 models: List[torch.nn.Module],
                 ensemble_method: str = 'voting',
                 num_classes: int = 2,
                 hidden_dim: int = 64,
                 dropout: float = 0.5):
        super().__init__()
        self.models = torch.nn.ModuleList(models)
        self.ensemble_method = ensemble_method
        self.num_classes = num_classes
        
        if ensemble_method == 'concat':
            # Calculate total embedding dimension from all models
            total_dim = sum(model.output_dim for model in models)
            self.classifier = torch.nn.Linear(total_dim, num_classes)
            
        elif ensemble_method == 'transform':
            # Fixed-size hidden layer regardless of number of models
            total_dim = sum(model.output_dim for model in models)
            self.transform = torch.nn.Linear(total_dim, hidden_dim)
            self.classifier = torch.nn.Linear(hidden_dim, num_classes)
            
        self.dropout = dropout
    
    def forward(self, data):
        """
        Forward pass for ensemble graph classification.
        """
        if self.ensemble_method == 'voting':
            # Get logits from each model
            all_logits = []
            for model in self.models:
                if hasattr(model, 'forward_data'):
                    # Use specialized data handling if available
                    logits = model.forward_data(data)
                else:
                    # Extract appropriate inputs based on model type
                    if isinstance(model, GATForGraphClassification):
                        logits = model(data.x, data.edge_index, data.batch, getattr(data, 'edge_attr', None))
                        # print(f"Logits shape from GAT model: {logits.shape}")
                    elif isinstance(model, HANForGraphClassification):
                        # print(f"data: {data}")
                        heter_data = to_hetero_batch(data)
                        # print(f"Heterogeneous data: {heter_data}")
                        logits = model(heter_data.x_dict, heter_data.edge_index_dict, batch=heter_data.batch)
                        # print(f"Logits shape from HAN model: {logits.shape}")
                    elif isinstance(model, RGCNForGraphClassification):
                        # Generate edge_type if not present
                        if not hasattr(data, 'edge_type'):
                            edge_type = get_edge_type(data.edge_index, 
                                    source_indices=data.ptr[:-1].tolist() if hasattr(data, 'ptr') else [0])
                        else:
                            edge_type = data.edge_type
                        logits = model(data.x, data.edge_index, edge_type, data.batch)
                        # print(f"Logits shape from RGCN model: {logits.shape}")
                    else:
                        raise TypeError(f"Unsupported model type: {type(model)}")
                
                all_logits.append(logits)
            
            # For training, we need to return logits, not predictions
            # Average the logits from all models
            if self.training:
                all_logits_stacked = torch.stack(all_logits, dim=0)
                return torch.mean(all_logits_stacked, dim=0)
            else:
                # For evaluation/inference, we can do voting on the predicted classes
                all_preds = [torch.argmax(logit, dim=1) for logit in all_logits]
                all_preds = torch.stack(all_preds, dim=0)
                # Get the most common prediction (mode) for each sample
                final_preds_values, _ = torch.mode(all_preds, dim=0)
                
                # Convert predictions back to one-hot format for consistency
                batch_size = final_preds_values.size(0)
                final_logits = torch.zeros(batch_size, self.num_classes, device=data.x.device)
                for i in range(batch_size):
                    final_logits[i, final_preds_values[i]] = 1.0
                
                return final_logits
            
        elif self.ensemble_method == 'average':
            # Average logits from all models
            all_logits = []
            for model in self.models:
                if hasattr(model, 'forward_data'):
                    # Use specialized data handling if available
                    logits = model.forward_data(data)
                else:
                    # Extract appropriate inputs based on model type
                    if isinstance(model, GATForGraphClassification):
                        logits = model(data.x, data.edge_index, data.batch, getattr(data, 'edge_attr', None))
                    elif isinstance(model, HANForGraphClassification):
                        heter_data = to_hetero_batch(data)
                        logits = model(heter_data.x_dict, heter_data.edge_index_dict, batch=heter_data.batch)
                    elif isinstance(model, RGCNForGraphClassification):
                        # Generate edge_type if not present
                        if not hasattr(data, 'edge_type'):
                            edge_type = get_edge_type(data.edge_index, 
                                    source_indices=data.ptr[:-1].tolist() if hasattr(data, 'ptr') else [0])
                        else:
                            edge_type = data.edge_type
                        logits = model(data.x, data.edge_index, edge_type, data.batch)
                    else:
                        raise TypeError(f"Unsupported model type: {type(model)}")
                
                all_logits.append(logits)
            
            # Stack and average logits
            all_logits = torch.stack(all_logits, dim=0)
            avg_logits = torch.mean(all_logits, dim=0)
            return avg_logits
            
        elif self.ensemble_method == 'concat' or self.ensemble_method == 'transform':
            # Get embeddings from each model
            all_embeddings = []
            for model in self.models:
                if hasattr(model, 'get_embedding_data'):
                    # Use specialized data handling if available
                    embed = model.get_embedding_data(data)
                else:
                    # Extract appropriate inputs based on model type
                    if isinstance(model, GATForGraphClassification):
                        embed = model.get_embedding(data.x, data.edge_index, data.batch, getattr(data, 'edge_attr', None))
                    elif isinstance(model, HANForGraphClassification):
                        heter_data = to_hetero_batch(data)
                        embed = model.get_embedding(heter_data.x_dict, heter_data.edge_index_dict, batch=heter_data.batch)
                    elif isinstance(model, RGCNForGraphClassification):
                        # Generate edge_type if not present
                        if not hasattr(data, 'edge_type'):
                            edge_type = get_edge_type(data.edge_index, 
                                    source_indices=data.ptr[:-1].tolist() if hasattr(data, 'ptr') else [0])
                        else:
                            edge_type = data.edge_type
                        embed = model.get_embedding(data.x, data.edge_index, edge_type, data.batch)
                    else:
                        raise TypeError(f"Unsupported model type: {type(model)}")
                
                all_embeddings.append(embed)
            
            # Concatenate embeddings
            combined = torch.cat(all_embeddings, dim=1)
            
            if self.ensemble_method == 'transform':
                combined = F.relu(self.transform(combined))
                combined = F.dropout(combined, p=self.dropout, training=self.training)
            
            # Apply classifier
            combined = F.dropout(combined, p=self.dropout, training=self.training)
            logits = self.classifier(combined)
            return logits
    
    def get_embedding(self, data):
        embed_dict = {}
        for model in self.models:
            if hasattr(model, 'get_embedding_data'):
                # Use specialized data handling if available
                embed = model.get_embedding_data(data)
            else:
                # Extract appropriate inputs based on model type
                if isinstance(model, GATForGraphClassification):
                    embed = model.get_embedding(data.x, data.edge_index, data.batch, getattr(data, 'edge_attr', None))
                elif isinstance(model, HANForGraphClassification):
                    heter_data = to_hetero_batch(data)
                    embed = model.get_embedding(heter_data.x_dict, heter_data.edge_index_dict, batch=heter_data.batch)
                elif isinstance(model, RGCNForGraphClassification):
                    # Generate edge_type if not present
                    if not hasattr(data, 'edge_type'):
                        edge_type = get_edge_type(data.edge_index, 
                                source_indices=data.ptr[:-1].tolist() if hasattr(data, 'ptr') else [0])
                    else:
                        edge_type = data.edge_type
                    embed = model.get_embedding(data.x, data.edge_index, edge_type, data.batch)
                else:
                    raise TypeError(f"Unsupported model type: {type(model)}")
            
            embed_dict[model.__class__.__name__] = embed
        return embed_dict
    
    @classmethod
    def load_from_checkpoint(cls, folder_path, device=None):
        """
        Class method to load a trained ensemble model from a checkpoint folder.
        
        Args:
            cls: The class itself (passed automatically)
            folder_path (str): Path to the folder containing the model files
            device (str, optional): Device to load the model on ('cpu', 'cuda', etc.)
                                Default is None (auto-detect)
        
        Returns:
            model: The loaded EnsembleGraphClassifier model ready for inference
            config: Dictionary containing model configuration and training results
        """
        import os
        import json
        import torch

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
        
        # BERT feature dimension
        in_channels = 768
        
        # Create the ensemble model with the appropriate submodels
        print(f"Creating model with parameters: {params}")
        model = cls(
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
                    num_layers=num_layers
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
        
        # Load the checkpoint
        checkpoint = torch.load(model_path, map_location=device)
        
        # Extract state dict
        if 'model_state_dict' in checkpoint:
            state_dict = checkpoint['model_state_dict']
        else:
            state_dict = checkpoint
        
        # Load the state dictionary with strict=False to handle missing keys
        model.load_state_dict(state_dict, strict=False)
        
        # Move model to device and set to evaluation mode
        model = model.to(device)
        model.eval()
        
        print(f"Model loaded with test accuracy: {config.get('test_accuracy', 'N/A')}")
        return model, config