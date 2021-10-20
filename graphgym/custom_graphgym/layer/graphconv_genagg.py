import torch
from torch_geometric.nn import GraphConv
from torch_geometric.graphgym.register import register_layer
from genagg.AggGNN import patch_conv_with_aggr
from genagg import GenAggSparse

from torch_geometric.graphgym.models.layer import LayerConfig

GraphConvGenAgg = patch_conv_with_aggr(GraphConv, GenAggSparse)

# Need to define a wrapper for GraphGym
class GraphConvGenAggWrapper(torch.nn.Module):
    def __init__(self, layer_config: LayerConfig, **kwargs):
        super().__init__()
        self.model = GraphConvGenAgg(layer_config.dim_in,
                                      layer_config.dim_out,
                                      bias=layer_config.has_bias)

    def forward(self, batch):
        batch.x = self.model(batch.x, batch.edge_index)
        return batch

register_layer('graphconv_genagg', GraphConvGenAggWrapper)
