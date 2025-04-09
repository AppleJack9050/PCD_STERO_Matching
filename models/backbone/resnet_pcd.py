import torch
import torch.nn as nn
from .KPConv.models import KPEncoder, KPDecoder
import cpp_wrappers.grouping.lib.grouping_cuda as grouping

class ResNet_PCD(nn.Module):
    """
    A point cloud processing network that uses a KPConv encoder-decoder 
    architecture along with region grouping via a custom CUDA extension.
    """
    def __init__(self, config):
        super().__init__()
        self.encoder = KPEncoder(config=config)
        self.decoder = KPDecoder(config=config)
        self.neighbor_sel = config['neighbor_sel']

    def forward(self, batch):
        """
        Forward pass for the network.
        
        Args:
            batch (dict): A dictionary containing point cloud data under the key 'points'.
                          The first element is the input point cloud, and the last element
                          is the node set for grouping.
                          
        Returns:
            tuple: A tuple (feats_c, feats_f) where:
                   - feats_c: Encoder features
                   - feats_f: Decoder output features
        """
        # Extract the point cloud and the nodes for grouping
        point_cloud = batch['points'][0]
        nodes = batch['points'][-1]
        
        # Find nearest neighbor indices for region splitting and grouping
        pcd_id = knn(point_cloud.unsqueeze(0), nodes.unsqueeze(0), k=1)
        
        # Create a patch tensor on the same device as the input nodes
        patch_shape = (1, nodes.shape[0], self.neighbor_sel)
        patch = -torch.ones(patch_shape, dtype=torch.int, device=nodes.device)
        
        # Grouping operation via custom CUDA wrapper
        grouping.grouping_wrapper(
            patch.shape[0],
            pcd_id.shape[1],
            patch.shape[1],
            patch.shape[2],
            pcd_id.int(),
            patch.int()
        )
        patch[patch < 0] = 0
        
        # Encoder: Extract features and intermediate skip connections
        feats_c, skip_x = self.encoder(batch)
        
        # Decoder: Reconstruct features using the encoder outputs and skip connections
        feats_f = self.decoder(batch, feats_c, skip_x)
        
        return feats_c, feats_f
