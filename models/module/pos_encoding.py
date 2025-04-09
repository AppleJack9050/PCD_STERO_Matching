import math
import torch
from torch import nn
from typing import Any, Dict
from models.backbone.AttentionModules.cross import PointEncoder


class PositionEncodingSine(nn.Module):
    """
    This is a sinusoidal position encoding that generalized to 2-dimensional images
    """

    def __init__(self, d_model, max_shape=(256, 256), temp_bug_fix=True):
        """
        Args:
            max_shape (tuple): for 1/8 featmap, the max length of 256 corresponds to 2048 pixels
            temp_bug_fix (bool): As noted in this [issue](https://github.com/zju3dv/LoFTR/issues/41),
                the original implementation of LoFTR includes a bug in the pos-enc impl, which has little impact
                on the final performance. For now, we keep both impls for backward compatability.
                We will remove the buggy impl after re-training all variants of our released models.
        """
        super().__init__()

        pe = torch.zeros((d_model, *max_shape))
        y_position = torch.ones(max_shape).cumsum(0).float().unsqueeze(0)
        x_position = torch.ones(max_shape).cumsum(1).float().unsqueeze(0)
        if temp_bug_fix:
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / (d_model//2)))
        else:  # a buggy implementation (for backward compatability only)
            div_term = torch.exp(torch.arange(0, d_model//2, 2).float() * (-math.log(10000.0) / d_model//2))
        div_term = div_term[:, None, None]  # [C//4, 1, 1]
        pe[0::4, :, :] = torch.sin(x_position * div_term)
        pe[1::4, :, :] = torch.cos(x_position * div_term)
        pe[2::4, :, :] = torch.sin(y_position * div_term)
        pe[3::4, :, :] = torch.cos(y_position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # [1, C, H, W]

    def forward(self, x):
        """
        Args:
            x: [N, C, H, W]
        """
        return x + self.pe[:, :, :x.size(2), :x.size(3)]

def img_posenc(features: torch.Tensor, config: Dict[str, Any]) -> torch.Tensor:
    """
    Applies sine-based positional encoding to image features.

    This function instantiates the PositionEncodingSine module using configuration
    parameters from the 'coarse' section and applies it to the input feature tensor.

    Args:
        features (torch.Tensor): The input tensor representing image features.
        config (Dict[str, Any]): A configuration dictionary that must include a 'coarse' key
            containing 'd_model' (model dimension) and 'temp_bug_fix' settings.

    Returns:
        torch.Tensor: The feature tensor after applying sine-based positional encoding.
    """
    # Create an instance of the positional encoding module using provided config parameters.
    pos_encoding = PositionEncodingSine(
        d_model=config['coarse']['d_model'],
        temp_bug_fix=config['coarse']['temp_bug_fix']
    )
    
    # Apply the positional encoding to the features and return the result.
    return pos_encoding(features)



def pcd_posenc(features: torch.Tensor, data: Dict[str, torch.Tensor], config: Any) -> torch.Tensor:
    """
    Applies positional encoding to point cloud features.

    This function instantiates a PointEncoder using configuration parameters,
    centers the last set of point coordinates by subtracting their mean,
    and adds the resulting positional encoding to the input features.

    Args:
        features (torch.Tensor): Input features tensor.
        data (Dict[str, torch.Tensor]): Dictionary containing point data. Expected to have a key 'points'.
        config (Any): Configuration object containing attributes like `gnn_feats_dim`.

    Returns:
        torch.Tensor: The features tensor updated with positional encoding.
    """
    # Initialize the point encoder with the desired dimensions and hidden layer sizes.
    point_encoder = PointEncoder(config.gnn_feats_dim, [32, 64, 128, 256])
    
    # Extract the last set of point coordinates.
    point_coordinates = data['points'][-1]
    
    # Center the coordinates by subtracting the mean along the last dimension.
    centered_coordinates = point_coordinates - torch.mean(point_coordinates, dim=-1, keepdim=True)
    
    # Apply the point encoder and add the encoding to the input features.
    encoded_position = point_encoder(centered_coordinates)
    
    return features + encoded_position