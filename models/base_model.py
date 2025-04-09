import torch
import torch.nn as nn
from einops.einops import rearrange
from .backbone.resnet_img import ResNetFPN_4_1
from .backbone.resnet_pcd import ResNet_PCD
from .module import LocalFeatureTransformer, FinePreprocess
from .module.pos_encoding import img_posenc, pcd_posenc
from .module.coarse_matching import CoarseMatching

class PICNet(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.img_ext = ResNetFPN_4_1(config)  # Extracts image features
        self.pcd_ext = ResNet_PCD(config)     # Extracts point cloud features

        self.coarse_matching = CoarseMatching(config['match_coarse'])
        self.fine_preprocess = FinePreprocess(config)

        self.ctxattn_coarse = LocalFeatureTransformer(config['coarse'])
        self.ctxattn_fine = LocalFeatureTransformer(config["fine"])

    def forward(self, data):
        # Extract features 
        img_feat_c, img_feat_f = self.img_ext(data['img']['feats'])  # ()
        pcd_feat_c, pcd_feat_f = self.img_ext(data['img']['feats'])  # ()
        
        # Coarse-level
        # 1. positional encoding and transformation
        img_feat_c = rearrange(img_posenc(img_feat_c, self.config), 'b c h w -> b (h w) c')
        pcd_feat_c = rearrange(pcd_posenc(pcd_feat_c, data, self.config), 'b c n -> b n c')
        # 2. cross-attention
        img_feat_c, pcd_feat_c = self.ctxattn_coarse(img_feat_c, pcd_feat_c)
        # 3. matching
        self.coarse_matching(img_feat_c, pcd_feat_c, data)
        
        # Fine-level
        # 1. feature refinement
        img_feat_f_unfold, pcd_feat_f_unfold = self.fine_preprocess(img_feat_f, pcd_feat_f, pcd_feat_c, pcd_feat_c, data)
        # 2. cross-attention
        img_feat_f_unfold, pcd_feat_f_unfold = self.ctxattn_fine(img_feat_f_unfold, pcd_feat_f_unfold)
        # 3. matching
        self.fine_matching(img_feat_f_unfold, pcd_feat_f_unfold, data)