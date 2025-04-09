import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.einops import rearrange, repeat


class FinePreprocess(nn.Module):
    def __init__(self, config):
        super().__init__()

        self.config = config
        self.cat_c_feat = config['fine_concat_coarse_feat']
        self.W = self.config['fine_window_size']

        d_model_c = self.config['coarse']['d_model']
        d_model_f = self.config['fine']['d_model']
        self.d_model_f = d_model_f
        if self.cat_c_feat:
            self.down_proj = nn.Linear(d_model_c, d_model_f, bias=True)
            self.merge_feat = nn.Linear(2*d_model_f, d_model_f, bias=True)

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.kaiming_normal_(p, mode="fan_out", nonlinearity="relu")

    def forward(self, feat_f0, feat_f1, feat_c0, feat_c1, data):
        W = self.W
        stride = data['hw0_f'][0] // data['hw0_c'][0]

        data.update({'W': W})
        if data['b_ids'].shape[0] == 0:
            feat0 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            feat1 = torch.empty(0, self.W**2, self.d_model_f, device=feat_f0.device)
            return feat0, feat1

        # img
        # 1. unfold(crop) all local windows
        feat_f0_unfold = F.unfold(feat_f0, kernel_size=(W, W), stride=stride, padding=W//2)
        feat_f0_unfold = rearrange(feat_f0_unfold, 'b (c ww) l -> b l ww c', ww=W**2)
        # 2. select only the predicted matches
        feat_f0_unfold = feat_f0_unfold[data['b_ids'], data['i_ids'], :, :]  # [b, l, ww, c]

        # pcd 
        feat_f1_sel = feat_f1[data['b_ids'], :, data['j_ids']]  # [b, c, s_l]
        feat_f1_sel = rearrange(feat_f0_unfold, 'b c l -> b l c')  # [b, c, s_l]
        feat_f1_unfold = feat_f1_sel.view(feat_f1_sel.shape[0], feat_c1.shape[1], self.config.neighbor_sel, -1)
        
        return feat_f0_unfold, feat_f1_unfold
