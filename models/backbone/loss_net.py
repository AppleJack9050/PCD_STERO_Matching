from loguru import logger
import cv2
import torch
import torch.nn as nn
import numpy as np


class PCINet_Loss(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config  # config under the global namespace
        self.loss_config = config['loftr']['loss']
        self.match_type = self.config['loftr']['match_coarse']['match_type']
        self.sparse_spvs = self.config['loftr']['match_coarse']['sparse_spvs']
        
        # coarse-level
        self.correct_thr = self.loss_config['fine_correct_thr']
        self.c_pos_w = self.loss_config['pos_weight']
        self.c_neg_w = self.loss_config['neg_weight']
        # fine-level
        self.fine_type = self.loss_config['fine_type']

    def compute_coarse_loss(self, conf, conf_gt, weight=None):
        """ Point-wise CE / Focal Loss with 0 / 1 confidence as gt.
        Args:
            conf (torch.Tensor): (N, HW0, HW1) / (N, HW0+1, HW1+1)
            conf_gt (torch.Tensor): (N, HW0, HW1)
            weight (torch.Tensor): (N, HW0, HW1)
        """
        pos_mask, neg_mask = conf_gt == 1, conf_gt == 0
        c_pos_w, c_neg_w = self.c_pos_w, self.c_neg_w
        # corner case: no gt coarse-level match at all
        if not pos_mask.any():  # assign a wrong gt
            pos_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_pos_w = 0.
        if not neg_mask.any():
            neg_mask[0, 0, 0] = True
            if weight is not None:
                weight[0, 0, 0] = 0.
            c_neg_w = 0.

        if self.loss_config['coarse_type'] == 'cross_entropy':
            assert not self.sparse_spvs, 'Sparse Supervision for cross-entropy not implemented!'
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            loss_pos = - torch.log(conf[pos_mask])
            loss_neg = - torch.log(1 - conf[neg_mask])
            if weight is not None:
                loss_pos = loss_pos * weight[pos_mask]
                loss_neg = loss_neg * weight[neg_mask]
            return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
        elif self.loss_config['coarse_type'] == 'focal':
            conf = torch.clamp(conf, 1e-6, 1-1e-6)
            alpha = self.loss_config['focal_alpha']
            gamma = self.loss_config['focal_gamma']
            
            if self.sparse_spvs:
                pos_conf = conf[:, :-1, :-1][pos_mask] \
                            if self.match_type == 'sinkhorn' \
                            else conf[pos_mask]
                loss_pos = - alpha * torch.pow(1 - pos_conf, gamma) * pos_conf.log()
                # calculate losses for negative samples
                if self.match_type == 'sinkhorn':
                    neg0, neg1 = conf_gt.sum(-1) == 0, conf_gt.sum(1) == 0
                    neg_conf = torch.cat([conf[:, :-1, -1][neg0], conf[:, -1, :-1][neg1]], 0)
                    loss_neg = - alpha * torch.pow(1 - neg_conf, gamma) * neg_conf.log()
                else:
                    # These is no dustbin for dual_softmax, so we left unmatchable patches without supervision.
                    # we could also add 'pseudo negtive-samples'
                    pass
                # handle loss weights
                if weight is not None:
                    # Different from dense-spvs, the loss w.r.t. padded regions aren't directly zeroed out,
                    # but only through manually setting corresponding regions in sim_matrix to '-inf'.
                    loss_pos = loss_pos * weight[pos_mask]
                    if self.match_type == 'sinkhorn':
                        neg_w0 = (weight.sum(-1) != 0)[neg0]
                        neg_w1 = (weight.sum(1) != 0)[neg1]
                        neg_mask = torch.cat([neg_w0, neg_w1], 0)
                        loss_neg = loss_neg[neg_mask]
                
                loss =  c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean() \
                            if self.match_type == 'sinkhorn' \
                            else c_pos_w * loss_pos.mean()
                return loss
                # positive and negative elements occupy similar propotions. => more balanced loss weights needed
            else:  # dense supervision (in the case of match_type=='sinkhorn', the dustbin is not supervised.)
                loss_pos = - alpha * torch.pow(1 - conf[pos_mask], gamma) * (conf[pos_mask]).log()
                loss_neg = - alpha * torch.pow(conf[neg_mask], gamma) * (1 - conf[neg_mask]).log()
                if weight is not None:
                    loss_pos = loss_pos * weight[pos_mask]
                    loss_neg = loss_neg * weight[neg_mask]
                return c_pos_w * loss_pos.mean() + c_neg_w * loss_neg.mean()
                # each negative element occupy a smaller propotion than positive elements. => higher negative loss weight needed
        else:
            raise ValueError('Unknown coarse loss: {type}'.format(type=self.loss_config['coarse_type']))
        
    def compute_fine_loss(self, expec_f, expec_f_gt):
        if self.fine_type == 'l2_with_std':
            return self._compute_fine_loss_l2_std(expec_f, expec_f_gt)
        elif self.fine_type == 'l2':
            return self._compute_fine_loss_l2(expec_f, expec_f_gt)
        else:
            raise NotImplementedError()

    def _compute_fine_loss_l2(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 2] <x, y>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr
        if correct_mask.sum() == 0:
            if self.training:  # this seldomly happen when training, since we pad prediction with gt
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
            else:
                return None
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask]) ** 2).sum(-1)
        return offset_l2.mean()

    def _compute_fine_loss_l2_std(self, expec_f, expec_f_gt):
        """
        Args:
            expec_f (torch.Tensor): [M, 3] <x, y, std>
            expec_f_gt (torch.Tensor): [M, 2] <x, y>
        """
        # correct_mask tells you which pair to compute fine-loss
        correct_mask = torch.linalg.norm(expec_f_gt, ord=float('inf'), dim=1) < self.correct_thr

        # use std as weight that measures uncertainty
        std = expec_f[:, 2]
        inverse_std = 1. / torch.clamp(std, min=1e-10)
        weight = (inverse_std / torch.mean(inverse_std)).detach()  # avoid minizing loss through increase std

        # corner case: no correct coarse match found
        if not correct_mask.any():
            if self.training:  # this seldomly happen during training, since we pad prediction with gt
                               # sometimes there is not coarse-level gt at all.
                logger.warning("assign a false supervision to avoid ddp deadlock")
                correct_mask[0] = True
                weight[0] = 0.
            else:
                return None

        # l2 loss with std
        offset_l2 = ((expec_f_gt[correct_mask] - expec_f[correct_mask, :2]) ** 2).sum(-1)
        loss = (offset_l2 * weight[correct_mask]).mean()

        return loss
    
    @torch.no_grad()
    def compute_c_weight(self, data):
        """ compute element-wise weights for computing coarse-level loss. """
        if 'mask0' in data:
            c_weight = (data['mask0'].flatten(-2)[..., None] * data['mask1'].flatten(-2)[:, None]).float()
        else:
            c_weight = None
        return c_weight
    
    
    def compute_rotation_matrix_from_6d(poses):
        """
        Converts a 6D rotation representation into a 3x3 rotation matrix.
        Args:
            poses (Tensor): A tensor of shape (B, 6) where B is the batch size.
        Returns:
            Tensor: A tensor of shape (B, 3, 3) representing the rotation matrices.
        """
        a1 = poses[:, :3]
        a2 = poses[:, 3:]
        # First basis vector: normalized
        b1 = F.normalize(a1, dim=1)
        # Make a2 orthogonal to b1 and normalize
        b2 = F.normalize(a2 - (b1 * a2).sum(dim=1, keepdim=True) * b1, dim=1)
        # Third basis vector: cross product to ensure orthogonality
        b3 = torch.cross(b1, b2, dim=1)
        # Stack vectors to form rotation matrix
        R = torch.stack((b1, b2, b3), dim=2)  # Shape: (B, 3, 3)
        return R

    def pose_loss(predicted_pose, gt_pose):
        """
        Computes the combined pose loss between the predicted and ground truth poses.
        Both poses are provided as dictionaries with keys:
        - 'translation': Tensor of shape (B, 3)
        - 'rotation6d': Tensor of shape (B, 6)
        
        The loss consists of:
        - Translation loss: Mean Squared Error between translation vectors.
        - Rotation loss: Geodesic loss between rotation matrices.
        
        Args:
            predicted_pose (dict): Contains 'translation' and 'rotation6d' tensors.
            gt_pose (dict): Contains 'translation' and 'rotation6d' tensors.
            
        Returns:
            Tensor: The total loss (scalar).
        """
        # Compute translation loss
        pred_t = predicted_pose['translation']
        gt_t = gt_pose['translation']
        trans_loss = F.mse_loss(pred_t, gt_t)
        
        # Convert 6D rotation representations to 3x3 matrices
        pred_R = compute_rotation_matrix_from_6d(predicted_pose['rotation6d'])
        gt_R = compute_rotation_matrix_from_6d(gt_pose['rotation6d'])
        
        # Compute rotation loss via the geodesic distance between rotations
        # Relative rotation matrix: R_diff = R_pred^T * R_gt
        R_diff = torch.matmul(pred_R.transpose(1, 2), gt_R)
        # Compute the trace of R_diff for each sample
        trace = R_diff[:, 0, 0] + R_diff[:, 1, 1] + R_diff[:, 2, 2]
        # Clamp values to ensure they are in a valid range for arccos
        trace = torch.clamp(trace, -1 + 1e-6, 3 - 1e-6)
        # Geodesic distance (in radians)
        rot_loss = torch.acos((trace - 1) / 2).mean()
        
        # Combine the translation and rotation losses
        total_loss = trans_loss + rot_loss
        return total_loss

    def forward(self, data):
        # Define 3D object points (for example, corners of a square in the XY plane)
        objectPoints = [] # matched point cloud

        # Define corresponding 2D image points (in pixel coordinates)
        # These points should be in the same order as objectPoints.
        imagePoints = [] # matched pixel

        # Define the camera intrinsic parameters (example values)
        fx, fy, cx, cy = data['img']['intrinsic_params']
        cameraMatrix = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0,  0,  1]
        ], dtype=np.float32)

        # Assume no lens distortion (or provide your distortion coefficients if available)
        distCoeffs = np.zeros((4, 1), dtype=np.float32)

        # Set RANSAC parameters
        iterationsCount = 100      # Number of RANSAC iterations
        reprojectionError = 8.0      # Reprojection error threshold in pixels
        confidence = 0.99            # Confidence level
        retval, rvec, tvec, inliers = cv2.solvePnPRansac(
            objectPoints, imagePoints, cameraMatrix, distCoeffs,
            iterationsCount=iterationsCount,
            reprojectionError=reprojectionError,
            confidence=confidence,
            flags=cv2.SOLVEPNP_ITERATIVE
        )

        # prdicted pose
        pred_pose = {'translation': tvec, 'rotation6d': rvec}

        # ground truth
        gt_pose = data['img']['pose']

        loss = pose_loss(pred_pose, gt_pose)

        data.update({"loss": loss})
