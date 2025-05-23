import torch
import numpy as np
import cpp_wrappers.cpp_subsampling.grid_subsampling as cpp_subsampling
import cpp_wrappers.cpp_neighbors.radius_neighbors as cpp_neighbors
import time

class Timer(object):
    '''
    A simple timer
    '''

    def __init__(self):
        self.reset()

    def reset(self):
        self.total_time = 0.
        self.calls = 0
        self.start_time = 0.
        self.diff = 0.
        self.avg = 0.

    def tic(self):
        self.start_time = time.time()

    def toc(self, average=True):
        self.diff = time.time() - self.start_time
        self.total_time += self.diff
        self.calls += 1
        self.avg = self.total_time / self.calls
        if average:
            return self.avg
        else:
            return self.diff

def square_distance(src, tgt, normalize=False):
    '''
    Calculate Euclide distance between every two points
    :param src: source point cloud in shape [B, N, C]
    :param tgt: target point cloud in shape [B, M, C]
    :param normalize: whether to normalize calculated distances
    :return:
    '''

    B, N, _ = src.shape
    _, M, _ = tgt.shape
    dist = -2. * torch.matmul(src, tgt.permute(0, 2, 1).contiguous())
    if normalize:
        dist += 2
    else:
        dist += torch.sum(src ** 2, dim=-1).unsqueeze(-1)
        dist += torch.sum(tgt ** 2, dim=-1).unsqueeze(-2)

    dist = torch.clamp(dist, min=1e-12, max=None)
    return dist

def point2node(nodes, points):
    '''
    Assign each point to a certain node according to nearest neighbor search
    :param nodes: [M, 3]
    :param points: [N, 3]
    :return: idx [N], indicating the id of node that each point belongs to
    '''
    M, _ = nodes.size()
    N, _ = points.size()
    dist = square_distance(points.unsqueeze(0), nodes.unsqueeze(0))[0]

    idx = dist.topk(k=1, dim=-1, largest=False)[1] #[B, N, 1], ignore the smallest element as it's the query itself

    idx = idx.squeeze(-1)
    return idx

def point2node_correspondences(src_nodes, src_points, tgt_nodes, tgt_points, point_correspondences, thres=0., device='cpu'):
    '''
    Based on point correspondences & point2node relationships, calculate node correspondences
    :param src_nodes: Nodes of source point cloud
    :param src_points: Points of source point cloud
    :param tgt_nodes: Nodes of target point cloud
    :param tgt_points: Points of target point cloud
    :param point_correspondences: Ground truth point correspondences
    :return: node_corr_mask: Overlap ratios between nodes
             node_corr: Node correspondences sampled for training
    '''
    #####################################
    # calc visible ratio for each node
    src_visible, tgt_visible = point_correspondences[:, 0], point_correspondences[:, 1]

    src_vis, tgt_vis = torch.zeros((src_points.shape[0])).to(device), torch.zeros((tgt_points.shape[0])).to(device)

    src_vis[src_visible] = 1.
    tgt_vis[tgt_visible] = 1.

    src_vis = src_vis.nonzero().squeeze(1)
    tgt_vis = tgt_vis.nonzero().squeeze(1)

    src_vis_num = torch.zeros((src_nodes.shape[0])).to(device)
    src_tot_num = torch.ones((src_nodes.shape[0])).to(device)

    src_idx = point2node(src_nodes, src_points)
    idx, cts = torch.unique(src_idx, return_counts=True)
    src_tot_num[idx] = cts.float()

    src_idx_ = src_idx[src_vis]
    idx_, cts_ = torch.unique(src_idx_, return_counts=True)
    src_vis_num[idx_] = cts_.float()

    src_node_vis = src_vis_num / src_tot_num

    tgt_vis_num = torch.zeros((tgt_nodes.shape[0])).to(device)
    tgt_tot_num = torch.ones((tgt_nodes.shape[0])).to(device)


    tgt_idx = point2node(tgt_nodes, tgt_points)
    idx, cts = torch.unique(tgt_idx, return_counts=True)
    tgt_tot_num[idx] = cts.float()

    tgt_idx_ = tgt_idx[tgt_vis]
    idx_, cts_ = torch.unique(tgt_idx_, return_counts=True)
    tgt_vis_num[idx_] = cts_.float()

    tgt_node_vis = tgt_vis_num / tgt_tot_num

    src_corr = point_correspondences[:, 0]  # [K]
    tgt_corr = point_correspondences[:, 1]  # [K]

    src_node_corr = torch.gather(src_idx, 0, src_corr)
    tgt_node_corr = torch.gather(tgt_idx, 0, tgt_corr)

    index = src_node_corr * tgt_idx.shape[0] + tgt_node_corr

    index, counts = torch.unique(index, return_counts=True)


    src_node_corr = index // tgt_idx.shape[0]
    tgt_node_corr = index % tgt_idx.shape[0]

    node_correspondences = torch.zeros(size=(src_nodes.shape[0] + 1, tgt_nodes.shape[0] + 1), dtype=torch.float32).to(device)

    node_corr_mask = torch.zeros(size=(src_nodes.shape[0] + 1, tgt_nodes.shape[0] + 1), dtype=torch.float32).to(device)
    node_correspondences[src_node_corr, tgt_node_corr] = counts.float()
    node_correspondences = node_correspondences[:-1, :-1]

    node_corr_sum_row = torch.sum(node_correspondences, dim=1, keepdim=True)
    node_corr_sum_col = torch.sum(node_correspondences, dim=0, keepdim=True)

    node_corr_norm_row = (node_correspondences / (node_corr_sum_row + 1e-10)) * src_node_vis.unsqueeze(1).expand(src_nodes.shape[0], tgt_nodes.shape[0])

    node_corr_norm_col = (node_correspondences / (node_corr_sum_col + 1e-10)) * tgt_node_vis.unsqueeze(0).expand(src_nodes.shape[0], tgt_nodes.shape[0])

    node_corr_mask[:-1, :-1] = torch.min(node_corr_norm_row, node_corr_norm_col)
    ############################################################
    # Binary masks
    #node_corr_mask[:-1, :-1] = (node_corr_mask[:-1, :-1] > 0.01)
    #node_corr_mask[-1, :-1] = torch.clamp(1. - torch.sum(node_corr_mask[:-1, :-1], dim=0), min=0.)
    #node_corr_mask[:-1, -1] = torch.clamp(1. - torch.sum(node_corr_mask[:-1, :-1], dim=1), min=0.)
    re_mask = (node_corr_mask[:-1, :-1] > thres)
    node_corr_mask[:-1, :-1] = node_corr_mask[:-1, :-1] * re_mask

    #####################################################
    # Soft weighted mask, best Performance
    node_corr_mask[:-1, -1] = 1. - src_node_vis
    node_corr_mask[-1, :-1] = 1. - tgt_node_vis
    #####################################################

    node_corr = node_corr_mask[:-1, :-1].nonzero()
    return node_corr_mask, node_corr

def batch_grid_subsampling_kpconv(points, batches_len, features=None, labels=None, sampleDl=0.1, max_p=0, verbose=0, random_grid_orient=True):
    """
    CPP wrapper for a grid subsampling (method = barycenter for points and features)
    """
    if (features is None) and (labels is None):
        s_points, s_len = cpp_subsampling.subsample_batch(points,
                                                          batches_len,
                                                          sampleDl=sampleDl,
                                                          max_p=max_p,
                                                          verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len)

    elif (labels is None):
        s_points, s_len, s_features = cpp_subsampling.subsample_batch(points,
                                                                      batches_len,
                                                                      features=features,
                                                                      sampleDl=sampleDl,
                                                                      max_p=max_p,
                                                                      verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features)

    elif (features is None):
        s_points, s_len, s_labels = cpp_subsampling.subsample_batch(points,
                                                                    batches_len,
                                                                    classes=labels,
                                                                    sampleDl=sampleDl,
                                                                    max_p=max_p,
                                                                    verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_labels)

    else:
        s_points, s_len, s_features, s_labels = cpp_subsampling.subsample_batch(points,
                                                                              batches_len,
                                                                              features=features,
                                                                              classes=labels,
                                                                              sampleDl=sampleDl,
                                                                              max_p=max_p,
                                                                              verbose=verbose)
        return torch.from_numpy(s_points), torch.from_numpy(s_len), torch.from_numpy(s_features), torch.from_numpy(s_labels)


def batch_neighbors_kpconv(queries, supports, q_batches, s_batches, radius, max_neighbors):
    """
    Computes neighbors for a batch of queries and supports, apply radius search
    :param queries: (N1, 3) the query points
    :param supports: (N2, 3) the support points
    :param q_batches: (B) the list of lengths of batch elements in queries
    :param s_batches: (B)the list of lengths of batch elements in supports
    :param radius: float32
    :return: neighbors indices
    """

    neighbors = cpp_neighbors.batch_query(queries, supports, q_batches, s_batches, radius=radius)
    if max_neighbors > 0:
        return torch.from_numpy(neighbors[:, :max_neighbors])
    else:
        return torch.from_numpy(neighbors)


def collate_fn_descriptor(list_data, config, neighborhood_limits):
    batched_points_list = []
    batched_features_list = []
    batched_lengths_list = []
    assert len(list_data) == 1

    for ind, (
    src_pcd, tgt_pcd, src_feats, tgt_feats, rot, trans, matching_inds, src_pcd_raw, tgt_pcd_raw, sample) in enumerate(
            list_data):
        batched_points_list.append(src_pcd)
        batched_points_list.append(tgt_pcd)
        batched_features_list.append(src_feats)
        batched_features_list.append(tgt_feats)
        batched_lengths_list.append(len(src_pcd))
        batched_lengths_list.append(len(tgt_pcd))

    batched_features = torch.from_numpy(np.concatenate(batched_features_list, axis=0))
    batched_points = torch.from_numpy(np.concatenate(batched_points_list, axis=0))
    batched_lengths = torch.from_numpy(np.array(batched_lengths_list)).int()

    # Starting radius of convolutions
    r_normal = config.first_subsampling_dl * config.conv_radius

    # Starting layer
    layer_blocks = []
    layer = 0

    # Lists of inputs
    input_points = []
    input_neighbors = []
    input_pools = []
    input_upsamples = []
    input_batches_len = []

    for block_i, block in enumerate(config.architecture):

        # Stop when meeting a global pooling or upsampling
        if 'global' in block or 'upsample' in block:
            break

        # Get all blocks of the layer
        if not ('pool' in block or 'strided' in block):
            layer_blocks += [block]
            if block_i < len(config.architecture) - 1 and not ('upsample' in config.architecture[block_i + 1]):
                continue

        # Convolution neighbors indices
        # *****************************

        if layer_blocks:
            # Convolutions are done in this layer, compute the neighbors with the good radius
            if np.any(['deformable' in blck for blck in layer_blocks[:-1]]):
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal
            conv_i = batch_neighbors_kpconv(batched_points, batched_points, batched_lengths, batched_lengths, r,
                                            neighborhood_limits[layer])

        else:
            # This layer only perform pooling, no neighbors required
            conv_i = torch.zeros((0, 1), dtype=torch.int64)

        # Pooling neighbors indices
        # *************************

        # If end of layer is a pooling operation
        if 'pool' in block or 'strided' in block:

            # New subsampling length
            dl = 2 * r_normal / config.conv_radius

            # Subsampled points
            pool_p, pool_b = batch_grid_subsampling_kpconv(batched_points, batched_lengths, sampleDl=dl)

            # Radius of pooled neighbors
            if 'deformable' in block:
                r = r_normal * config.deform_radius / config.conv_radius
            else:
                r = r_normal

            # Subsample indices
            pool_i = batch_neighbors_kpconv(pool_p, batched_points, pool_b, batched_lengths, r,
                                            neighborhood_limits[layer])

            # Upsample indices (with the radius of the next layer to keep wanted density)
            up_i = batch_neighbors_kpconv(batched_points, pool_p, batched_lengths, pool_b, 2 * r,
                                          neighborhood_limits[layer])

        else:
            # No pooling in the end of this layer, no pooling indices required
            pool_i = torch.zeros((0, 1), dtype=torch.int64)
            pool_p = torch.zeros((0, 3), dtype=torch.float32)
            pool_b = torch.zeros((0,), dtype=torch.int64)
            up_i = torch.zeros((0, 1), dtype=torch.int64)

        # Updating input lists
        input_points += [batched_points.float()]
        input_neighbors += [conv_i.long()]
        input_pools += [pool_i.long()]
        input_upsamples += [up_i.long()]
        input_batches_len += [batched_lengths]

        # New points for next layer
        batched_points = pool_p
        batched_lengths = pool_b

        # Update radius and reset blocks
        r_normal *= 2
        layer += 1
        layer_blocks = []

    len_src_pcd = input_batches_len[0][0]
    pcds = input_points[0]
    src_pcd, tgt_pcd = pcds[:len_src_pcd], pcds[len_src_pcd:]

    len_src_nodes = input_batches_len[config.node_id][0]
    nodes = input_points[config.node_id]
    src_node, tgt_node = nodes[:len_src_nodes], nodes[len_src_nodes:]
    if not config.mode == 'test':
        matching_mask, node_corr = point2node_correspondences(src_node, src_pcd, tgt_node, tgt_pcd, matching_inds, thres=config['patch_overlap_thres'])
    else:
        matching_mask, node_corr = None, None

        ###############
        # Return inputs
        ###############
    dict_inputs = {
        'points': input_points,
        'neighbors': input_neighbors,
        'pools': input_pools,
        'upsamples': input_upsamples,
        'features': batched_features.float(),
        'stack_lengths': input_batches_len,
        'rot': torch.from_numpy(rot),
        'trans': torch.from_numpy(trans),
        'correspondences': matching_inds,
        'src_pcd_raw': torch.from_numpy(src_pcd_raw).float(),
        'tgt_pcd_raw': torch.from_numpy(tgt_pcd_raw).float(),
        'sample': sample,
        'matching_mask': matching_mask,
        'node_corr': node_corr
    }
    return dict_inputs


def calibrate_neighbors(dataset, config, collate_fn, keep_ratio=0.8, samples_threshold=2000):
    timer = Timer()
    last_display = timer.total_time

    # From config parameter, compute higher bound of neighbors number in a neighborhood
    hist_n = int(np.ceil(4 / 3 * np.pi * (config.deform_radius + 1) ** 3))
    neighb_hists = np.zeros((config.num_layers, hist_n), dtype=np.int32)

    # Get histogram of neighborhood sizes i in 1 epoch max.
    for i in range(len(dataset)):
        timer.tic()
        batched_input = collate_fn([dataset[i]], config, neighborhood_limits=[hist_n] * 5)
        # update histogram
        counts = [torch.sum(neighb_mat < neighb_mat.shape[0], dim=1).numpy() for neighb_mat in batched_input['neighbors']]
        hists = [np.bincount(c, minlength=hist_n)[:hist_n] for c in counts]
        neighb_hists += np.vstack(hists)
        timer.toc()

        if timer.total_time - last_display > 0.1:
            last_display = timer.total_time
            print(f"Calib Neighbors {i:08d}: timings {timer.total_time:4.2f}s")

        if np.min(np.sum(neighb_hists, axis=1)) > samples_threshold:
            break

    cumsum = np.cumsum(neighb_hists.T, axis=0)
    percentiles = np.sum(cumsum < (keep_ratio * cumsum[hist_n - 1, :]), axis=0)

    neighborhood_limits = percentiles
    print('\n')

    return neighborhood_limits
