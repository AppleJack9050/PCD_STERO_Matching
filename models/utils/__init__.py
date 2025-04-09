import torch

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

def knn(src, tgt, k, normalize=False):
    '''
    Find K-nearest neighbor when ref==tgt and query==src
    Return index of knn, [B, N, k]
    '''
    dist = square_distance(src, tgt, normalize)
    _, idx = torch.topk(dist, k, dim=-1, largest=False, sorted=True)
    return idx