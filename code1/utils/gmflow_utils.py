import torch
import torch.nn.functional as F
from collections import OrderedDict

"""
This code is from GMFlow [CVPR'22] and MatchNeRF [arxiv'23]
"""

def generate_window_grid(h_min, h_max, w_min, w_max, len_h, len_w, device=None):
    assert device is not None

    x, y = torch.meshgrid([torch.linspace(w_min, w_max, len_w, device=device),
                           torch.linspace(h_min, h_max, len_h, device=device)],
                          indexing='ij')
    grid = torch.stack((x, y), -1).transpose(0, 1).float()  # [H, W, 2]

    return grid

def split_feature(feature,
                  num_splits=2,
                  channel_last=False,
                  ):
    if channel_last:  # [B, H, W, C]
        b, h, w, c = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, num_splits, h // num_splits, num_splits, w // num_splits, c
                               ).permute(0, 1, 3, 2, 4, 5).reshape(b_new, h_new, w_new, c)  # [B*K*K, H/K, W/K, C]
    else:  # [B, C, H, W]
        b, c, h, w = feature.size()
        assert h % num_splits == 0 and w % num_splits == 0

        b_new = b * num_splits * num_splits
        h_new = h // num_splits
        w_new = w // num_splits

        feature = feature.view(b, c, num_splits, h // num_splits, num_splits, w // num_splits
                               ).permute(0, 2, 4, 1, 3, 5).reshape(b_new, c, h_new, w_new)  # [B*K*K, C, H/K, W/K]

    return feature


def merge_splits(splits,
                 num_splits=2,
                 channel_last=False,
                 ):
    if channel_last:  # [B*K*K, H/K, W/K, C]
        b, h, w, c = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, h, w, c)
        merge = splits.permute(0, 1, 3, 2, 4, 5).contiguous().view(
            new_b, num_splits * h, num_splits * w, c)  # [B, H, W, C]
    else:  # [B*K*K, C, H/K, W/K]
        b, c, h, w = splits.size()
        new_b = b // num_splits // num_splits

        splits = splits.view(new_b, num_splits, num_splits, c, h, w)
        merge = splits.permute(0, 3, 1, 4, 2, 5).contiguous().view(
            new_b, c, num_splits * h, num_splits * w)  # [B, C, H, W]

    return merge


def normalize_img(img0, img1):
    # loaded images are in [0, 255]
    # normalize by ImageNet mean and std
    mean = torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1).to(img1.device)
    std = torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1).to(img1.device)
    img0 = (img0 / 255. - mean) / std
    img1 = (img1 / 255. - mean) / std

    return img0, img1


def sample_features_by_grid(raw_whole_feats, grid, align_corners=True, mode='bilinear', padding_mode='border',
                            local_radius=0, local_dilation=1):
    if local_radius <= 0:
        return F.grid_sample(raw_whole_feats, grid, align_corners=align_corners, mode=mode, padding_mode=padding_mode)

    # --- sample on a local grid
    # unnomarlize original gird
    h, w = raw_whole_feats.shape[-2:]
    c = torch.Tensor([(w - 1) / 2., (h - 1) / 2.]).float().to(grid.device)  # inverse scale
    unnorm_grid = (grid * c + c).reshape(grid.shape[0], -1, 2)  # [B, n_rays*n_pts, 2]
    # build local grid
    local_h = 2 * local_radius + 1
    local_w = 2 * local_radius + 1
    window_grid = generate_window_grid(-local_radius, local_radius,
                                       -local_radius, local_radius,
                                       local_h, local_w, device=raw_whole_feats.device)  # [2R+1, 2R+1, 2]
    window_grid = window_grid.reshape(1, -1, 2).repeat(grid.shape[0], 1, 1) * local_dilation  # [B, (2R+1)^2, 2]
    # merge grid and normalize
    sample_grid = unnorm_grid.unsqueeze(2) + window_grid.unsqueeze(1)  # [B, n_rays*n_pts, (2R+1)^2, 2]
    c = torch.Tensor([(w + local_w * local_dilation - 1) / 2.,
                    (h + local_h * local_dilation - 1) / 2.]).float().to(sample_grid.device)  # inverse scale
    norm_sample_grid = (sample_grid - c) / c  # range (-1, 1)
    # sample features
    sampled_feats = F.grid_sample(raw_whole_feats, norm_sample_grid,
                                align_corners=align_corners, mode=mode, padding_mode=padding_mode)  # [B, C, n_rays*n_pts, (2R+1)^2]
    # merge features of local grid
    b, c, n = sampled_feats.shape[:3]
    n_rays, n_pts = grid.shape[1:3]
    sampled_feats = sampled_feats.reshape(b, c*n, local_h, local_w)  # [B, C*n_rays*n_pts, 2R+1, 2R+1]
    avg_feats = F.adaptive_avg_pool2d(sampled_feats, (1, 1))  # [B, C*n_rays*n_pts, 1, 1]
    avg_feats = avg_feats.reshape(b, c, n_rays, n_pts)
    return avg_feats

