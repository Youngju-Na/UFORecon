import numpy as np 

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import (rearrange, reduce, repeat)

from .encoder_utils.grid_sample import grid_sample_2d, grid_sample_3d
from .attention.transformer import LocalFeatureTransformer

import math
PI = math.pi

class PositionEncoding(nn.Module):
    def __init__(self, L=10):
        super().__init__()
        self.L = L
        self.augmented = rearrange((PI * 2 ** torch.arange(-1, self.L - 1)), "L -> L 1 1 1")

    def forward(self, x):
        sin_term = torch.sin(self.augmented.type_as(x) * rearrange(x, "RN SN Dim -> 1 RN SN Dim")) # BUG? 
        cos_term = torch.cos(self.augmented.type_as(x) * rearrange(x, "RN SN Dim -> 1 RN SN Dim") )
        sin_cos_term = torch.stack([sin_term, cos_term])

        sin_cos_term = rearrange(sin_cos_term, "Num2 L RN SN Dim -> (RN SN) (L Num2 Dim)")

        return sin_cos_term
    
class PositionalEncoding_NeRF(torch.nn.Module):
    """
    Implement NeRF's positional encoding
    """

    def __init__(self, num_freqs=6, d_in=3, freq_factor=np.pi, include_input=False):
        super().__init__()
        self.num_freqs = num_freqs
        self.d_in = d_in
        self.freqs = freq_factor * 2.0 ** torch.arange(0, num_freqs)
        self.d_out = self.num_freqs * 2 * d_in
        self.include_input = include_input
        if include_input:
            self.d_out += d_in
        # f1 f1 f2 f2 ... to multiply x by
        self.register_buffer(
            "_freqs", torch.repeat_interleave(self.freqs, 2).view(1, -1, 1)
        )
        # 0 pi/2 0 pi/2 ... so that
        # (sin(x + _phases[0]), sin(x + _phases[1]) ...) = (sin(x), cos(x)...)
        _phases = torch.zeros(2 * self.num_freqs)
        _phases[1::2] = np.pi * 0.5
        self.register_buffer("_phases", _phases.view(1, -1, 1))

    def forward(self, x):
        """
        Apply positional encoding (new implementation)
        :param x (D1, D2, ..., self.d_in)
        :return (D1, D2, ..., self.d_out)
        """

        # flattening x if it has leading batch dimensions
        original_shape = x.shape
        if len(original_shape) > 2:
            x = x.reshape(-1, original_shape[-1])

        embed = x.unsqueeze(1).repeat(1, self.num_freqs * 2, 1)
        embed = torch.sin(torch.addcmul(self._phases, embed, self._freqs))
        embed = embed.view(x.shape[0], -1)
        if self.include_input:
            embed = torch.cat((x, embed), dim=-1)

        if embed.shape[:-1] != original_shape[:-1]:
            embed = embed.reshape(*original_shape[:-1], self.d_out)
        return embed

    @classmethod
    def from_conf(cls, conf, d_in=3):
        # PyHocon construction
        return cls(
            conf.get_int("num_freqs", 6),
            d_in,
            conf.get_float("freq_factor", np.pi),
            conf.get_bool("include_input", True),
        )


class RayTransformer(nn.Module):
    """
    Ray transformer
    """

    def __init__(self, args, img_feat_dim=32, fea_volume_dim=24, sim_feat_dim=26):

        super().__init__()

        self.args = args
        self.offset =  [[0, 0, 0]]

        self.volume_reso = args.volume_reso # 96
        self.only_volume = False
        if self.only_volume:
            assert self.volume_reso > 0, "if only use volume feature, must have volume"

        self.depthcode = None
        if self.args.mvs_depth_guide > 0  and self.args.depth_pos_encoding:
            self.depthcode = PositionalEncoding_NeRF(num_freqs=4, d_in=1) #TODO: change input
        
        self.depth_dim = self.depthcode.d_out if self.args.depth_pos_encoding else 0
        
        if not self.args.mvs_depth_guide > 0:
            self.depth_dim = 0
        
        if self.args.use_dir_srdf:
            self.dircode = PositionalEncoding_NeRF(num_freqs=4, d_in=3, include_input=True) #TODO: change input
                
        if self.args.use_dir_srdf:
            self.dirdim = self.dircode.d_out
        else:
            self.dirdim = 0

        self.img_feat_dim = img_feat_dim 
        self.sim_feat_dim = 8 if self.args.explicit_similarity else 0
        self.sim_feat_fix = 16 if self.args.explicit_similarity else 0
        self.fea_volume_dim = fea_volume_dim if self.volume_reso > 0 else 0
        
        self.PE_d_hid = 8
        
        if self.sim_feat_dim > 0: #* assumes self.args.explicit_similarity = True
            self.pre_sim_mlp = nn.Sequential(
                nn.Linear(self.sim_feat_dim, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 32), nn.ReLU(inplace=True),
                nn.Linear(32, self.sim_feat_fix),
            )
        
        # transformers
        self.density_view_transformer = LocalFeatureTransformer(d_model=self.img_feat_dim + self.fea_volume_dim + self.sim_feat_fix + self.depth_dim + self.dirdim, 
                                    nhead=8, layer_names=['self'], attention='linear')

        self.density_ray_transformer = LocalFeatureTransformer(d_model=self.img_feat_dim + self.PE_d_hid + self.fea_volume_dim + self.sim_feat_fix + self.depth_dim + self.dirdim, 
                                    nhead=8, layer_names=['self'], attention='linear')

        if self.only_volume:
            self.DensityMLP = nn.Sequential(
                nn.Linear(self.fea_volume_dim, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1))
        else:
            self.DensityMLP = nn.Sequential(
                nn.Linear(self.img_feat_dim + self.PE_d_hid + self.fea_volume_dim+self.sim_feat_fix+self.depth_dim+self.dirdim, 32), nn.ReLU(inplace=True),
                nn.Linear(32, 16), nn.ReLU(inplace=True),
                nn.Linear(16, 1))

        self.relu = nn.ReLU(inplace=True)

        # learnable view token
        self.viewToken = ViewTokenNetwork(dim=self.img_feat_dim + self.fea_volume_dim + self.sim_feat_fix + self.depth_dim + self.dirdim)
        self.softmax = nn.Softmax(dim=-2)

        # to calculate radiance weight
        self.linear_radianceweight_1_softmax = nn.Sequential(
            nn.Linear(self.img_feat_dim+3+self.fea_volume_dim+self.sim_feat_fix+self.depth_dim, 16), nn.ReLU(inplace=True),
            nn.Linear(16, 8), nn.ReLU(inplace=True),
            nn.Linear(8, 1),
        )

    def order_posenc(self, d_hid, n_samples):
        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]
        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table)

        return sinusoid_table

    def forward(self, point3D, batch, source_imgs_feat, fea_volume=None, cond_info=None, points_projected=None, mask_valid=None):

        B, NV, _, H, W = batch['source_imgs'].shape
        _, RN, SN, _ = point3D.shape
        FDim = source_imgs_feat.size(2) # feature dim
        CN = len(self.offset)
        
        s_idx = batch['start_idx'] if 'start_idx' in batch.keys() else 1 #* start_idx should be zero in inference and one in training
        
        # calculate relative direction (DimX: 3)
        vector_1 = (point3D - repeat(batch['ref_pose_inv'][:,:3,-1], "B DimX -> B 1 1 DimX")) #*batch['ref_pose_inv'][:,:3,-1] denotes camera position
        vector_1 = repeat(vector_1, "B RN SN DimX -> B 1 RN SN DimX")
        vector_2 = (point3D.unsqueeze(1) - repeat(batch['source_poses_inv'][:,:,:3,-1], "B L DimX -> B L 1 1 DimX")) # B L RN SN DimX
        vector_1 = vector_1/torch.linalg.norm(vector_1, dim=-1, keepdim=True) # normalize to get direction
        vector_2 = vector_2/torch.linalg.norm(vector_2, dim=-1, keepdim=True)
        dir_relative = vector_1 - vector_2 #* relative direction from source to reference
        dir_relative = dir_relative.float() #* (B, source_views, RN, SN, DimX)

        if self.args.volume_reso > 0: #* 96
            assert fea_volume != None
            if self.args.volume_type == "featuregrid":
                fea_volume_feat = grid_sample_3d(fea_volume, point3D.unsqueeze(1).float())
                fea_volume_feat = rearrange(fea_volume_feat, "B C RN SN -> (B RN SN) C")
            elif self.args.volume_type == "correlation":
                fea_volume_feat = rearrange(fea_volume, "B RN SN C -> (B RN SN) C")
            
        # -------- project points to feature map
        
        # B NV RN SN CN DimXYZ
        point3D = repeat(point3D, "B RN SN DimX -> B NV RN SN DimX", NV=NV).float()
        point3D = torch.cat([point3D, torch.ones_like(point3D[:,:,:,:,:1])], axis=4)
        
        # B NV 4 4 -> (B NV) 4 4
        if points_projected is None or mask_valid is None:
            points_in_pixel = torch.bmm(rearrange(batch['source_poses'], "B NV M_1 M_2 -> (B NV) M_1 M_2", M_1=4, M_2=4), 
                                    rearrange(point3D, "B NV RN SN DimX -> (B NV) DimX (RN SN)")) 
            
            points_in_pixel = rearrange(points_in_pixel, "(B NV) DimX (RN SN) -> B NV DimX RN SN", B=B, RN=RN)
            points_in_pixel = points_in_pixel[:,:,:3]
            # in 2D pixel coordinate
            mask_valid_depth = points_in_pixel[:,:,2]>0  #B NV RN SN
            mask_valid_depth = mask_valid_depth.float()
            points_in_pixel = points_in_pixel[:,:,:2] / points_in_pixel[:,:,2:3]
        else:
            points_in_pixel = points_projected.permute(0, 1, 4, 2, 3)
            mask_valid_depth = mask_valid
        
        img_feat_sampled, mask = grid_sample_2d(rearrange(source_imgs_feat, "B NV C H W -> (B NV) C H W"), 
                                rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2")) #* shape: (B*NV, C, RN, SN)
        img_rgb_sampled, _ = grid_sample_2d(rearrange(batch['source_imgs'], "B NV C H W -> (B NV) C H W"), 
                                rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2")) #*shape: (B*NV, C, RN, SN)
        
        #TODO: grid sample 2d depth
        depth_info=None
        if self.args.mvs_depth_guide > 0 and self.args.depth_pos_encoding:
            if self.args.volume_type == "correlation":
                
                depths_mvs = batch['depth_info'][:, 0:].float() #* B NV H W
            else:
                depths_mvs = batch['depths_mvs_h'][:, s_idx:].float() #* B NV H W
                
            ref_depth_sampled, _ = grid_sample_2d(rearrange(depths_mvs, "B NV H W -> (B NV) 1 H W"), 
                                    rearrange(points_in_pixel, "B NV Dim2 RN SN -> (B NV) RN SN Dim2")) #* shape: (B*NV, 1 RN, SN)
            ref_depth_sampled = rearrange(ref_depth_sampled, '(B NV) 1 RN SN -> (B NV) 1 (RN SN)', B=B, NV=NV)
            #* rotation
            point3D_cam = torch.bmm(rearrange(batch['w2cs'][:, s_idx:, :3, :3], "B NV M_1 M_2 -> (B NV) M_1 M_2", M_1=3, M_2=3),
                                    rearrange(point3D[..., :-1], "B NV RN SN DimX -> (B NV) DimX (RN SN)"))
            #* translation
            point3D_cam = point3D_cam + rearrange(batch['w2cs'][:, s_idx:, :3, -1], "B NV DimX -> (B NV) DimX 1") #* (B NV) 3 (RN SN)
            
            depth_dist = ref_depth_sampled - point3D_cam[:,-1:, :] #* (B NV) 1 (RN SN)
            depth_dist = rearrange(depth_dist, "B_NV 1 RN_SN -> B_NV RN_SN 1")
            depth_info = self.depthcode(depth_dist) #* (B NV) (RN SN) L
        
        
        #! mask out invalid 3d points (depth < 0) or out of image (x < 0 or x > W or y < 0 or y > H)
        mask = rearrange(mask, "(B NV) RN SN -> B NV RN SN", B=B)
        mask = mask * mask_valid_depth
        img_feat_sampled = rearrange(img_feat_sampled, "(B NV) C RN SN -> B NV C RN SN", B=B)
        img_rgb_sampled = rearrange(img_rgb_sampled, "(B NV) C RN SN -> B NV C RN SN", B=B)

        # --------- run transformer to aggregate information
        # -- 1. view transformer
        x = rearrange(img_feat_sampled, "B NV C RN SN -> (B RN SN) NV C")
        
        if self.args.volume_reso > 0:
            x_fea_volume_feat = repeat(fea_volume_feat, "B_RN_SN C -> B_RN_SN NV C", NV=NV)
            x = torch.cat([x, x_fea_volume_feat], axis=-1) #* 32 + 24 = 56
        
        if cond_info is not None:
            # sim_feats = torch.cat([cond_info['feat_info'], cond_info['color_info'], cond_info['mask_info']], dim=-1) #* 10, 9, 3 = 22
            sim_feats = cond_info['feat_info'] #* use only this because it is invariant to the number of views
            
            sim_feats = self.pre_sim_mlp(sim_feats)
            sim_feats = repeat(sim_feats, "B RN SN C -> B RN SN NV C", NV=NV)
            sim_feats = rearrange(sim_feats, "B RN SN NV C -> (B RN SN) NV C")
            x = torch.cat([x, sim_feats], axis=-1) #* 56 + 16 = 72
        
        if depth_info is not None:
            depth_info = rearrange(depth_info, "(B NV) (RN SN) L -> (B RN SN) NV L", B=B, NV=NV, RN=RN, SN=SN)
            x = torch.cat([x, depth_info], axis=-1) #* 72 + 8 = 80
        
        if self.args.use_dir_srdf:
            dir_relative = rearrange(dir_relative, "B NV RN SN Dim3 -> B RN SN NV Dim3")
            dir_relative = rearrange(dir_relative, "B RN SN NV Dim3 -> (B RN SN) NV Dim3", B=B)
            
            dir_relative = self.dircode(dir_relative)
            x = torch.cat([x, dir_relative], axis=-1)
            
        
        # add additional view aggregation token
        view_token = self.viewToken(x)
        view_token = rearrange(view_token, "B_RN_SN C -> B_RN_SN 1 C")
        x = torch.cat([view_token, x], axis=1)
        
        x = self.density_view_transformer(x)

        x1 = rearrange(x, "B_RN_SN NV C -> NV B_RN_SN C")
        x = x1[0] #reference
        view_feature = x1[1:]

        if self.only_volume:
            x = rearrange(x_fea_volume_feat, "(B RN SN) NV C -> NV (B RN) SN C", B=B, RN=RN, SN=SN)[0]
        else:
            # -- 2. ray transformer
            # add positional encoding
            x = rearrange(x, "(B RN SN) C -> (B RN) SN C", RN=RN, B=B, SN=SN)
            x = torch.cat([x, repeat(self.order_posenc(d_hid=self.PE_d_hid, n_samples=SN).type_as(x), 
                                        "SN C -> B_RN SN C", B_RN = B*RN)], axis=2)

            x = self.density_ray_transformer(x)
        
        srdf = self.DensityMLP(x)

        # calculate weight using view transformers result
        view_feature = rearrange(view_feature, "NV (B RN SN) C -> B RN SN NV C", B=B, RN=RN, SN=SN)
        dir_relative = rearrange(dir_relative, "B NV RN SN Dim3 -> B RN SN NV Dim3")

        x_weight = torch.cat([view_feature, dir_relative], axis=-1)
        x_weight = self.linear_radianceweight_1_softmax(x_weight)
        mask = rearrange(mask, "B NV RN SN -> B RN SN NV 1")
        x_weight[mask==0] = -1e9
        weight = self.softmax(x_weight)
        
        radiance = (img_rgb_sampled * rearrange(weight, "B RN SN L 1 -> B L 1 RN SN", B=B, RN=RN)).sum(axis=1)
        radiance = rearrange(radiance, "B DimRGB RN SN -> (B RN SN) DimRGB")
        
        return radiance, srdf, points_in_pixel  


class ViewTokenNetwork(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.register_parameter('view_token', nn.Parameter(torch.randn([1,dim])))

    def forward(self, x):
        return torch.ones([len(x), 1]).type_as(x) * self.view_token
