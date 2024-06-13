import os, piq
from re import I
from stat import UF_OPAQUE

import numpy as np
import torch
import torch.nn.functional as F
from torch import optim

from PIL import Image
from tqdm import tqdm
import pytorch_lightning as pl
import cv2
import mcubes
from einops import (rearrange, reduce, repeat)

from .encoder_utils.sampler import FixedSampler, ImportanceSampler
from .encoder_utils.fmt.TransMVSNet import TransMVSNet
from .encoder_utils.single_variance_network import SingleVarianceNetwork
from .encoder_utils.renderer import VolumeRenderer

from .ray_transformer import RayTransformer
from .feature_volume import FeatureVolume, MVSVolume
from .utils.gmflow_utils import sample_features_by_grid
from .misc import camera


class UFORecon(pl.LightningModule):
    def __init__(self, args):
        super().__init__()

        self.args = args
        self.patch_size = args.patch_size
        self.sW = args.sW
        self.sH = args.sH 
               
        self.train_ray_num = args.train_ray_num

        if self.args.extract_geometry: # testing
            self.point_num = args.test_sample_coarse
            self.point_num_2 = args.test_sample_fine
        else:
            self.point_num = args.coarse_sample
            self.point_num_2 = args.fine_sample
        
        self.transmvsnet = TransMVSNet(refine=False, ndepths=[int(nd) for nd in args.ndepths.split(",") if nd],
                        depth_interals_ratio=[float(d_i) for d_i in args.depth_inter_r.split(",") if d_i],
                        share_cr=args.share_cr,
                        cr_base_chs=[int(ch) for ch in args.cr_base_chs.split(",") if ch],
                        grad_method=args.grad_method)
        
        self.fixed_sampler = FixedSampler(point_num = self.point_num)
        self.importance_sampler = ImportanceSampler(point_num = self.point_num_2)
        self.deviation_network = SingleVarianceNetwork(0.3) # add variance network

        self.ref_ = None
        
        self.renderer = VolumeRenderer(args)
        self.ray_transformer = RayTransformer(args = args) 

        if self.args.volume_type=="featuregrid" and self.args.volume_reso>0:
            self.feature_volume = FeatureVolume(self.args.volume_reso)
        if self.args.volume_type=="correlation" and self.args.volume_reso>0:
            self.feature_volume = MVSVolume(in_channels=1, base_channels=8)

        self.pos_encoding = self.order_posenc(d_hid=16, n_samples=self.point_num)
        self.pos_encoding_2 = self.order_posenc(d_hid=16, n_samples=self.point_num + self.point_num_2)
        
        # self.pre_conv = None
        self.pre_conv = torch.nn.Conv2d(in_channels=128, out_channels=32, kernel_size=1, stride=1, padding=0, bias=False)

    def configure_optimizers(self):
        transmvsnet_params = []
        uforecon_params = []
        for name, param in self.named_parameters():
            if "transmvsnet" in name:
                transmvsnet_params.append(param)
            else:
                uforecon_params.append(param)
        
        # freeze MVS if needed
        for param in transmvsnet_params:
            param.requires_grad = False
            
                
        total_optimizer = optim.Adam(uforecon_params, lr=self.args.uforecon_lr)
        return total_optimizer
    

    def order_posenc(self, d_hid, n_samples):
        """
        positional encoding of the sample ordering on a ray
        """

        def get_position_angle_vec(position):
            return [position / np.power(10000, 2 * (hid_j // 2) / d_hid) for hid_j in range(d_hid)]

        sinusoid_table = np.array([get_position_angle_vec(pos_i) for pos_i in range(n_samples)])
        sinusoid_table[:, 0::2] = np.sin(sinusoid_table[:, 0::2])  # dim 2i
        sinusoid_table[:, 1::2] = np.cos(sinusoid_table[:, 1::2])  # dim 2i+1
        sinusoid_table = torch.from_numpy(sinusoid_table)
        
        return sinusoid_table

    def visualize_depth(self, depth, mask=None, depth_min=None, depth_max=None, direct=False):
        """Visualize the depth map with colormap.
        Rescales the values so that depth_min and depth_max map to 0 and 1,
        respectively.
        """
        if not direct:
            depth = 1.0 / (depth + 1e-6)
        invalid_mask = np.logical_or(np.isnan(depth), np.logical_not(np.isfinite(depth)))
        if mask is not None:
            invalid_mask += np.logical_not(mask)
        if depth_min is None:
            depth_min = np.percentile(depth[np.logical_not(invalid_mask)], 5)
        if depth_max is None:
            depth_max = np.percentile(depth[np.logical_not(invalid_mask)], 95)
        depth[depth < depth_min] = depth_min
        depth[depth > depth_max] = depth_max
        depth[invalid_mask] = depth_max

        depth_scaled = (depth - depth_min) / (depth_max - depth_min)
        depth_scaled_uint8 = np.uint8(depth_scaled * 255)
        depth_color = cv2.applyColorMap(depth_scaled_uint8, cv2.COLORMAP_MAGMA)
        depth_color[invalid_mask, :] = 0

        os.makedirs("./depth_images_test", exist_ok=True)
        cv2.imwrite(os.path.join("./depth_images_test", "{}".format('depth_est.png')), depth_color)
        
        return depth_color

    def build_feature_volume(self, batch, source_imgs_feat):
        return self.feature_volume(source_imgs_feat, batch)
    
    def build_mvs_volume(self, batch, feature_volume):
        return self.feature_volume(batch, feature_volume)
    
    def build_pairs(self, imgs, proj_mats, depth_values):
        
        #! imgs: B, N, 3, H, W
        N = imgs.shape[1]
        # if N == 3: [0, 1, 2], [1, 2, 0], [2, 0, 1]
        # if N == 4: [0, 1, 2, 3], [1, 2, 3, 0], [2, 3, 0, 1], [3, 0, 1, 2] 
        nums = [i for i in range(N)]
        all_combinations = []
        for i in range(N):
            all_combinations.append(nums[i:] + nums[:i])
        all_combinations = np.array(all_combinations)
        
        #indexing from images according to all_combinations using gather
        imgs = imgs[:, all_combinations]
        
        for stage in ['stage1', 'stage2', 'stage3']:
            proj_mats[stage] = rearrange(proj_mats[stage][:,all_combinations], "B N V dim2 dim4 dim4_2 -> (B N) V dim2 dim4 dim4_2", B=imgs.shape[0], N=N,V=N)
        
        imgs = rearrange(imgs, "B N V C H W -> (B N) V C H W", N=N)
        
        depth_values = depth_values.expand(imgs.shape[0], -1)
        return imgs, proj_mats, depth_values

    def get_vit_feature(self, x):
        mean = torch.tensor([0.485, 0.456, 0.406],
                            device=x.device).reshape(1, 3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225],
                           device=x.device).reshape(1, 3, 1, 1)
        x = F.interpolate(x, size=(224, 224))
        x = (x - mean) / std
        return self.ext.get_feature_from_input(x)[-1][0, 0, :]
    
    def get_match_feat(self, imgs, attn_splits_list=None, cur_n_src_views=3):
        '''this code is heavily from MatchNeRF [arxiv'23]'''
        if attn_splits_list is None:
            attn_splits_list = self.opts.encoder.attn_splits_list #* [2]
        img1s = imgs[:, :cur_n_src_views]

        out_dict = self.match_enc(imgs=img1s, attn_splits_list=attn_splits_list, keep_raw_feats=True,
                                 wo_self_attn=False) #* GMFlow output 
        '''
        out_dict consists of two parts: 
        index_lists = [(a, b) for a in range(n_views - 1) for b in range(a + 1, n_views)] 
        
        1. aug_feat0s: list of features from the first images 
        2. aug_feat1s: list of features from the second imagrayes 
        '''
        # split the output
        img_feat_list = []
        img_sum_feat_list = []
        index_lists = [(a, b) for a in range(cur_n_src_views - 1) for b in range(a + 1, cur_n_src_views)]
        for scale_idx in range(len(out_dict['aug_feat0s'])):
            img_feat = [[] for _ in range(cur_n_src_views)] #* len: src_views
            
            img1s_feats = out_dict['aug_feat0s'][scale_idx]
            img2s_feats = out_dict['aug_feat1s'][scale_idx]
            for feat_i, (i_idx, j_idx) in enumerate(index_lists):
                img_feat[i_idx].append(img1s_feats[:, feat_i])
                img_feat[j_idx].append(img2s_feats[:, feat_i])
                
            # post-process the output
            img_sum_feat = [[] for _ in range(cur_n_src_views)] #* 
            for k, v in enumerate(img_feat): #* N x (N-1) 2-dimension list, N: number of combinations
                img_feat[k] = torch.cat(v, dim=1) #! before
                #! after
                unsqueezed_v = [f.unsqueeze(1) for f in v]
                unsqueezed_v = torch.cat(unsqueezed_v, dim=1)
                sum_feat = torch.sum(unsqueezed_v, dim=1, keepdim=False) / (float(unsqueezed_v.shape[1]) + 1e-10)  #B 1 X Y Z C
                # var_feat = torch.sum((unsqueezed_v - mean_feat)**2, dim=1, keepdim=True)  # B 1 X Y Z C
                img_sum_feat[k] = sum_feat
            
            img_feat = torch.stack(img_feat, dim=1)  # BxVxCxHxW
            img_feat_list.append(img_feat)

            img_sum_feat = torch.stack(img_sum_feat, dim=1)  # BxVxCxHxW
            img_sum_feat_list.append(img_sum_feat)
            
        return img_feat_list, img_sum_feat_list
    
    def query_cond_info(self, point_samples, src_poses, src_images, src_feats_list, extract_similarity=False):
        '''
            this code is heavily borrowed from MatchNeRF [arxiv'23]
            query conditional information from source images, using the reference position.
            point_samples: B, n_rays, n_samples, 3
            src_poses: dict, all camera information of source images 
                        'extrinsics': B, N, 3, 4; 'intrinsics': B, N, 3, 3; 'near_fars': B, N, 2
            src_images: B, n_views, 3, H, W. range: [0, 1] !!!
        '''
        batch_size, n_views, _, img_h, img_w = src_images.shape
        assert src_feats_list is not None, "Must provide the image feature for info query."

        cos_n_group = [8] #! opt
        cos_n_group = [cos_n_group] if isinstance(cos_n_group, int) else cos_n_group
        feat_data_list = [[] for _ in range(len(src_feats_list))]
        color_data = []
        mask_data = []

        # query information from each source view
        point_samples_pixel, _, mask_valid = camera.get_coord_ref_ndc(src_poses, point_samples, extract_similarity=extract_similarity)
        
        if not self.args.explicit_similarity:
            return point_samples_pixel, mask_valid
        
        for view_idx in range(n_views):
            grid = point_samples_pixel[:,view_idx, ...] #* (B, n_rays, n_samples, 2) or (B, xyz, 2)

            if extract_similarity:
                grid = grid.unsqueeze(1) #* (B, 1, xyz, 2)
            
            # query enhanced features infomation from each view
            for scale_idx, img_feat_cur_scale in enumerate(src_feats_list):
                raw_whole_feats = img_feat_cur_scale[:, view_idx]
                sampled_feats = sample_features_by_grid(raw_whole_feats, grid, align_corners=True, mode='bilinear', padding_mode='border',
                                                        local_radius=0, 
                                                        local_dilation=1) #! shape: B C RN SN
                feat_data_list[scale_idx].append(sampled_feats)
            
            # query color
            color_data.append(F.grid_sample(src_images[:, view_idx], grid, align_corners=True, mode='bilinear', padding_mode='border'))

            # record visibility mask for further usage
            in_mask = ((grid > -1.0) * (grid < 1.0))
            in_mask = (in_mask[..., 0] * in_mask[..., 1]).float()
            mask_data.append(in_mask.unsqueeze(1))

        # merge queried information from all views
        all_data = {}
        # merge extracted enhanced features
        merged_feat_data = []
        for feat_data_idx, raw_feat_data in enumerate(feat_data_list): # loop over scale
            cur_updated_feat_data = []
            # split features from each view
            split_feat_data = [torch.split(x, int(x.shape[1] / (n_views - 1)), dim=1) for x in raw_feat_data] #* 4x3 list
            # calculate simliarity for feature from the same transformer
            index_lists = [(a, b) for a in range(n_views - 1) for b in range(a, n_views - 1)]
            for i_idx, j_idx in index_lists:
                input_a = split_feat_data[i_idx][j_idx]  # B x C x N_rays x N_pts
                input_b = split_feat_data[j_idx + 1][i_idx]
                iB, iC, iR, iP = input_a.shape
                group_a = input_a.reshape(iB, cos_n_group[feat_data_idx], int(iC / cos_n_group[feat_data_idx]), iR, iP)
                group_b = input_b.reshape(iB, cos_n_group[feat_data_idx], int(iC / cos_n_group[feat_data_idx]), iR, iP)
                cur_updated_feat_data.append(torch.nn.CosineSimilarity(dim=2)(group_a, group_b))
            cur_updated_feat_data = torch.stack(cur_updated_feat_data, dim=1)  # [B, n_pairs, n_groups, n_rays, n_pts]
            
            cur_updated_feat_data = torch.mean(cur_updated_feat_data, dim=1, keepdim=True)
            cur_updated_feat_data = cur_updated_feat_data.reshape(cur_updated_feat_data.shape[0], -1, *cur_updated_feat_data.shape[-2:])
            merged_feat_data.append(cur_updated_feat_data)
            
        merged_feat_data = torch.cat(merged_feat_data, dim=1)
        # all_data.append(merged_feat_data)
        all_data['feat_info'] = merged_feat_data

        # merge extracted color data
        merged_color_data = torch.cat(color_data, dim=1)
        # all_data.append(merged_color_data)
        all_data['color_info'] = merged_color_data

        # merge visibility masks
        merged_mask_data = torch.cat(mask_data, dim=1)
        # all_data.append(merged_mask_data)
        all_data['mask_info'] = merged_mask_data

        # all_data = torch.cat(all_data, dim=1)[0].permute(1, 2, 0)
        for k, v in all_data.items():
            all_data[k] = v.permute(0, 2, 3, 1)  # (b, n_rays, n_samples, n_dim)

        return all_data, point_samples_pixel, mask_valid
    
    
    def sample2rgb(self, batch, points_x, z_val, ray_d, ray_idx, source_imgs_feat, feature_volume, match_feature):
        B, L, _, imgH, imgW = batch['source_imgs'].shape
        RN = ray_idx.shape[1]
        _, _, SN, _ = points_x.shape
        
        s_idx = batch['start_idx'] if 'start_idx' in batch.keys() else 1 #* start_idx should be zero in inference and one in training
    
        #* query features from encoder
        src_extr = batch['w2cs'][:, s_idx:, :3, :]
        src_intr = batch['intrinsics'][:, s_idx:, :, :]
        src_near_fars = batch['near_fars'][:, s_idx:, :]
        src_poses = {'extrinsics': src_extr, 'intrinsics': src_intr, 'near_fars': src_near_fars}
        
        #* query conditional information (feature similarity) from source images
        cond_info = None
        if self.args.explicit_similarity:
            cond_info, point_samples_pixel, mask_valid = self.query_cond_info(points_x, batch['source_poses'], batch['source_imgs'], match_feature)
        
        #TODO: query depth information from transmvsnet cost volume
        if self.args.volume_type == "correlation":
            volume_info = self.query_depth_from_volume(points_x, batch['source_poses'], feature_volume, near_far=batch['near_fars'][0][0], stages=["stage1", "stage2", "stage3"]) #! ablation
            feature_volume = volume_info
        
        #* Ray Transformer
        radiance, srdf, points_in_pixel = self.ray_transformer(points_x, batch, source_imgs_feat, feature_volume, 
                                                               cond_info=cond_info, points_projected=point_samples_pixel, mask_valid=mask_valid)

        ray_d = repeat(ray_d, "RN Dim3 -> RN SN Dim3", SN=SN)
        
        #* NeuS Renderer
        rgb, depth, opacity, weight, variance = self.renderer.render(rearrange(z_val, "B RN SN -> (B RN) SN"),  
                                        rearrange(radiance, "(B RN SN) C -> (B RN) SN C", B=B, RN=RN),
                                        srdf.squeeze(dim=2),
                                        deviation_network=self.deviation_network)

        rgb = rearrange(rgb, "(B RN) C -> B RN C", B=B).float()
        depth = rearrange(depth, "(B RN) -> B RN", B=B)
        opacity = rearrange(opacity, "(B RN) -> B RN", B=B)
        weight = rearrange(weight, "(B RN) SN -> B RN SN", B=B)

        return rgb, depth, srdf, opacity, weight, points_in_pixel, variance

    def query_depth_from_volume(self, point_samples, poses, feature_volume, near_far=None, stages=["stage1", "stage2", "stage3"]):
        '''
            query geometry information from source images, using the reference position.
            point_samples: B, n_rays, n_samples, 3
            ref_pose: dict, all camera information of source images 
                        'extrinsics': B, N, 3, 4; 'intrinsics': B, N, 3, 3; 'near_fars': B, N, 2
            src_images: B, n_views, 3, H, W. range: [0, 1] !!!
        '''
        N = poses.size(1)
        for n in range(N):
            _, points_samples_pixel, mask_valid = camera.get_coord_ref_ndc(poses[:, n:n+1, ...], point_samples, near_far=near_far, pad=True)
            
            features_list = []
            for stage in stages:
                feat_vol = feature_volume[stage]['feature_volume'][n:n+1]
                weight_vol = feature_volume[stage]['weight_volume'][n:n+1]
                
                H, W = points_samples_pixel.shape[-3:-1]
                # grid = points_samples_pixel.view(-1, 1, H,  W, 3) * 2 - 1.0  # [1 1 H W 3] (x,y,z)#TODO check
                grid = points_samples_pixel.view(-1, 1, H, W, 3)  # [1 1 H W 3] (x,y,z)#TODO check
                features = F.grid_sample(feat_vol, grid, mode='bilinear', align_corners=True, padding_mode='zeros')[:,:,0].permute(2,3,0,1).squeeze()
                weights = F.grid_sample(weight_vol, grid, mode='bilinear', align_corners=True, padding_mode='zeros')[:,:,0].permute(2,3,0,1).squeeze()
                features = features.reshape(-1, features.shape[-1])
                weights = weights.reshape(-1, 1)
                features_list.append(features)
                if stage == "stage1":
                    weights_L = weights
                else:
                    weights_L = weights_L + weights
            
            features_L = torch.cat(features_list, dim=-1)
            if n == 0:
                G_all = features_L * weights_L
                W_all = weights_L
            else:
                G_all = G_all + features_L * weights_L
                W_all = W_all + weights_L
        
        G_all = G_all / (W_all + 1e-8)
        
        return rearrange(G_all, "(B RN SN) Dim -> B RN SN Dim", B=feat_vol.shape[0], RN=H, SN=W)

    
    def infer(self, batch, ray_idx, source_imgs_feat, feature_volume=None, extract_geometry=False, match_feature=None, ray_idx_all=None, is_train=True):

        B, L, _, imgH, imgW = batch['source_imgs'].shape
        RN = ray_idx.shape[1]
        
        if not extract_geometry:
            # gt rgb for rays
            ref_img = rearrange(batch['ref_img'], "B DimRGB H W -> B DimRGB (H W)")
            rgb_gt = torch.gather(ref_img, 2, repeat(ray_idx, "B RN -> B DimRGB RN", DimRGB=3))
            rgb_gt = rearrange(rgb_gt, "B C RN -> B RN C")

            # gt depth for rays
            ref_depth = rearrange(batch['depths_h'][:,0], "B H W -> B (H W)") # use only depth of reference view 
            depth_gt = torch.gather(ref_depth, 1, ray_idx)
        
        # ---------------------- ray sampling ----------------------    
        ray_d = torch.gather(batch['ray_d'], 2, repeat(ray_idx, "B RN -> B DimX RN", DimX=3))
        ray_d = rearrange(ray_d, "B DimX RN -> (B RN) DimX")
        ray_o = repeat(batch['ray_o'], "B DimX -> B DimX RN", RN = RN) 
        ray_o = rearrange(ray_o, "B DimX RN -> (B RN) DimX")

        #TODO ---------------------- coarse sampling along the ray ---------------------
        if 'near_fars' in batch.keys():
            near_z = batch['near_fars'][:,0,0]
            near_z = repeat(near_z, "B -> B RN", RN=RN)
            near_z = rearrange(near_z, "B RN -> (B RN)")
            far_z = batch['near_fars'][:,0,1]
            far_z = repeat(far_z, "B -> B RN", RN=RN)
            far_z = rearrange(far_z, "B RN -> (B RN)")
        
            if extract_geometry:
                camera_ray_d = torch.gather(batch['cam_ray_d'], 2, repeat(ray_idx, "B RN -> B DimX RN", DimX=3))
                camera_ray_d = rearrange(camera_ray_d, "B DimX RN -> (B RN) DimX")
                near_z = near_z / camera_ray_d[:,2]
                far_z = far_z / camera_ray_d[:,2]
            
            points_x, z_val, points_d = self.fixed_sampler.sample_ray(ray_o, ray_d, near_z=near_z, far_z=far_z)

        else:
            points_x, z_val, points_d = self.fixed_sampler.sample_ray(ray_o, ray_d)

        # SN is sample point number along the ray
        points_x = rearrange(points_x, "(B RN) SN DimX -> B RN SN DimX", B = B) 
        points_d = rearrange(points_d, "(B RN) SN DimX -> B RN SN DimX", B = B)

        points_x = points_x.float()
        points_d = points_d.float()
        
        z_val = rearrange(z_val, "(B RN) SN -> B RN SN", B = B)

        
        #* sample2rgb
        rgb, depth, srdf, opacity, weight, points_in_pixel, _ = self.sample2rgb(batch, points_x, z_val, ray_d, ray_idx, 
                    source_imgs_feat, feature_volume=feature_volume, match_feature=match_feature)
        
        
        if extract_geometry and self.args.test_coarse_only:
            srdf = rearrange(srdf, "(B RN) SN Dim1 ->B RN SN Dim1", B=B)
            srdf = srdf.squeeze(-1)
            return srdf, points_x, depth, rgb

        # ---------------------- fine sampling along the ray ----------------------
        points_x_2, z_val_2, points_d_2 = self.importance_sampler.sample_ray(ray_o, ray_d, 
                                                        rearrange(weight, "B RN SN -> (B RN) SN", B=B).detach(), 
                                                        rearrange(z_val, "B RN SN -> (B RN) SN").detach())
        points_x_2 = points_x_2.float()
        points_d_2 = points_d_2.float()
        
        # SN is sample point number along the ray
        points_x_2 = rearrange(points_x_2, "(B RN) SN DimX -> B RN SN DimX", B = B)
        points_d_2 = rearrange(points_d_2, "(B RN) SN DimX -> B RN SN DimX", B = B)
        z_val_2 = rearrange(z_val_2, "(B RN) SN -> B RN SN", B = B)

        points_x_all = torch.cat([points_x, points_x_2], axis=2)
        z_val_all = torch.cat([z_val, z_val_2], axis=2)
        sample_sort_idx = torch.sort(z_val_all,axis=2)[1]
        z_val_all = torch.gather(z_val_all, 2, sample_sort_idx)
        points_x_all = torch.gather(points_x_all, 2, repeat(sample_sort_idx, "B RN SN -> B RN SN 3"))

        rgb_2, depth_2, srdf_2, opacity_2, weight_2, points_in_pixel_2, variance = self.sample2rgb(batch, 
        points_x_all, z_val_all, ray_d, ray_idx, source_imgs_feat, feature_volume=feature_volume, match_feature=match_feature)

        if extract_geometry:
            srdf_2 = rearrange(srdf_2, "(B RN) SN Dim1 ->B RN SN Dim1", B=B)
            srdf_2 = srdf_2.squeeze(-1)
            return srdf_2, points_x_all, depth_2, rgb_2

        return rgb_gt, rgb, depth, depth_gt, srdf, opacity, weight, points_in_pixel,\
            rgb_2, depth_2, srdf_2, opacity_2, weight_2, points_in_pixel_2,\
            z_val, z_val_all, variance

    def ext_fpn_feats(self, batch, source_imgs):
        B, L, _, imgH, imgW = source_imgs.shape
        source_imgs = rearrange(source_imgs, "B L C H W -> (B L) C H W")
        source_imgs_feat = self.fpn_extractor(source_imgs)
        source_imgs_feat = rearrange(source_imgs_feat, "(B L) C H W -> B L C H W", L=L) #* (B, 4, 32, 128, 160)

        return source_imgs_feat

    def training_step(self, batch, batch_idx):
        
        #! --------------------------------------------from here --------------------------------------------
        B, L, _, imgH, imgW = batch['source_imgs'].shape
        stages = ["stage1"]
        
        # ---------------------- step 0: infer image features ----------------------
        proj_matrices, near_fars, depth_values_org = batch['proj_matrices'], batch['near_fars'], batch['depth_values_org_scale']
        source_imgs = batch['source_imgs']            
        all_imgs_pair, all_proj_mats_pair, depth_values_org_pair = self.build_pairs(source_imgs, proj_matrices, depth_values_org)
        source_imgs_feat, volume_info = self.transmvsnet(all_imgs_pair, all_proj_mats_pair, depth_values_org_pair)
        for i in range(len(source_imgs_feat)):
            source_imgs_feat[i]['stage1'] = source_imgs_feat[i]['stage1'][0:1]
        src_match_feats_list = self.transmvsnet.get_match_feat(source_imgs_feat, cur_n_src_views=self.args.train_n_view-1)
        
        del all_imgs_pair, all_proj_mats_pair, depth_values_org_pair
        
        source_imgs_feat_dict = {}
        for stage in stages:
            source_imgs_feat_dict[stage] = torch.stack([feat[stage] for feat in source_imgs_feat], dim=1) #* (B V C H W)
        source_imgs_feat = source_imgs_feat_dict["stage1"]
        
        if self.args.volume_type =="featuregrid" and self.args.volume_reso > 0:
            feature_volume = self.build_feature_volume(batch, source_imgs_feat) #* (B, 4, 32, 128, 160)
        elif self.args.volume_type == "correlation":
            feature_volume = {"stage1": volume_info['stage1']['cost_volume'], "stage2": volume_info['stage2']['cost_volume'], "stage3": volume_info['stage3']['cost_volume']}
            volume_feat, volume_weight = self.build_mvs_volume(batch, feature_volume['stage1']) #* (B, 1, D, H, W)
            volume_feat_2, volume_weight_2 = self.build_mvs_volume(batch, feature_volume['stage2'])
            volume_feat_3, volume_weight_3 = self.build_mvs_volume(batch, feature_volume['stage3'])
            
            feature_volume = { "stage1": {"feature_volume": volume_feat, "weight_volume": volume_weight}, 
                               "stage2": {"feature_volume": volume_feat_2, "weight_volume": volume_weight_2},
                               "stage3": {"feature_volume": volume_feat_3, "weight_volume": volume_weight_3}}

        else:
            raise NotImplementedError
        
        if self.args.mvs_depth_guide > 0:
            depth_info = volume_info['stage3']['depth'] * batch['scale_factor']
            batch['depth_info'] = depth_info.unsqueeze(0)
        
        if volume_info is not None:
            del volume_info 
        
        # ---------------------- step 1: sample rays --------------------------------
        ray_idx = torch.argsort(torch.rand(B, imgH * imgW).type_as(batch['ray_o']), dim=-1)[:,:self.train_ray_num] #* random sampling
        ray_idx_all = repeat(torch.arange(imgH * imgW), "HW -> B HW", B = B).type_as(batch['ray_o']).long() 

        rgb_gt, rgb, depth, depth_gt, srdf, opacity, weight, points_in_pixel, \
            rgb_2, depth_2, srdf_2, opacity_2, weight_2, points_in_pixel_2, \
            z_val, z_val_all, variance = self.infer(batch=batch, 
                                                ray_idx=ray_idx, 
                                                ray_idx_all = ray_idx_all,
                                                source_imgs_feat=source_imgs_feat,
                                                feature_volume=feature_volume,
                                                match_feature=src_match_feats_list,
                                                )
        B, _ = depth_gt.size()

        # color loss
        loss_rgb = torch.nn.functional.mse_loss(rgb, rgb_gt)
        loss_rgb2 = torch.nn.functional.mse_loss(rgb_2, rgb_gt)

        # Depth loss
        mask_depth = (depth_gt!=0) & (depth_gt>=batch['near_fars'][:,0,0:1]) & (depth_gt<=batch['near_fars'][:,0,1:2])
        mask_depth_imgsize = (batch['depths_h'][:,0] != 0) & (batch['depths_h'][:,0] >= batch['near_fars'][:,0,0:1][0][0]) & (batch['depths_h'][:,0] <= batch['near_fars'][:,0,1:2][0][0])
        if torch.sum(mask_depth)>0:
            # masked out where gt depth is invalid
            loss_depth_ray =  torch.nn.functional.l1_loss(depth[mask_depth], depth_gt[mask_depth]) 
            loss_depth_ray2 = torch.nn.functional.l1_loss(depth_2[mask_depth], depth_gt[mask_depth])
        else:
            loss_depth_ray = loss_depth_ray2 = 0.0
        
        loss = self.args.weight_rgb * (loss_rgb + loss_rgb2) + \
            self.args.weight_depth * (loss_depth_ray + loss_depth_ray2)
            
        self.log("train/depth_ray_coarse", loss_depth_ray)
        self.log("train/depth_ray_fine", loss_depth_ray2)
        self.log("train/rgb_coarse", loss_rgb)
        self.log("train/rgb_fine", loss_rgb2)
        self.log("train/loss_all", loss)
        self.log("train/variance", variance)

        return loss


    def validation_epoch_end(self, batch_parts):
        # average epoches
        psnr_coarse = [i['psnr/coarse'] for i in batch_parts]
        psnr_fine = [i['psnr/fine'] for i in batch_parts]
        loss_rgb_coarse = [i['val/loss_rgb_coarse'] for i in batch_parts]
        loss_rgb_fine = [i['val/loss_rgb_fine'] for i in batch_parts]
        loss_depth_coarse = [i['val/loss_depth_coarse'] for i in batch_parts]
        loss_depth_fine = [i['val/loss_depth_fine'] for i in batch_parts]

        psnr_coarse = sum(psnr_coarse) / len(psnr_coarse)
        psnr_fine = sum(psnr_fine) / len(psnr_fine)
        loss_rgb_coarse = sum(loss_rgb_coarse) / len(loss_rgb_coarse)
        loss_rgb_fine = sum(loss_rgb_fine) / len(loss_rgb_fine)
        loss_depth_coarse = sum(loss_depth_coarse) / len(loss_depth_coarse)
        loss_depth_fine = sum(loss_depth_fine) / len(loss_depth_fine)
        
        # logging
        self.log("psnr/coarse", psnr_coarse, sync_dist=True)
        self.log("psnr/fine", psnr_fine, sync_dist=True)
        self.log("val/rgb_coarse", loss_rgb_coarse, sync_dist=True)
        self.log("val/rgb_fine", loss_rgb_fine, sync_dist=True)
        self.log("val/loss_depth_coarse", loss_depth_coarse, sync_dist=True)
        self.log("val/loss_depth_fine", loss_depth_fine, sync_dist=True)

        loss = loss_rgb_coarse + loss_rgb_fine

        return loss


    def validation_step(self, batch, batch_idx):
        if self.args.extract_geometry:
            self.extract_geometry(batch, batch_idx)
            # return dummy data
            return {"val/loss_rgb_coarse":0,
                    "val/loss_rgb_fine":0,
                    "val/loss_depth_coarse":0,
                    "val/loss_depth_fine":0,
                    "psnr/coarse":0,
                    "psnr/fine":0}
            
        # if self.args.extract_similarity:
        #     self.extract_similarity(batch, batch_idx)
        #     # return dummy data
        #     return {"val/loss_rgb_coarse":0,
        #             "val/loss_rgb_fine":0,
        #             "val/loss_depth_coarse":0,
        #             "val/loss_depth_fine":0,
        #             "psnr/coarse":0,
        #             "psnr/fine":0}
            
        B, L, _, imgH, imgW = batch['source_imgs'].shape
        
        scan_name = batch['meta'][0].split("_")[0]
        ref_view = batch['meta'][0].split("_")[-1]
        
        # ---------------------- step 0: infer image features -------------------------------
        stages = ["stage1"]
        # source_imgs_feat
        
        proj_matrices, near_fars, depth_values_org = batch['proj_matrices'], batch['near_fars'], batch['depth_values_org_scale']

        source_imgs = batch['source_imgs']       
        all_imgs_pair, all_proj_mats_pair, depth_values_org_pair = self.build_pairs(source_imgs, proj_matrices, depth_values_org)
        source_imgs_feat, volume_info = self.transmvsnet(all_imgs_pair, all_proj_mats_pair, depth_values_org_pair)
        for i in range(len(source_imgs_feat)):
            source_imgs_feat[i]['stage1'] = source_imgs_feat[i]['stage1'][0:1]
        src_match_feats_list = self.transmvsnet.get_match_feat(source_imgs_feat, cur_n_src_views=self.args.train_n_view-1)

        source_imgs_feat_dict = {}
        for stage in stages:
            source_imgs_feat_dict[stage] = torch.stack([feat[stage] for feat in source_imgs_feat], dim=1) #* (B V C H W)
        source_imgs_feat = source_imgs_feat_dict["stage1"]
        
        if self.args.volume_type =="featuregrid" and self.args.volume_reso > 0:
            feature_volume = self.build_feature_volume(batch, source_imgs_feat) #* (B, 4, 32, 128, 160)
        elif self.args.volume_type == "correlation":
            feature_volume = {"stage1": volume_info['stage1']['cost_volume'], "stage2": volume_info['stage2']['cost_volume'], "stage3": volume_info['stage3']['cost_volume']}
            volume_feat, volume_weight = self.build_mvs_volume(batch, feature_volume['stage1'])
            volume_feat_2, volume_weight_2 = self.build_mvs_volume(batch, feature_volume['stage2'])
            volume_feat_3, volume_weight_3 = self.build_mvs_volume(batch, feature_volume['stage3'])
            
            feature_volume = { "stage1": {"feature_volume": volume_feat, "weight_volume": volume_weight}, 
                               "stage2": {"feature_volume": volume_feat_2, "weight_volume": volume_weight_2},
                               "stage3": {"feature_volume": volume_feat_3, "weight_volume": volume_weight_3}}
        else:
            raise NotImplementedError
        
        if self.args.mvs_depth_guide > 0:
            depth_info = volume_info['stage3']['depth'] * batch['scale_factor'] #TODO: stage1을 쓰면 coarse한 Depth, stage3을 쓰면 fine한 Depth
            batch['depth_info'] = depth_info.unsqueeze(0)
        
        if volume_info is not None:
            del volume_info 
        
        # ---------------------- step 1: sample rays -----------------------------------------
        ray_idx_all = repeat(torch.arange(imgH * imgW), "HW -> B HW", B = B).type_as(batch['ray_o']).long() 

        rgb_list, rgb_gt_list, depth_list, rgb_list_2, depth_list_2 = [], [], [], [], []
        
        for ray_idx in tqdm(torch.split(ray_idx_all, self.train_ray_num, dim=1)):
            rgb_gt, rgb, depth, _, _, _, _, _, \
                rgb_2, depth_2, _, _, _, _, _, _, variance = \
                        self.infer(batch=batch, ray_idx=ray_idx, source_imgs_feat=source_imgs_feat, feature_volume=feature_volume, match_feature=src_match_feats_list, is_train=False)

            rgb_list.append(rgb)
            rgb_gt_list.append(rgb_gt)
            depth_list.append(depth)
            rgb_list_2.append(rgb_2)
            depth_list_2.append(depth_2)
            
        depth_list_save = depth_list.copy()

        rgb_list_save = rgb_list_2.copy()
        rgb_list = torch.cat(rgb_list, dim=1)
        rgb_gt_list = torch.cat(rgb_gt_list, axis=1)
        depth_list = torch.cat(depth_list, axis=1)
        rgb_list_2 = torch.cat(rgb_list_2, dim=1)
        depth_list_2 = torch.cat(depth_list_2, axis=1)

        # move to cpu
        to_CPU = lambda x: x.cpu().numpy()
        variance = to_CPU(variance)

        rgb_imgs = rearrange(rgb_list, "B (H W) DimRGB -> B DimRGB H W", H=imgH)
        rgb_gt_imgs = rearrange(rgb_gt_list, "B (H W) DimRGB -> B DimRGB H W", H=imgH)
        depths = rearrange(depth_list, "B (H W) -> B H W", H=imgH)
        rgb_imgs_2 = rearrange(rgb_list_2, "B (H W) DimRGB -> B DimRGB H W", H=imgH)
        depths_2 = rearrange(depth_list_2, "B (H W) -> B H W", H=imgH)
        
        # metrics
        loss_rgb = torch.nn.functional.mse_loss(rgb_list, rgb_gt_list)
        loss_rgb_2 = torch.nn.functional.mse_loss(rgb_list_2, rgb_gt_list)

        psnr_coarse = piq.psnr(torch.clamp(rgb_imgs, max=1, min=0), torch.clamp(rgb_gt_imgs, max=1, min=0)).item()
        psnr_fine = piq.psnr(torch.clamp(rgb_imgs_2, max=1, min=0), torch.clamp(rgb_gt_imgs, max=1, min=0)).item()

        # return depth loss and log it
        depth_gt = batch['depths_h'][:,0]
        
        # Depth loss
        B,H,W = depth_gt.size()
        mask_depth = (depth_gt!=0) & (depth_gt>=batch['near_fars'][:,0:1,0:1]) & (depth_gt<=batch['near_fars'][:,0:1,1:2])

        if torch.sum(mask_depth)>0:
            # masked out where gt depth is invalid
            loss_depth_ray =  torch.nn.functional.l1_loss(depths[mask_depth], depth_gt[mask_depth]) 
            loss_depth_ray2 = torch.nn.functional.l1_loss(depths_2[mask_depth], depth_gt[mask_depth])
        else:
            loss_depth_ray = loss_depth_ray2 = 0.0

        rgbs = torch.cat(rgb_list_save, dim=1).view(imgH, imgW,-1)
        #* tensor to numpy
        rgbs = rgbs.cpu().numpy()
        rgbs = (rgbs.astype(np.float32) * 255).astype(np.uint8)
        
        depths = torch.cat(depth_list_save, dim=1).view(imgH, imgW) # H W
        depths = depths * batch['scale_mat'][0][0, 0]  # scale back
        depths = depths.cpu().numpy()
        
        os.makedirs(os.path.join(self.args.logdir, scan_name, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.args.logdir, "depth", scan_name), exist_ok=True)
        os.makedirs(os.path.join(self.args.logdir, "rgb", scan_name), exist_ok=True)
        
        #* save depth and rgb images
        depth_save = ((depths / np.max(depths)).astype(np.float32) * 255).astype(np.uint8)
        Image.fromarray(depth_save).save(os.path.join(self.args.logdir, scan_name, "depth", "%s.png"%ref_view))
        Image.fromarray(rgbs).save(os.path.join(self.args.logdir, "rgb", scan_name, "%s.jpg"%ref_view))

        extrinsic_np = batch['w2cs'][0][0].cpu().numpy()

        np.save(os.path.join(self.args.logdir, "depth", scan_name, "%s.npy"%ref_view), 
                {"depth": depths, "extrinsic":extrinsic_np, "intrinsic": batch['intrinsics'][0][0].cpu().numpy()})
        
        
        return {"val/loss_rgb_coarse":loss_rgb.item(), 
                "val/loss_rgb_fine":loss_rgb_2.item(), 
                "val/loss_depth_coarse":loss_depth_ray.item(), 
                "val/loss_depth_fine":loss_depth_ray2.item(), 
                "psnr/coarse":psnr_coarse, 
                "psnr/fine":psnr_fine,
                "val/variance": variance}


    def extract_geometry(self, batch, batch_idx):
        
        B, L, _, imgH, imgW = batch['source_imgs'].shape #* L denotes the number of source images
        scan_name = batch['meta'][0].split("-")[1]
        ref_view = batch['meta'][0].split("-")[-1]
        os.makedirs(os.path.join(self.args.out_dir, scan_name, "depth"), exist_ok=True)
        os.makedirs(os.path.join(self.args.out_dir, "depth", scan_name), exist_ok=True)
        os.makedirs(os.path.join(self.args.out_dir, "rgb", scan_name), exist_ok=True)

        print("strat extracting geometry")
        ray_idx_all = repeat(torch.arange(imgH * imgW), "HW -> B HW", B = B).type_as(batch['ray_o']).long()
        depth_list, rgb_list  = [], []
        
        stages = ["stage1"]
        
        volume_info = None
        proj_matrices, near_fars, depth_values_org = batch['proj_matrices'], batch['near_fars'], batch['depth_values_org_scale']

        source_imgs = batch['source_imgs']            
        all_imgs_pair, all_proj_mats_pair, depth_values_org_pair = self.build_pairs(source_imgs, proj_matrices, depth_values_org)
        source_imgs_feat, volume_info = self.transmvsnet(all_imgs_pair, all_proj_mats_pair, depth_values_org_pair)
        for i in range(len(source_imgs_feat)):
            source_imgs_feat[i]['stage1'] = source_imgs_feat[i]['stage1'][0:1]

        src_match_feats_list = self.transmvsnet.get_match_feat(source_imgs_feat, cur_n_src_views=self.args.test_n_view)
        
        source_imgs_feat_dict = {}
        for stage in stages:
            source_imgs_feat_dict[stage] = torch.stack([feat[stage] for feat in source_imgs_feat], dim=1) #* (B V C H W)
        source_imgs_feat = source_imgs_feat_dict["stage1"]
        
        if self.args.volume_type =="featuregrid" and self.args.volume_reso > 0:
            feature_volume = self.build_feature_volume(batch, source_imgs_feat) #* (B, 4, 32, 128, 160)
        elif self.args.volume_type == "correlation":
            feature_volume = {"stage1": volume_info['stage1']['cost_volume'], "stage2": volume_info['stage2']['cost_volume'], "stage3": volume_info['stage3']['cost_volume']}
            volume_feat, volume_weight = self.build_mvs_volume(batch, feature_volume['stage1'])
            volume_feat_2, volume_weight_2 = self.build_mvs_volume(batch, feature_volume['stage2'])
            volume_feat_3, volume_weight_3 = self.build_mvs_volume(batch, feature_volume['stage3'])
            
            feature_volume = { "stage1": {"feature_volume": volume_feat, "weight_volume": volume_weight}, 
                               "stage2": {"feature_volume": volume_feat_2, "weight_volume": volume_weight_2},
                               "stage3": {"feature_volume": volume_feat_3, "weight_volume": volume_weight_3}}
        else:
            raise NotImplementedError
        
        if self.args.mvs_depth_guide > 0:
            depth_info = volume_info['stage3']['depth'] * batch['scale_factor'] #TODO: stage1을 쓰면 coarse한 Depth, stage3을 쓰면 fine한 Depth
            batch['depth_info'] = depth_info.unsqueeze(0)
        
        if volume_info is not None:
            del volume_info 
        
        #* infer
        for ray_idx in tqdm(torch.split(ray_idx_all, self.args.test_ray_num, dim=1)):
            srdf, points_x, depth, rgb = self.infer(batch=batch, ray_idx=ray_idx, source_imgs_feat=source_imgs_feat, 
                                feature_volume=feature_volume, match_feature=src_match_feats_list, extract_geometry=True, is_train=False)

            ray_d = torch.gather(batch['cam_ray_d'], 2, repeat(ray_idx, "B RN -> B DimX RN", DimX=3))
            ray_d = rearrange(ray_d, "B DimX RN -> B RN DimX")

            depth = (depth.unsqueeze(-1) * ray_d)[:,:,2]
            depth_list.append(depth)
            rgb_list.append(rgb)

        depths = torch.cat(depth_list, dim=1).view(imgH, imgW) # H W
        depths = depths * batch['scale_mat'][0][0, 0]  #! scale back
        rgbs = torch.cat(rgb_list, dim=1).view(imgH, imgW,-1)

        #* tensor to numpy
        depths = depths.cpu().numpy()
        rgbs = rgbs.cpu().numpy()
        rgbs = (rgbs.astype(np.float32) * 255).astype(np.uint8)
        
        #* save depth and rgb images
        depth_save = ((depths / np.max(depths)).astype(np.float32) * 255).astype(np.uint8)
        Image.fromarray(depth_save).save(os.path.join(self.args.out_dir, scan_name, "depth", "%s.png"%ref_view))
        Image.fromarray(rgbs).save(os.path.join(self.args.out_dir, "rgb", scan_name, "%s.jpg"%ref_view))

        extrinsic_np = batch['extrinsic_render_view'][0].cpu().numpy()

        np.save(os.path.join(self.args.out_dir, "depth", scan_name, "%s.npy"%ref_view), 
                {"depth": depths, "extrinsic":extrinsic_np, "intrinsic": batch['intrinsic_render_view'][0].cpu().numpy()})
        
    def extract_similarity(self, batch, batch_idx, bound_min=[-1., -1., -1.], bound_max=[1., 1., 1.], resolution=512, threshold=0.0):
        
        bound_min = torch.tensor(bound_min, dtype=torch.float32)
        bound_max = torch.tensor(bound_max, dtype=torch.float32)
        device = batch['source_imgs'].device
        
        stages=['stage1']
        proj_matrices, near_fars, depth_values_org = batch['proj_matrices'], batch['near_fars'], batch['depth_values_org_scale']
        source_imgs = batch['source_imgs']            
        all_imgs_pair, all_proj_mats_pair, depth_values_org_pair = self.build_pairs(source_imgs, proj_matrices, depth_values_org)
        source_imgs_feat, volume_info = self.transmvsnet(all_imgs_pair, all_proj_mats_pair, depth_values_org_pair)
        for i in range(len(source_imgs_feat)):
            source_imgs_feat[i]['stage1'] = source_imgs_feat[i]['stage1'][0:1]
        src_match_feats_list = self.transmvsnet.get_match_feat(source_imgs_feat, cur_n_src_views=self.args.test_n_view)
        
        del all_imgs_pair, all_proj_mats_pair, depth_values_org_pair
        
        source_imgs_feat_dict = {}
        for stage in stages:
            source_imgs_feat_dict[stage] = torch.stack([feat[stage] for feat in source_imgs_feat], dim=1) #* (B V C H W)
        source_imgs_feat = source_imgs_feat_dict["stage1"]
            
        u = self.extract_fields(bound_min, bound_max, resolution,
                                self.query_cond_info,
                                # - sdf need to be multiplied by -1
                                device,
                                batch,
                                src_match_feats_list
                                # * 3d feature volume
                                )
    
        b_max_np = bound_max.detach().cpu().numpy()
        b_min_np = bound_min.detach().cpu().numpy()

        threshold = 0.99
        print('fraction occupied', np.mean(u > threshold))
        vertices, triangles = mcubes.marching_cubes(u, threshold)
        
        print('done', vertices.shape, triangles.shape)
        scan = batch['meta'][0].split("-")[1]
        mcubes.export_mesh(vertices, triangles, "sim_{}_{}.dae".format(scan, 512))

        # vertices = vertices / (resolution - 1.0) * (b_max_np - b_min_np)[None, :] + b_min_np[None, :]
        
        return u
        
    @torch.no_grad()
    def extract_fields(self, bound_min, bound_max, resolution, query_func, device, batch, src_feats_list):
        N = 64
        X = torch.linspace(bound_min[0], bound_max[0], resolution).split(N)
        Y = torch.linspace(bound_min[1], bound_max[1], resolution).split(N)
        Z = torch.linspace(bound_min[2], bound_max[2], resolution).split(N)

        u = np.zeros([resolution, resolution, resolution], dtype=np.float32)
        with torch.no_grad():
            for xi, xs in enumerate(X):
                for yi, ys in enumerate(Y):
                    for zi, zs in enumerate(Z):
                        xx, yy, zz = torch.meshgrid(xs, ys, zs)
                        pts = torch.cat([xx.reshape(-1, 1), yy.reshape(-1, 1), zz.reshape(-1, 1)], dim=-1).to(device)

                        # ! attention, the query function is different for extract geometry and fields
                        cond_info = query_func(pts, batch['source_poses'], batch['source_imgs'], src_feats_list, extract_similarity=True)
                        sim = cond_info[0]['feat_info']
                        mean_sim = sim.mean(dim=-1).reshape(len(xs), len(ys), len(zs)).detach().cpu().numpy()
                        
                        u[xi * N: xi * N + len(xs), yi * N: yi * N + len(ys), zi * N: zi * N + len(zs)] = mean_sim
        return u