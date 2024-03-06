import torch
import torch.nn as nn
import torch.nn.functional as F
from .module import *
from .FMT import FMT_with_pathway

Align_Corners_Range = False

class PixelwiseNet(nn.Module):

    def __init__(self):

        super(PixelwiseNet, self).__init__()
        self.conv0 = ConvBnReLU3D(in_channels=1, out_channels=16, kernel_size=1, stride=1, pad=0)
        self.conv1 = ConvBnReLU3D(in_channels=16, out_channels=8, kernel_size=1, stride=1, pad=0)
        self.conv2 = nn.Conv3d(in_channels=8, out_channels=1, kernel_size=1, stride=1, padding=0)
        self.output = nn.Sigmoid()

    def forward(self, x1):
        """forward.
        :param x1: [B, 1, D, H, W]
        """
        x1 = self.conv2(self.conv1(self.conv0(x1))).squeeze(1) # [B, D, H, W]
        output = self.output(x1)
        output = torch.max(output, dim=1, keepdim=True)[0] # [B, 1, H ,W]

        return output


class DepthNet(nn.Module):
    def __init__(self):
        super(DepthNet, self).__init__()
        self.pixel_wise_net = PixelwiseNet()

    def forward(self, features, proj_matrices, depth_values, num_depth, cost_regularization, prob_volume_init=None, view_weights=None, mvs_volume_only=False):
        """forward.

        :param stage_idx: int, index of stage, [1, 2, 3], stage_1 corresponds to lowest image resolution
        :param features: torch.Tensor, TODO: [B, C, H, W]
        :param proj_matrices: torch.Tensor,
        :param depth_values: torch.Tensor, TODO: [B, D, H, W]
        :param num_depth: int, Ndepth
        :param cost_regularization: nn.Module, regularization network
        :param prob_volume_init:
        :param view_weights: pixel wise view weights for src views
        """
        proj_matrices = torch.unbind(proj_matrices, 1)
        assert len(features) == len(proj_matrices), "Different number of images and projection matrices"
        assert depth_values.shape[1] == num_depth, "depth_values.shape[1]:{}  num_depth:{}".format(depth_values.shapep[1], num_depth)

        # step 1. feature extraction
        ref_feature, src_features = features[0], features[1:] # [B, C, H, W]
        ref_proj, src_projs = proj_matrices[0], proj_matrices[1:] # [B, 2, 4, 4]

        # step 2. differentiable homograph, build cost volume
        if view_weights == None:
            view_weight_list = []

        similarity_sum = 0
        pixel_wise_weight_sum = 1e-5

        for i, (src_fea, src_proj) in enumerate(zip(src_features, src_projs)): # src_fea: [B, C, H, W]
            src_proj_new = src_proj[:, 0].clone() # [B, 4, 4]
            src_proj_new[:, :3, :4] = torch.matmul(src_proj[:, 1, :3, :3], src_proj[:, 0, :3, :4])
            ref_proj_new = ref_proj[:, 0].clone() # [B, 4, 4]
            ref_proj_new[:, :3, :4] = torch.matmul(ref_proj[:, 1, :3, :3], ref_proj[:, 0, :3, :4])
            warped_volume = homo_warping_trans(src_fea, src_proj_new, ref_proj_new, depth_values)
            similarity = (warped_volume * ref_feature.unsqueeze(2)).mean(1, keepdim=True)

            if view_weights == None:
                view_weight = self.pixel_wise_net(similarity) # [B, 1, H, W]
                view_weight_list.append(view_weight)
            else:
                view_weight = view_weights[:, i:i+1]

            if self.training:
                similarity_sum = similarity_sum + similarity * view_weight.unsqueeze(1) # [B, 1, D, H, W]
                pixel_wise_weight_sum = pixel_wise_weight_sum + view_weight.unsqueeze(1) # [B, 1, 1, H, W]
            else:
                # TODO: this is only a temporal solution to save memory, better way?
                similarity_sum += similarity * view_weight.unsqueeze(1)
                pixel_wise_weight_sum += view_weight.unsqueeze(1)

            del warped_volume
        # aggregate multiple similarity across all the source views
        similarity = similarity_sum.div_(pixel_wise_weight_sum) # [B, 1, D, H, W]
        
        # step 3. cost volume regularization
        cost_reg = cost_regularization(similarity)
        
        if not mvs_volume_only:
            prob_volume_pre = cost_reg.squeeze(1)

            if prob_volume_init is not None:
                prob_volume_pre += prob_volume_init

            prob_volume = torch.exp(F.log_softmax(prob_volume_pre, dim=1))
            depth = depth_wta(prob_volume, depth_values=depth_values)

            with torch.no_grad():
                photometric_confidence = torch.max(prob_volume, dim=1)[0]
    
            if view_weights == None:
                view_weights = torch.cat(view_weight_list, dim=1) # [B, Nview, H, W]
                return {"depth": depth,  "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values, "cost_volume": cost_reg}, view_weights.detach()
            else:
                return {"depth": depth,  "photometric_confidence": photometric_confidence, "prob_volume": prob_volume, "depth_values": depth_values, "cost_volume": cost_reg}


class TransMVSNet(nn.Module):
    def __init__(self, refine=False, ndepths=[48, 32, 8], depth_interals_ratio=[4, 2, 1], share_cr=False,
            grad_method="detach", arch_mode="fpn", cr_base_chs=[8, 8, 8]):
        super(TransMVSNet, self).__init__()
        self.refine = refine
        self.share_cr = share_cr
        self.ndepths = ndepths
        self.depth_interals_ratio = depth_interals_ratio
        self.grad_method = grad_method
        self.arch_mode = arch_mode
        self.cr_base_chs = cr_base_chs
        self.num_stage = len(ndepths)
        print("**********netphs:{}, depth_intervals_ratio:{},  grad:{}, chs:{}************".format(ndepths,
            depth_interals_ratio, self.grad_method, self.cr_base_chs))

        assert len(ndepths) == len(depth_interals_ratio)

        self.stage_infos = {
                "stage1":{
                    "scale": 4.0,
                    },
                "stage2": {
                    "scale": 2.0,
                    },
                "stage3": {
                    "scale": 1.0,
                    }
                }

        self.feature = FeatureNet(base_channels=8)

        self.FMT_with_pathway = FMT_with_pathway()

        if self.share_cr:
            self.cost_regularization = CostRegNet(in_channels=1, base_channels=8)
        else:
            self.cost_regularization = nn.ModuleList([CostRegNet(in_channels=1, base_channels=self.cr_base_chs[i])
                for i in range(self.num_stage)])
        if self.refine:
            self.refine_network = RefineNet()

        self.DepthNet = DepthNet()

    def forward(self, imgs, proj_matrices, depth_values, test_tnt=False, mvs_volume_only=False):
        
        depth_min = float(depth_values[0, 0].cpu().numpy())
        depth_max = float(depth_values[0, -1].cpu().numpy())
        depth_interval = (depth_max - depth_min) / depth_values.size(1)

        # step 1. feature extraction
        features_backbone = []
        for nview_idx in range(imgs.size(1)):
            img = imgs[:, nview_idx]
            features_backbone.append(self.feature(img))
        
        features = self.FMT_with_pathway(features_backbone, ref_idx=0) #* ref_idx:0, src_idx: 1,...,N
        
        outputs = {}
        depth, cur_depth = None, None
        view_weights = None
        for stage_idx in range(self.num_stage):
            features_stage = [feat["stage{}".format(stage_idx + 1)] for feat in features]
            proj_matrices_stage = proj_matrices["stage{}".format(stage_idx + 1)]
            stage_scale = self.stage_infos["stage{}".format(stage_idx + 1)]["scale"]
            
            Using_inverse_d = False

            if depth is not None:
                if self.grad_method == "detach":
                    cur_depth = depth.detach()
                else:
                    cur_depth = depth
                cur_depth = F.interpolate(cur_depth.unsqueeze(1),
                        [img.shape[2], img.shape[3]], mode='bilinear',
                        align_corners=Align_Corners_Range).squeeze(1)
            else:
                cur_depth = depth_values

            # [B, D, H, W]
            depth_range_samples = get_depth_range_samples(cur_depth=cur_depth,
                    ndepth=self.ndepths[stage_idx],
                    depth_inteval_pixel=self.depth_interals_ratio[stage_idx] * depth_interval,
                    dtype=img[0].dtype,
                    device=img[0].device,
                    shape=[img.shape[0], img.shape[2], img.shape[3]],
                    max_depth=depth_max,
                    min_depth=depth_min,
                    use_inverse_depth=Using_inverse_d)
            
            if stage_idx + 1 > 1: # for stage 2 and 3
                view_weights = F.interpolate(view_weights, scale_factor=2, mode="nearest")

            if view_weights == None: # stage 1
                outputs_stage, view_weights = self.DepthNet(
                        features_stage,
                        proj_matrices_stage,
                        depth_values=F.interpolate(depth_range_samples.unsqueeze(1), [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear', align_corners=Align_Corners_Range).squeeze(1),
                        num_depth=self.ndepths[stage_idx],
                        cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx], view_weights=view_weights)
            else:
                outputs_stage = self.DepthNet(
                        features_stage,
                        proj_matrices_stage,
                        depth_values=F.interpolate(depth_range_samples.unsqueeze(1), [self.ndepths[stage_idx], img.shape[2]//int(stage_scale), img.shape[3]//int(stage_scale)], mode='trilinear', align_corners=Align_Corners_Range).squeeze(1),
                        num_depth=self.ndepths[stage_idx],
                        cost_regularization=self.cost_regularization if self.share_cr else self.cost_regularization[stage_idx], view_weights=view_weights)

            wta_index_map = torch.argmax(outputs_stage['prob_volume'], dim=1, keepdim=True).type(torch.long)
            depth = torch.gather(outputs_stage['depth_values'], 1, wta_index_map).squeeze(1)
            outputs_stage['depth'] = depth

            outputs["stage{}".format(stage_idx + 1)] = outputs_stage
            outputs.update(outputs_stage)

        if self.refine:
            refined_depth = self.refine_network(torch.cat((imgs[:, 0], depth), 1))
            outputs["refined_depth"] = refined_depth

        return features_backbone, outputs 
    
    def build_volume_costvar_img(self, imgs, feats, proj_mats, depth_values, pad=0):
        # feats: (B, V, C, H, W)
        # proj_mats: (B, V, 3, 4)
        # depth_values: (B, D, H, W)
        # cost_reg: nn.Module of input (B, C, D, h, w) and output (B, 1, D, h, w)
        # volume_sum [B, G, D, h, w]
        # prob_volume [B D H W]
        # volume_feature [B C D H W]

        B, V, C, H, W = feats.shape
        D = depth_values.shape[1]
        ref_feats, src_feats = feats[:, 0], feats[:, 1:]
        src_feats = src_feats.permute(1, 0, 2, 3, 4)  # (V-1, B, C, h, w)
        proj_mats = proj_mats[:, 1:]
        proj_mats = proj_mats.permute(1, 0, 2, 3)  # (V-1, B, 3, 4)

        if pad > 0:
            ref_feats = F.pad(ref_feats, (pad, pad, pad, pad), "constant", 0)

        img_feat = torch.empty((B, 9 + 32, D, *ref_feats.shape[-2:]), device=feats.device, dtype=torch.float)
        imgs = F.interpolate(imgs.view(B * V, *imgs.shape[2:]), (H, W), mode='bilinear', align_corners=False).view(B, V,-1,H,W).permute(1, 0, 2, 3, 4)
        img_feat[:, :3, :, pad:H + pad, pad:W + pad] = imgs[0].unsqueeze(2).expand(-1, -1, D, -1, -1)

        ref_volume = ref_feats.unsqueeze(2).repeat(1, 1, D, 1, 1)  # (B, C, D, h, w)

        volume_sum = ref_volume
        volume_sq_sum = ref_volume ** 2

        del ref_feats

        in_masks = torch.ones((B, V, D, H + pad * 2, W + pad * 2), device=volume_sum.device)
        for i, (src_img, src_feat, proj_mat) in enumerate(zip(imgs[1:], src_feats, proj_mats)):
            warped_volume, grid = homo_warp(src_feat, proj_mat, depth_values, pad=pad)
            img_feat[:, (i + 1) * 3:(i + 2) * 3], _ = homo_warp(src_img, proj_mat, depth_values, src_grid=grid, pad=pad)

            grid = grid.view(B, 1, D, H + pad * 2, W + pad * 2, 2)
            in_mask = ((grid > -1.0) * (grid < 1.0))
            in_mask = (in_mask[..., 0] * in_mask[..., 1])
            in_masks[:, i + 1] = in_mask.float()

            if self.training:
                volume_sum = volume_sum + warped_volume
                volume_sq_sum = volume_sq_sum + warped_volume ** 2
            else:
                volume_sum += warped_volume
                volume_sq_sum += warped_volume.pow_(2)

            del warped_volume, src_feat, proj_mat
        del src_feats, proj_mats

        count = 1.0 / torch.sum(in_masks, dim=1, keepdim=True)
        img_feat[:, -32:] = volume_sq_sum * count - (volume_sum * count) ** 2
        del volume_sq_sum, volume_sum, count

        return img_feat, in_masks
    
    def forward_nerf(self, imgs, proj_mats, near_far, pad=0,  return_color=False, lindisp=False, ):
        # imgs: (B, V, 3, H, W)
        # proj_mats: (B, V, 3, 4) from fine to coarse
        # init_depth_min, depth_interval: (B) or float
        # near_far (B, V, 2)

        B, V, _, H, W = imgs.shape

        imgs = imgs.reshape(B * V, 3, H, W)
        feats = self.extract_feature(imgs)  # (B*V, 8, H, W), (B*V, 16, H//2, W//2), (B*V, 32, H//4, W//4)

        imgs = imgs.view(B, V, 3, H, W)

        feats_l = feats  # (B*V, C, h, w)

        feats_l = feats_l.view(B, V, *feats_l.shape[1:])  # (B, V, C, h, w)

        D = 128
        t_vals = torch.linspace(0., 1., steps=D, device=imgs.device, dtype=imgs.dtype)  # (B, D)
        near, far = near_far  # assume batch size==1
        if not lindisp:
            depth_values = near * (1.-t_vals) + far * (t_vals)
        else:
            depth_values = 1. / (1. / near * (1. - t_vals) + 1. / far * (t_vals))

        depth_values = depth_values.unsqueeze(0)
        # volume_feat, in_masks = self.build_volume_costvar(feats_l, proj_mats, depth_values, pad=pad)
        volume_feat, in_masks = self.build_volume_costvar_img(imgs, feats_l, proj_mats, depth_values, pad=pad)
        if return_color:
            feats_l = torch.cat((volume_feat[:,:V*3].view(B, V, 3, *volume_feat.shape[2:]),in_masks.unsqueeze(2)),dim=2)


        volume_feat = self.cost_reg_2(volume_feat)  # (B, 1, D, h, w)
        volume_feat = volume_feat.reshape(1,-1,*volume_feat.shape[2:])

        return volume_feat, feats_l, depth_values
    
    def extract_feature(self, features):
        pass
    
    
    def get_match_feat(self, features, cur_n_src_views=3):
        '''
        features: source images features from FPN layers
        
        '''
        out_dict = self.FMT_with_pathway.extract_cross_features(features)
        
        '''
        out_dict consists of two parts: 
        index_lists = [(a, b) for a in range(n_views - 1) for b in range(a + 1, n_views)] 
        
        1. aug_feat0s: list of features from the first images 
        2. aug_feat1s: list of features from the second images 
        '''
        
        # split the output
        img_feat_list = []
        index_lists = [(a, b) for a in range(cur_n_src_views - 1) for b in range(a + 1, cur_n_src_views)]
        for scale_idx in range(len(out_dict['aug_feat0s'])):
            img_feat = [[] for _ in range(cur_n_src_views)] #* len: src_views
            
            img1s_feats = out_dict['aug_feat0s'][scale_idx]
            img2s_feats = out_dict['aug_feat1s'][scale_idx]
            for feat_i, (i_idx, j_idx) in enumerate(index_lists):
                img_feat[i_idx].append(img1s_feats[:, feat_i])
                img_feat[j_idx].append(img2s_feats[:, feat_i])
                
            # post-process the output
            for k, v in enumerate(img_feat): #* N x (N-1) 2-dimension list, N: number of combinations
                img_feat[k] = torch.cat(v, dim=1) #! before
            
            img_feat = torch.stack(img_feat, dim=1)  # BxVxCxHxW
            img_feat_list.append(img_feat)
            
        return img_feat_list