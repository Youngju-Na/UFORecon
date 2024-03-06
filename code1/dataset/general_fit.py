import torch
import cv2 as cv
import numpy as np
import os
import logging
from einops import repeat
from .scene_transform import get_boundingbox


def load_K_Rt_from_P(filename, P=None):
    if P is None:
        lines = open(filename).read().splitlines()
        if len(lines) == 4:
            lines = lines[1:]
        lines = [[x[0], x[1], x[2], x[3]] for x in (x.split(" ") for x in lines)]
        P = np.asarray(lines).astype(np.float32).squeeze()

    out = cv.decomposeProjectionMatrix(P)
    K = out[0]
    R = out[1]
    t = out[2]

    K = K / K[2, 2]
    intrinsics = np.eye(4)
    intrinsics[:3, :3] = K

    pose = np.eye(4, dtype=np.float32)
    pose[:3, :3] = R.transpose()
    pose[:3, 3] = (t[:3] / t[3])[:, 0]

    return intrinsics, pose


class GeneralFit:
    def __init__(self, root_dir, scan_id, n_views=3,
                 img_wh=[768, 576], clip_wh=[0, 0], 
                 N_rays=512, test_ref_view=None, ndepths=192, dataset="blendedmvs", use_mask=False):
        super(GeneralFit, self).__init__()
        logging.info('Load data: Begin')

        self.root_dir = root_dir
        self.scan_id = scan_id
        self.offset_dist = 0.0 # 25mm, assume the metric is meter
        self.n_views = n_views 
        self.render_views = n_views
        self.test_ref_view = test_ref_view
        self.metas = self.build_list()
        self.num_img = len(self.metas)
        self.dataset = dataset
        self.use_mask = use_mask
        
        # set the test view combs
        
        # self.test_img_idx = list(range(n_views)) # 0,1,2,3,4
        self.test_img_idx = list(self.metas)

        self.data_dir = os.path.join(self.root_dir, self.scan_id)

        if self.dataset == "blendedmvs":
            self.img_wh = [768, 576]
        elif self.dataset == "mvimage":
            self.img_wh = [960, 544]
        
        self.clip_wh = clip_wh

        if len(self.clip_wh) == 2:
            self.clip_wh = self.clip_wh + self.clip_wh

        self.N_rays = N_rays
        
        self.bbox_min = np.array([-1.0, -1.0, -1.0])
        self.bbox_max = np.array([1.0, 1.0, 1.0])
        self.partial_vol_origin = torch.Tensor([-1., -1., -1.])

        self.img_W, self.img_H = self.img_wh
        h_line = (np.linspace(0,self.img_H-1,self.img_H))*2/(self.img_H-1) - 1
        w_line = (np.linspace(0,self.img_W-1,self.img_W))*2/(self.img_W-1) - 1
        h_mesh, w_mesh = np.meshgrid(h_line, w_line, indexing='ij')
        self.w_mesh_flat = w_mesh.reshape(-1)
        self.h_mesh_flat = h_mesh.reshape(-1)
        self.homo_pixel = np.stack([self.w_mesh_flat, self.h_mesh_flat, np.ones(len(self.h_mesh_flat)), np.ones(len(self.h_mesh_flat))])

        self.ndepths = ndepths
        
        logging.info('Load data: End')


    def build_list(self):
        metas = []  
        pair_file = os.path.join(self.root_dir, self.scan_id, "cams", "pair.txt")
            
        # read the pair file
        with open(pair_file) as f:
            num_viewpoint = int(f.readline())
            for view_idx in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                
                if len(self.test_ref_view) > 0:
                    if ref_view not in self.test_ref_view:
                        continue
                    else:
                        src_views = self.test_ref_view
                
                metas.append((ref_view, src_views))
        
        # print("dataset", "metas:", len(metas))
            
        return metas
            

    def read_cam_file(self, filename):
        """
        Load camera file e.g., 00000000_cam.txt
        """
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
        intrinsics_[:3, :3] = intrinsics
        P = intrinsics_ @ extrinsics
        # depth_min & depth_interval: line 11
        near = float(lines[11].split()[0])
        far = float(lines[11].split()[-1])
        
        self.depth_min = near
        self.depth_interval = float(lines[11].split()[1]) * 1.06

        return P, near, far
    
    def read_cam_file_mvimage(self, filename):
        """
        Load camera file e.g., 00000000_cam.txt
        """
        with open(filename) as f:
            lines = [line.rstrip() for line in f.readlines()]
        # extrinsics: line [1,5), 4x4 matrix
        extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
        extrinsics = extrinsics.reshape((4, 4))
        # intrinsics: line [7-10), 3x3 matrix
        intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
        intrinsics = intrinsics.reshape((3, 3))
        intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
        intrinsics_[:3, :3] = intrinsics
        P = intrinsics_ @ extrinsics
        # depth_min & depth_interval: line 11
        near = 400.
        far = 900.
        

        return P, near, far


    def load_scene(self, images_list, world_mats_np, ref_w2c, masks_list=None):
        all_images = []
        all_masks = []
        all_intrinsics = []
        all_w2cs = []
        all_w2cs_original = []
        all_render_w2cs = []
        all_render_w2cs_original = []
        

        for idx in range(len(images_list)):
            image = cv.imread(images_list[idx])
            original_h, original_w, _ = image.shape
            scale_x = self.img_wh[0] / original_w
            scale_y = self.img_wh[1] / original_h
            image = cv.resize(image, (self.img_wh[0], self.img_wh[1])) / 255.
            
            if len(masks_list) > 0 and self.use_mask:
                mask = cv.imread(masks_list[idx], 0) 
                mask = cv.resize(mask, (self.img_wh[0], self.img_wh[1])) / 254.
                # apply foreground mask to remove background
                image = image * np.expand_dims(mask, -1)
                
            image = image[self.clip_wh[1]:self.img_wh[1] - self.clip_wh[3],
                    self.clip_wh[0]:self.img_wh[0] - self.clip_wh[2]]
            all_images.append(np.transpose(image[:, :, ::-1], (2, 0, 1)))

            P = world_mats_np[idx]
            P = P[:3, :4]
            intrinsics, c2w = load_K_Rt_from_P(None, P)
            w2c = np.linalg.inv(c2w)

            render_c2w = c2w.copy()
            render_c2w[:3,3] += render_c2w[:3,0]*self.offset_dist
            render_w2c = np.linalg.inv(render_c2w)

            intrinsics[:1] *= scale_x
            intrinsics[1:2] *= scale_y

            intrinsics[0, 2] -= self.clip_wh[0]
            intrinsics[1, 2] -= self.clip_wh[1]

            all_intrinsics.append(intrinsics)
            # - transform from world system to ref-camera system
            all_w2cs.append(w2c @ np.linalg.inv(ref_w2c))
            all_render_w2cs.append(render_w2c @ np.linalg.inv(ref_w2c))
            all_w2cs_original.append(w2c)
            all_render_w2cs_original.append(render_w2c)

        all_images = torch.from_numpy(np.stack(all_images)).to(torch.float32)
        all_intrinsics = torch.from_numpy(np.stack(all_intrinsics)).to(torch.float32)
        all_w2cs = torch.from_numpy(np.stack(all_w2cs)).to(torch.float32)
        all_render_w2cs = torch.from_numpy(np.stack(all_render_w2cs)).to(torch.float32)

        return all_images, all_intrinsics, all_w2cs, all_w2cs_original, all_render_w2cs, all_render_w2cs_original


    def cal_scale_mat(self, img_hw, intrinsics, extrinsics, near_fars, factor=1.):
        center, radius, _ = get_boundingbox(img_hw, intrinsics, extrinsics, near_fars)
        radius = radius * factor
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center.cpu().numpy()
        scale_mat = scale_mat.astype(np.float32)

        return scale_mat, 1. / radius.cpu().numpy()


    def scale_cam_info(self, all_images, all_intrinsics, all_w2cs, scale_mat, all_render_w2cs):
        new_intrinsics = []
        new_w2cs = []
        new_c2ws = []
        new_render_w2cs = []
        new_render_c2ws = []
        proj_matrices = []
        new_near_fars = []
        
        for idx in range(len(all_images)):
            intrinsics = all_intrinsics[idx]
            P = intrinsics @ all_w2cs[idx] @ scale_mat
            P = P.cpu().numpy()[:3, :4]

            c2w = load_K_Rt_from_P(None, P)[1]
            w2c = np.linalg.inv(c2w)
            new_w2cs.append(w2c)
            new_c2ws.append(c2w)
            new_intrinsics.append(intrinsics)

            camera_o = c2w[:3, 3]
            dist = np.sqrt(np.sum(camera_o ** 2))
            near = dist - 1
            far = dist + 1
            new_near_fars.append([0.95 * near, 1.05 * far])
            
            proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32) 
            intrinsic_cp = all_intrinsics[idx].clone()[:3, :3]
            intrinsic_cp[:2] /= 4 # 1/4->1/2->1
            proj_mat[0, :4, :4] = all_w2cs[idx]
            proj_mat[1, :3, :3] = intrinsic_cp
            proj_matrices.append(proj_mat)
                
            P = intrinsics @ all_render_w2cs[idx] @ scale_mat
            P = P.cpu().numpy()[:3, :4]

            c2w = load_K_Rt_from_P(None, P)[1]
            w2c = np.linalg.inv(c2w)
            new_render_w2cs.append(w2c)
            new_render_c2ws.append(c2w)

        new_intrinsics, new_w2cs, new_c2ws, new_near_fars = \
            np.stack(new_intrinsics), np.stack(new_w2cs), np.stack(new_c2ws), \
            np.stack(new_near_fars)
        new_render_w2cs, new_render_c2ws = np.stack(new_render_w2cs), np.stack(new_render_c2ws)
        
        new_intrinsics = torch.from_numpy(np.float32(new_intrinsics))
        new_w2cs = torch.from_numpy(np.float32(new_w2cs))
        new_c2ws = torch.from_numpy(np.float32(new_c2ws))
        new_near_fars = torch.from_numpy(np.float32(new_near_fars))
        new_render_w2cs = torch.from_numpy(np.float32(new_render_w2cs))
        new_render_c2ws = torch.from_numpy(np.float32(new_render_c2ws))

        #! transmvsnet
        proj_matrices = np.stack(proj_matrices)
        stage2_pjmats = proj_matrices.copy()
        stage2_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 2
        stage3_pjmats = proj_matrices.copy()
        stage3_pjmats[:, 1, :2, :] = proj_matrices[:, 1, :2, :] * 4
    
        proj_matrices_ms = {
            "stage1": proj_matrices,
            "stage2": stage2_pjmats,
            "stage3": stage3_pjmats
        }
        
        return new_intrinsics, new_w2cs, new_c2ws, new_near_fars, new_render_w2cs, new_render_c2ws, proj_matrices_ms


    def __len__(self):
        return self.num_img


    def __getitem__(self, idx):
        sample = {}
        ref_view, src_views = self.metas[idx]
        view_ids = [ref_view] + src_views[:self.n_views-1]
        idx = list(range(self.n_views))
        render_idx = 0
        src_idx = idx
        # src_idx = self.test_img_idx[:]

        world_mats_np = []
        images_list = []
        raw_near_fars = []
        masks_list = []
        for i, vid in enumerate(view_ids):
            if self.dataset == "blendedmvs":
                img_filename = os.path.join(self.data_dir, 'blended_images/{:0>8}_masked.jpg'.format(vid))
            elif self.dataset == "mvimage":
                img_filename = os.path.join(self.data_dir, 'images/{:0>8}.jpg'.format(vid))
                scan = self.data_dir.split("/")[-1]
                mask_filename = os.path.join(os.path.dirname(self.data_dir), scan, 'masks', '{:0>8}_mask.jpg'.format(vid))
                masks_list.append(mask_filename)
                    
            images_list.append(img_filename)
            
            
            proj_mat_filename = os.path.join(self.data_dir, 'cams/{:0>8}_cam.txt'.format(vid))
            P, near_, far_ = self.read_cam_file(proj_mat_filename)

            # if self.dataset=="mvimage":
            #     P, near_, far_ = self.read_cam_file_mvimage(proj_mat_filename)
            
            raw_near_fars.append(np.array([near_,far_]))
            world_mats_np.append(P)
        raw_near_fars = np.stack(raw_near_fars)
        ref_world_mat = world_mats_np[0]
        ref_w2c = np.linalg.inv(load_K_Rt_from_P(None, ref_world_mat[:3, :4])[1])

        all_images, all_intrinsics, all_w2cs, all_w2cs_original, all_render_w2cs, all_render_w2cs_original = self.load_scene(images_list, 
                                                                                                                    world_mats_np, ref_w2c, masks_list=masks_list)

        
        scale_mat, scale_factor = self.cal_scale_mat(
            img_hw=[self.img_wh[1], self.img_wh[0]],
            intrinsics=all_intrinsics,
            extrinsics=all_w2cs,
            near_fars=raw_near_fars,
            factor=1.1)
        scaled_intrinsics, scaled_w2cs, scaled_c2ws, scaled_near_fars, scaled_render_w2cs, scaled_render_c2ws, proj_matrices_ms = self.scale_cam_info(all_images, 
                                                                            all_intrinsics, all_w2cs, scale_mat, all_render_w2cs)

        near_fars = torch.tensor(raw_near_fars[0])
        near_fars = near_fars/scale_mat[0,0]
        near_fars = near_fars.view(1,2)
        sample['near_fars'] = near_fars.float()

        sample['scale_mat'] = torch.from_numpy(scale_mat)
        sample['trans_mat'] = torch.from_numpy(np.linalg.inv(ref_w2c))
        sample['extrinsic_render_view'] = torch.from_numpy(all_render_w2cs_original[render_idx])
        
        sample['w2cs'] = scaled_w2cs  # (V, 4, 4)
        sample['intrinsics'] = scaled_intrinsics[:, :3, :3]  # (V, 3, 3)
        sample['intrinsic_render_view'] = sample['intrinsics'][render_idx]
        
        #! transmvsnet
        sample['proj_matrices'] = proj_matrices_ms

        depth_max = self.depth_interval * self.ndepths + self.depth_min
        depth_values_org_scale = np.arange(self.depth_min, depth_max, self.depth_interval, dtype=np.float32)
        sample['depth_values_org_scale'] = depth_values_org_scale
    
        sample['near_fars'] = scaled_near_fars.float()    
        sample['ref_img'] = all_images[render_idx]
        sample['source_imgs'] = all_images[src_idx]
        sample['scale_factor'] = scale_factor

        intrinsics_pad = repeat(torch.eye(4), "X Y -> L X Y", L = len(sample['w2cs'])).clone()
        intrinsics_pad[:,:3,:3] = sample['intrinsics']
        
        sample['ref_pose']         = (intrinsics_pad @ scaled_render_w2cs)[render_idx]     # 4, 4
        sample['source_poses']     = (intrinsics_pad @ sample['w2cs'])[src_idx]

        # from 0~W to NDC's -1~1
        normalize_matrix = torch.tensor([[1/((self.img_W-1)/2), 0, -1, 0], [0, 1/((self.img_H-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]])

        sample['ref_pose'] = normalize_matrix @ sample['ref_pose']
        sample['source_poses'] = normalize_matrix @ sample['source_poses']

        sample['ref_pose_inv'] = torch.inverse(sample['ref_pose'])
        sample['source_poses_inv'] = torch.inverse(sample['source_poses'])

        sample['ray_o'] = sample['ref_pose_inv'][:3,-1]      # 3

        tmp_ray_d = (sample['ref_pose_inv'] @ self.homo_pixel)[:3] - sample['ray_o'][:,None]

        sample['ray_d'] = tmp_ray_d / torch.norm(tmp_ray_d, dim=0)
        sample['ray_d'] = sample['ray_d'].float()


        cam_ray_d = ((torch.inverse(normalize_matrix @ intrinsics_pad[0])) @ self.homo_pixel)[:3]
        cam_ray_d = cam_ray_d / torch.norm(cam_ray_d, dim=0)
        sample['cam_ray_d'] = cam_ray_d.float()

        sample['meta'] = "%s-%s-refview%d"%(self.root_dir.split("/")[-1], self.scan_id, ref_view)

        sample['start_idx'] = 0
        return sample