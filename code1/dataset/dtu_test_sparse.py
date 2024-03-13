import torch
import cv2 as cv
import numpy as np
from PIL import Image
import os
import logging
from einops import repeat
from torchvision.transforms import InterpolationMode
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import pil_to_tensor

from .scene_transform import get_boundingbox

def read_pfm(filename):
    file = open(filename, 'rb')
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode('utf-8').rstrip()
    if header == 'PF':
        color = True
    elif header == 'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(r'^(\d+)\s(\d+)\s$', file.readline().decode('utf-8'))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>'  # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale

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


class DtuFitSparse:
    def __init__(self, root_dir, split, scan_id, n_views=3,
                 img_wh=[800, 640], clip_wh=[0, 0], original_img_wh=[1600, 1200],
                 N_rays=512, near=425, far=900, set=0, test_view_pair=None, depth_dir=None, ndepths=192):
        super(DtuFitSparse, self).__init__()
        logging.info('Load data: Begin')

        self.root_dir = root_dir
        self.train_dir = os.path.join(os.path.dirname(self.root_dir), 'DTU')
        self.depth_dir = depth_dir
        self.split = split
        self.scan_id = scan_id
        self.n_views = n_views
        self.offset_dist = 25 # 25mm
        self.render_views = n_views
        self.test_view_pair = test_view_pair
        self.scale_factor_diner = 0.7 / 872.
        self.depth_fname = 'TransMVSNet'
        self.downsample = 1.0
        self.ndepths = ndepths
            
        if set==0:
            # self.view_list = [23, 24, 33, 22, 15, 34, 14, 32, 16, 35, 25]
            self.view_list = self.test_view_pair #* new; [1, 23, 36] old:[25, 35, 16]
        else:
            self.view_list = [43, 42, 44, 33, 34, 32, 45, 23, 41, 24, 31]

        self.near = near
        self.far = far
        self.idx = self.view_list[:n_views]
        self.test_img_idx = list(range(n_views))

        if self.scan_id is not None:
            self.data_dir = os.path.join(self.root_dir, self.scan_id)
        else:
            self.data_dir = self.root_dir

        self.img_wh = img_wh #* [800, 624] or test with [640, 512]
        
        self.clip_wh = clip_wh

        if len(self.clip_wh) == 2:
            self.clip_wh = self.clip_wh + self.clip_wh

        self.original_img_wh = original_img_wh
        self.N_rays = N_rays

        self.world_mats_np = []
        self.images_list = []
        self.depths_list = [] #! temporal
        
        for vid in self.idx:
            proj_mat_filename = os.path.join(self.root_dir, 'cameras/{:0>8}_cam.txt'.format(vid))
            P = self.read_cam_file(proj_mat_filename)
            self.world_mats_np.append(P)
            img_filename = os.path.join(self.data_dir, 'image/{:0>6}.png'.format(vid))
            self.images_list.append(img_filename)
            
            #! load depth_mvs as a temporal solution
            # depth_filename = os.path.join(self.data_dir, self.depth_dir, '{}'.format(self.get_depth_fname(vid)))
            # depth_filename = os.path.join(self.train_dir, 'Depths_raw/{}/{}'.format(self.scan_id, self.get_depth_fname(vid)))
            # self.depths_list.append(depth_filename)

        self.raw_near_fars = np.stack([np.array([self.near, self.far]) for i in range(len(self.images_list))])
        ref_world_mat = self.world_mats_np[0]
        self.ref_w2c = np.linalg.inv(load_K_Rt_from_P(None, ref_world_mat[:3, :4])[1])

        self.all_images = []
        
        self.all_intrinsics = []
        self.all_w2cs = []
        self.all_w2cs_original = []
        self.all_render_w2cs = []
        self.all_render_w2cs_original = []

        self.load_scene()  # load the scene

        # ! estimate scale_mat
        self.scale_mat, self.scale_factor = self.cal_scale_mat(
            img_hw=[self.img_wh[1], self.img_wh[0]],
            intrinsics=self.all_intrinsics,
            extrinsics=self.all_w2cs,
            near_fars=self.raw_near_fars,
            factor=1.1)
        

        # * after scaling and translation, unit bounding box
        self.scaled_intrinsics, self.scaled_w2cs, self.scaled_c2ws, \
        self.scaled_near_fars, self.scaled_render_w2cs,  \
        self.scaled_render_c2ws, self.proj_matrices_ms = self.scale_cam_info()

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

        logging.info('Load data: End')


    def get_depth_fname(self, cam_id):
        name = f"depth_map_{cam_id:04d}_{self.depth_fname}.png"
        return name
    
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
        
        depth_min = float(lines[11].split()[0])
        depth_max = depth_min + float(lines[11].split()[1]) * 192
        self.depth_min = depth_min
        self.depth_interval = float(lines[11].split()[1]) * 1.06

        return P

    def read_mvs_depth(self, filename):
        """
        reading depth from either .png or .pfm file and returns it as torch tensor
        :param filename:
        :return:
        """
        if str(filename).endswith(".pfm"):
            depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (800, 800)
            depth_h = torch.from_numpy(depth_h)
            H, W = depth_h.shape
            h, w = int(H / 2), int(W / 2)
            depth_h = resize(depth_h[None][None], (h, w), interpolation=InterpolationMode.NEAREST)[0, 0]
            depth_h = depth_h[44:556, 80:720]  # (512, 640)

        elif str(filename).endswith(".png"):  # loading TransMVSNet prediction
            SCALE_FACTOR = 1e-4
            depth_h = pil_to_tensor(Image.open(filename)).float() * SCALE_FACTOR
            depth_h /= (0.7 / 872.)  # have to correct for scale factor used during TransMVSNet training
            depth_h = depth_h[0]  # (512, 640)

        else:
            print("file name: ", filename, " is not found")
            raise ValueError
        
        h, w = depth_h.shape[:2]
        # assert h == 512 and w == 640
        if self.downsample != 1:
            h, w = int(h * self.downsample), int(w * self.downsample)
        
        if depth_h.size() != (self.img_wh[1], self.img_wh[0]):
            depth_h = resize(depth_h[None][None], (self.img_wh[1], self.img_wh[0]), interpolation=InterpolationMode.NEAREST)[0, 0]
        mask = (depth_h > 0).float()
        depth_h *= self.scale_factor_diner

        depth_h = depth_h[None]  # (1, H, W)
        mask = mask[None]  # (1, H, W)

        return depth_h, mask
    
    def load_scene(self):

        scale_x = self.img_wh[0] / self.original_img_wh[0]
        scale_y = self.img_wh[1] / self.original_img_wh[1]

        for idx in range(len(self.images_list)):
            image = cv.imread(self.images_list[idx])
            image = cv.resize(image, (self.img_wh[0], self.img_wh[1])) / 255.

            image = image[self.clip_wh[1]:self.img_wh[1] - self.clip_wh[3],
                    self.clip_wh[0]:self.img_wh[0] - self.clip_wh[2]]
            self.all_images.append(np.transpose(image[:, :, ::-1], (2, 0, 1)))
            
            # depth_mvs_h = self.read_mvs_depth(self.depths_list[idx])[0] / self.scale_factor_diner
            # depth_mvs_h = cv.resize(depth_mvs_h.permute(1, 2, 0), (self.img_wh[0], self.img_wh[1]))
            # self.all_mvs_depths.append(depth_mvs_h)
            
            P = self.world_mats_np[idx]
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

            self.all_intrinsics.append(intrinsics)
            # - transform from world system to ref-camera system
            self.all_w2cs.append(w2c @ np.linalg.inv(self.ref_w2c))
            self.all_render_w2cs.append(render_w2c @ np.linalg.inv(self.ref_w2c))
            self.all_w2cs_original.append(w2c)
            self.all_render_w2cs_original.append(render_w2c)

        self.all_images = torch.from_numpy(np.stack(self.all_images)).to(torch.float32)
        # self.all_mvs_depths = torch.stack(self.all_mvs_depths).to(torch.float32)
        
        self.all_intrinsics = torch.from_numpy(np.stack(self.all_intrinsics)).to(torch.float32)
        self.all_w2cs = torch.from_numpy(np.stack(self.all_w2cs)).to(torch.float32)
        self.all_render_w2cs = torch.from_numpy(np.stack(self.all_render_w2cs)).to(torch.float32)
        self.all_w2cs_original = torch.from_numpy(np.stack(self.all_w2cs_original)).to(torch.float32)
        self.all_render_w2cs_original = torch.from_numpy(np.stack(self.all_render_w2cs_original)).to(torch.float32)
        
        
        self.img_wh = [self.img_wh[0] - self.clip_wh[0] - self.clip_wh[2],
                       self.img_wh[1] - self.clip_wh[1] - self.clip_wh[3]]


    def cal_scale_mat(self, img_hw, intrinsics, extrinsics, near_fars, factor=1.):
        center, radius, _ = get_boundingbox(img_hw, intrinsics, extrinsics, near_fars)
        radius = radius * factor
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center.cpu().numpy()
        scale_mat = scale_mat.astype(np.float32)

        return scale_mat, 1. / radius.cpu().numpy()


    def scale_cam_info(self):
        new_intrinsics = []
        new_near_fars = []
        new_w2cs = []
        new_c2ws = []
        new_render_w2cs = []
        new_render_c2ws = []
        proj_matrices = []
        
        for idx in range(len(self.all_images)):
            intrinsics = self.all_intrinsics[idx]
            P = intrinsics @ self.all_w2cs[idx] @ self.scale_mat
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
            intrinsic_cp = self.all_intrinsics[idx].clone()[:3, :3]
            intrinsic_cp[:2] /= 4 # 1/4->1/2->1
            proj_mat[0, :4, :4] = self.all_w2cs[idx]
            proj_mat[1, :3, :3] = intrinsic_cp
            proj_matrices.append(proj_mat)
            
            P = intrinsics @ self.all_render_w2cs[idx] @ self.scale_mat
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
        return self.render_views

    def __getitem__(self, idx):
        sample = {}
        render_idx = self.test_img_idx[idx % self.render_views] # 1장
        src_idx = self.test_img_idx[:]                          # 5장

        sample['scale_mat'] = torch.from_numpy(self.scale_mat)
        sample['trans_mat'] = torch.from_numpy(np.linalg.inv(self.ref_w2c))
        sample['extrinsic_render_view'] = torch.from_numpy(self.all_render_w2cs_original[render_idx])
        
        sample['w2cs'] = self.scaled_w2cs  # (V, 4, 4)
        sample['intrinsics'] = self.scaled_intrinsics[:, :3, :3]  # (V, 3, 3)
        sample['intrinsic_render_view'] = sample['intrinsics'][render_idx]

        sample['proj_matrices'] = self.proj_matrices_ms
        depth_max = self.depth_interval * self.ndepths + self.depth_min
        depth_values_org_scale = np.arange(self.depth_min, depth_max, self.depth_interval, dtype=np.float32)
        sample['depth_values_org_scale'] = depth_values_org_scale
        
        sample['near_fars'] = self.scaled_near_fars  # (V, 2)
        sample['ref_img'] = self.all_images[render_idx]
        sample['source_imgs'] = self.all_images[src_idx]
        sample['scale_factor'] = self.scale_factor

        intrinsics_pad = repeat(torch.eye(4), "X Y -> L X Y", L = len(sample['w2cs'])).clone()
        intrinsics_pad[:,:3,:3] = sample['intrinsics']

        
        sample['ref_pose']         = (intrinsics_pad @ self.scaled_render_w2cs)[render_idx]     # 4, 4
        sample['source_poses']     = (intrinsics_pad @ sample['w2cs'])[src_idx]

        # from 0~W to NDC's -1~1
        normalize_matrix = torch.tensor([[1/((self.img_W-1)/2), 0, -1, 0], [0, 1/((self.img_H-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]])

        sample['ref_pose'] = normalize_matrix @ sample['ref_pose']
        sample['source_poses'] = normalize_matrix @ sample['source_poses']

        sample['ref_pose_inv'] = torch.inverse(sample['ref_pose'])
        sample['source_poses_inv'] = torch.inverse(sample['source_poses'])

        sample['ray_o'] = sample['ref_pose_inv'][:3,-1]      # 3

        tmp_ray_d = (sample['ref_pose_inv'] @ self.homo_pixel)[:3] - sample['ray_o'][:,None]
        sample['ray_d'] = tmp_ray_d / torch.norm(tmp_ray_d, dim=0) # 3 120000
        sample['ray_d'] = sample['ray_d'].float()

        cam_ray_d = ((torch.inverse(normalize_matrix @ intrinsics_pad[0])) @ self.homo_pixel)[:3]
        cam_ray_d = cam_ray_d / torch.norm(cam_ray_d, dim=0)
        sample['cam_ray_d'] = cam_ray_d.float()

        # depths_mvs_h = self.all_mvs_depths * self.scale_factor
        
        sample['meta'] = "%s-%s-%08d"%(self.root_dir.split("/")[-1], self.scan_id, render_idx)

        sample['start_idx'] = 0
        return sample