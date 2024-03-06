# Adapted from SparseNeuS

from torch.utils.data import Dataset
import os, re
import numpy as np
import cv2
from PIL import Image
import torch
from torchvision import transforms as T
from termcolor import colored
from einops import repeat
import random
from .scene_transform import get_boundingbox
from torchvision.transforms.functional import resize
from torchvision.transforms.functional import pil_to_tensor
from torchvision.transforms import InterpolationMode

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

    out = cv2.decomposeProjectionMatrix(P)
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


class MVSDataset(Dataset):
    def __init__(self, root_dir, split, n_views=3, img_wh=(640, 512),
                 split_filepath=None, 
                 pair_filepath=None, 
                 N_rays=1024,
                 batch_size=1,
                 test_ref_views=[],
                 view_selection_type='best',
                 depth_fname='TransMVSNet',
                 ndepths=192):

        self.ndepths = ndepths
        self.root_dir = root_dir
        self.split = split

        self.img_wh = img_wh
        self.num_all_imgs = 49
        self.n_views = n_views 
        self.N_rays = N_rays
        self.batch_size = batch_size

        self.test_ref_views = test_ref_views  # used for validation
        self.scale_factor_diner = 0.7 / 872.
        self.downsample = 1.0
        self.scale_mat = np.float32(np.diag([1, 1, 1, 1.0]))
        self.view_selection_type = view_selection_type
        self.depth_fname = depth_fname
        
        if img_wh is not None:
            assert img_wh[0] % 32 == 0 and img_wh[1] % 32 == 0, \
                'img_wh must both be multiples of 32!'

        self.split_filepath = split_filepath
        self.pair_filepath = pair_filepath

        print(colored("loading all scenes together", 'red'))
        with open(self.split_filepath) as f:
            self.scans = [line.rstrip() for line in f.readlines()]

        self.all_intrinsics = []  # the cam info of the whole scene
        self.all_extrinsics = []
        self.all_near_fars = []

        self.metas, self.ref_src_pairs = self.build_metas()  # load ref-srcs view pairs info of the scene

        self.allview_ids = [i for i in range(self.num_all_imgs)]

        self.load_cam_info()  # load camera info of DTU, and estimate scale_mat

        self.build_remap() 
        self.define_transforms()
        self.conf2std = self._getconf2std()

        # * bounding box for rendering
        self.bbox_min = np.array([-1.0, -1.0, -1.0])
        self.bbox_max = np.array([1.0, 1.0, 1.0])
        
        #TODO: check what these lines are for 
        self.img_W, self.img_H = self.img_wh
        h_line = (np.linspace(0,self.img_H-1,self.img_H))*2/(self.img_H-1) - 1 # -1~1
        w_line = (np.linspace(0,self.img_W-1,self.img_W))*2/(self.img_W-1) - 1 # -1~1
        h_mesh, w_mesh = np.meshgrid(h_line, w_line, indexing='ij')
        self.w_mesh_flat = w_mesh.reshape(-1)
        self.h_mesh_flat = h_mesh.reshape(-1)
        self.homo_pixel = np.stack([self.w_mesh_flat, self.h_mesh_flat, np.ones(len(self.h_mesh_flat)), np.ones(len(self.h_mesh_flat))])  #[4,HW]


    def build_remap(self):
        self.remap = np.zeros(np.max(self.allview_ids) + 1).astype('int')
        for i, item in enumerate(self.allview_ids):
            self.remap[item] = i

    def _getconf2std(self):
        conf2std = lambda x: -2.5679e-2 * x + 3.2818e-2 #TODO: check this values
        return conf2std
    
    def define_transforms(self):
        self.transform = T.Compose([T.ToTensor()])

    
    def build_metas(self):
        """
        This function build metas 
        Returns:
            _type_:
        """
        
        metas = []
        ref_src_pairs = {} # referece view와 source view의 pair를 만든다.
        light_idxs = [3] if 'train' not in self.split else range(7)

        with open(self.pair_filepath) as f:
            num_viewpoint = int(f.readline())
            # viewpoints (49)
            for _ in range(num_viewpoint):
                ref_view = int(f.readline().rstrip())
                src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]

                ref_src_pairs[ref_view] = src_views

        for light_idx in light_idxs:
            for scan in self.scans:
                with open(self.pair_filepath) as f:
                    num_viewpoint = int(f.readline())
                    # viewpoints (49)
                    for _ in range(num_viewpoint):
                        ref_view = int(f.readline().rstrip())
                        
                        src_views = [int(x) for x in f.readline().rstrip().split()[1::2]]
                        #* implement random pair selection
                        if self.view_selection_type == 'random':
                            indices = [i for i in range(49) if i != ref_view]
                            src_views = random.sample(indices, self.n_views-1)
                        elif self.view_selection_type == 'best':
                            pass
                        else:
                            raise NotImplementedError
                        
                        # ! only for validation
                        if self.split != 'train' and len(self.test_ref_views) > 0:
                            if ref_view not in self.test_ref_views:
                                continue
                            else:
                                src_views = self.test_ref_views
                        
                        
                        metas += [(scan, light_idx, ref_view, src_views)] # scan, light_idx, ref_view, src_views

        return metas, ref_src_pairs


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
        # depth_min & depth_interval: line 11
        depth_min = float(lines[11].split()[0])
        depth_max = depth_min + float(lines[11].split()[1]) * 192
        
        self.depth_min = depth_min
        self.depth_interval = float(lines[11].split()[1]) * 1.06
        intrinsics_ = np.float32(np.diag([1, 1, 1, 1]))
        intrinsics_[:3, :3] = intrinsics

        return intrinsics_, extrinsics, [depth_min, depth_max]

    def load_cam_info(self):
        for vid in range(self.num_all_imgs):
            proj_mat_filename = os.path.join(self.root_dir,
                                             f'Cameras/train/{vid:08d}_cam.txt')
            intrinsic, extrinsic, near_far = self.read_cam_file(proj_mat_filename)
            intrinsic[:2] *= 4  # * the provided intrinsics is 4x downsampled, now keep the same scale with image
            self.all_intrinsics.append(intrinsic)
            self.all_extrinsics.append(extrinsic)
            self.all_near_fars.append(near_far)
        
        self.all_intrinsics_debug = self.all_intrinsics.copy()
        self.all_extrinsics_debug = self.all_extrinsics.copy()


    def read_depth(self, filename):
        depth_h = np.array(read_pfm(filename)[0], dtype=np.float32)  # (1200, 1600)
        depth_h = cv2.resize(depth_h, None, fx=0.5, fy=0.5,
                             interpolation=cv2.INTER_NEAREST)  # (600, 800)
        depth_h = depth_h[44:556, 80:720]  # (512, 640)
        return depth_h
    
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
        if self.split=='train':
            if h != 512:
                print("wrong size of depth map: ", filename)

        
        assert h == 512 and w == 640
        
        if self.downsample != 1:
            h, w = int(h * self.downsample), int(w * self.downsample)
            depth_h = resize(depth_h[None][None], (h, w), interpolation=InterpolationMode.NEAREST)[0, 0]
        mask = (depth_h > 0).float()
        depth_h *= self.scale_factor_diner

        depth_h = depth_h[None]  # (1, H, W)
        mask = mask[None]  # (1, H, W)

        return depth_h, mask

    def cal_scale_mat(self, img_hw, intrinsics, extrinsics, near_fars, factor=1.):
        center, radius, _ = get_boundingbox(img_hw, intrinsics, extrinsics, near_fars)

        radius = radius * factor
        scale_mat = np.diag([radius, radius, radius, 1.0])
        scale_mat[:3, 3] = center.cpu().numpy()
        scale_mat = scale_mat.astype(np.float32)

        return scale_mat, 1. / radius.cpu().numpy()

    
        
    def __len__(self):
        return len(self.metas)
    
    def get_depth_fname(self, cam_id):
        name = f"depth_map_{cam_id:04d}_{self.depth_fname}.png"
        return name

    def __getitem__(self, idx):
        sample = {}
        scan, light_idx, ref_view, src_views = self.metas[idx % len(self.metas)]

        if self.split == 'train':
            view_ids = [ref_view] + src_views[:self.n_views-1] # ref_view + n_views-1개의 src_view
        elif self.split == 'val' or self.split=='test':
            view_ids = [ref_view] + src_views
            
        w2c_ref = self.all_extrinsics[self.remap[ref_view]]
        w2c_ref_inv = np.linalg.inv(w2c_ref)

        imgs, depths_h, depths_mvs_h = [], [], []
        intrinsics, w2cs, near_fars = [], [], []  # record proj mats between views

        proj_matrices = []
        for i, vid in enumerate(view_ids):
            # NOTE that the id in image file names is from 1 to 49 (not 0~48)
            img_filename = os.path.join(self.root_dir,
                                        f'Rectified/{scan}_train/rect_{vid + 1:03d}_{light_idx}_r5000.png')
            depth_filename = os.path.join(self.root_dir,
                                          f'Depths_raw/{scan}/depth_map_{vid:04d}.pfm')
            
            # depth_mvs_filename = os.path.join(self.root_dir,
            #                                   f'Depths_raw/{scan}/{self.get_depth_fname(vid)}')
            
            # depth_std_filename = depth_mvs_filename.replace('.png', '_conf.png')

            # proj_mat_filename = os.path.join(self.root_dir, 'Cameras/train/{:0>8}_cam.txt').format(vid)
            
            # intrinsic, extrinsic, near_far = self.read_cam_file(proj_mat_filename)
            
            img = Image.open(img_filename)
            img = self.transform(img)
            imgs += [img]

            index_mat = self.remap[vid]
            near_fars.append(self.all_near_fars[index_mat])
            intrinsics.append(self.all_intrinsics[index_mat])
            w2cs.append(self.all_extrinsics[index_mat] @ w2c_ref_inv) #* reference view to source view
            

            if os.path.exists(depth_filename):  # and i == 0
                depth_h = self.read_depth(depth_filename)
                depths_h.append(depth_h)
                
                # if os.path.exists(depth_mvs_filename):
                #     depth_mvs_h = self.read_mvs_depth(depth_mvs_filename)[0] / self.scale_factor_diner
                #     depths_mvs_h.append(depth_mvs_h)
                #     depth_std = self.read_mvs_depth(depth_std_filename)[0]
                #     depths_std.append(depth_std)
                # else:
                #     raise ValueError
            
            if i == 0:
                depth_max = self.depth_interval * self.ndepths + self.depth_min
                depth_values_org_scale = np.arange(self.depth_min, depth_max, self.depth_interval, dtype=np.float32)
            
            
            #! added to make casmvsnet scaled version
            if i >= 1:
                intrinsic_cp = self.all_intrinsics[index_mat].copy()[:3, :3]
                intrinsic_cp[:2] /= 4 # 1/4->1/2->1
                proj_mat = np.zeros(shape=(2, 4, 4), dtype=np.float32)  #
                proj_mat[0, :4, :4] = self.all_extrinsics[index_mat]
                proj_mat[1, :3, :3] = intrinsic_cp
                proj_matrices.append(proj_mat)
            
        #ms proj_mats
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

        # depths_std = torch.stack(depths_std).float()
        # all_depths_std = self.conf2std(depths_std) #TODO check this function
        
        scale_mat, scale_factor = self.cal_scale_mat(img_hw=[self.img_wh[1], self.img_wh[0]],
                                                     intrinsics=intrinsics, extrinsics=w2cs,
                                                     near_fars=near_fars, factor=1.1)
        new_near_fars = []
        new_w2cs = []
        new_c2ws = []
        new_depths_h = []
        # new_depths_mvs_h = []
        
        
        for i, (intrinsic, extrinsic, depth) in enumerate(zip(intrinsics, w2cs, depths_h)):
        
            P = intrinsic @ extrinsic @ scale_mat # perspective matrix scaled by scale_mat
            P = P[:3, :4]
            c2w = load_K_Rt_from_P(None, P)[1] #* camera to world

            w2c = np.linalg.inv(c2w)
            new_w2cs.append(w2c)
            new_c2ws.append(c2w)

            
            
            camera_o = c2w[:3, 3] #* camera originf
            dist = np.sqrt(np.sum(camera_o ** 2))
            near = dist - 1
            far = dist + 1
            new_near_fars.append([0.95 * near, 1.05 * far])
            new_depths_h.append(depth * scale_factor)
            # new_depths_mvs_h.append(depth_mvs * scale_factor)


        imgs = torch.stack(imgs).float()
        depths_h = np.stack(new_depths_h)
        # depths_mvs_h = np.stack(new_depths_mvs_h)
        # depths_std = all_depths_std.numpy()
        
        intrinsics, w2cs, c2ws, near_fars = np.stack(intrinsics), np.stack(new_w2cs), np.stack(new_c2ws), np.stack(new_near_fars)
        start_idx = 0


        sample['images'] = imgs[start_idx:]  # (V, 3, H, W)
        sample['w2cs'] = torch.from_numpy(w2cs.astype(np.float32))[start_idx:]  # (V, 4, 4)
        sample['c2ws'] = torch.from_numpy(c2ws.astype(np.float32))[start_idx:]  # (V, 4, 4)
        sample['near_fars'] = torch.from_numpy(near_fars.astype(np.float32))[start_idx:]  # (V, 2)
        sample['intrinsics'] = torch.from_numpy(intrinsics.astype(np.float32))[start_idx:, :3, :3]  # (V, 3, 3)

        sample['meta'] = str(scan) + "_light" + str(light_idx) + "_refview" + str(ref_view)

        sample['scale_mat'] = torch.from_numpy(scale_mat)
        sample['trans_mat'] = torch.from_numpy(w2c_ref_inv) #* camera to world matrix of reference view

        sample['ref_img'] = sample['images'][0] # 3, 512, 640
        sample['source_imgs'] = sample['images'][1:] # 3, 3, 512, 640

        intrinsics_pad = repeat(torch.eye(4), "X Y -> L X Y", L = len(sample['w2cs'])).clone()
        intrinsics_pad[:,:3,:3] = sample['intrinsics']
        
        # extrinsics
        sample['ref_pose']         = (intrinsics_pad @ sample['w2cs'])[0]     # identity matrix for sample['w2cs'][0]
        sample['source_poses']     = (intrinsics_pad @ sample['w2cs'])[1:]    # 3, 4, 4
        
        # from 0~W to NDC's -1~1
        normalize_matrix = torch.tensor([[1/((self.img_W-1)/2), 0, -1, 0], [0, 1/((self.img_H-1)/2), -1, 0], [0,0,1,0], [0,0,0,1]])
        sample['ref_pose'] = normalize_matrix @ sample['ref_pose'] #* image to NDC
        sample['source_poses'] = normalize_matrix @ sample['source_poses']
        
        sample['ref_pose_inv'] = torch.inverse(sample['ref_pose']) #* ndc to reference camera coordinate
        sample['source_poses_inv'] = torch.inverse(sample['source_poses']) #* 
        
        sample['ray_o'] = sample['ref_pose_inv'][:3,-1]      # 3

        tmp_ray_d = (sample['ref_pose_inv'] @ self.homo_pixel)[:3] - sample['ray_o'][:,None]
        tmp_ray_d = tmp_ray_d / torch.linalg.norm(tmp_ray_d, dim=0, keepdim=True)
        sample['ray_d'] = tmp_ray_d #* defined in 

        cam_ray_d = (torch.inverse(normalize_matrix @ intrinsics_pad[0]) @ self.homo_pixel)[:3]
        cam_ray_d = cam_ray_d / torch.linalg.norm(cam_ray_d, dim=0, keepdim=True)
        sample['cam_ray_d'] = cam_ray_d

        depths_h = torch.from_numpy(depths_h.astype(np.float32))[start_idx:]
        # depths_mvs_h = torch.from_numpy(depths_mvs_h.astype(np.float32))[start_idx:]
        V,H,W = depths_h.size()
        depths_h = depths_h.view(V,-1)
        depths_h = depths_h/cam_ray_d[2:3,:]
        # depths_mvs_h = depths_mvs_h.view(V, -1)
        # depths_mvs_h = depths_mvs_h/cam_ray_d[2:3, :]
        
        #! transmvsnet
        sample['proj_matrices'] = proj_matrices_ms
        sample['depths_h'] = depths_h.view(V,H,W)
        # sample['depths_mvs_h'] = depths_mvs_h.view(V,H,W)
        # sample['depths_std'] = torch.from_numpy(depths_std)[start_idx:]
        sample['depth_values_org_scale'] = depth_values_org_scale
        
        sample['scale_factor'] = scale_factor
        
        
        return sample
