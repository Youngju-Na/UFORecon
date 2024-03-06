from pathlib import Path
import numpy as np
import os
from glob import glob
import cv2
import open3d as o3d
import torch
import trimesh
import trimesh
import json
from scipy.spatial.transform import Rotation, Slerp
from scipy.interpolate import interp1d
import sys
from code1.dataset.dtu_test_sparse import DtuFitSparse
from torch.utils.data import DataLoader
from rich.console import Console
from typing_extensions import Literal, assert_never


def read_cam_file(filename):
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
    
    depth_min = near
    depth_interval = float(lines[11].split()[1]) * 1.06

    return P, near, far
    
    
def glob_data(data_dir):
    data_paths = []
    data_paths.extend(glob(data_dir))
    data_paths = sorted(data_paths)
    return data_paths

def load(config_file):
    tmp_json = json.load(open(config_file))
    extrinsic = np.array(tmp_json["extrinsic"]).reshape(4, 4).T
    pose = np.linalg.inv(extrinsic)
    return pose    

def interpolate_trajectory(cameras, num_views: int = 300):
    """calculate interpolate path"""

    c2ws = np.stack(cameras.inverse().cpu().numpy())

    key_rots = Rotation.from_matrix(c2ws[:, :3, :3])
    key_times = list(range(len(c2ws)))
    slerp = Slerp(key_times, key_rots)
    interp = interp1d(key_times, c2ws[:, :3, 3], axis=0)
    render_c2ws = []
    for i in range(num_views):
        time = float(i) / num_views * (len(c2ws) - 1)
        cam_location = interp(time)
        cam_rot = slerp(time).as_matrix()
        c2w = np.eye(4)
        c2w[:3, :3] = cam_rot
        c2w[:3, 3] = cam_location
        render_c2ws.append(c2w)
    render_c2ws = torch.from_numpy(np.stack(render_c2ws, axis=0))
    
    return render_c2ws

def render_scan(scan_id, mesh, out_path):
    
    global data_dir
    
    instance_dir = os.path.join(data_dir, 'scan{0}'.format(scan_id))
    image_paths = glob_data(os.path.join(instance_dir, 'image', '*.png'))
    # n_images = len(image_paths)
    
    # create tmp camera pose file for open3d
    camera_config = Path("video_poses")
    camera_config.mkdir(exist_ok=True, parents=True)
    
    instance_dir = os.path.join(data_dir, 'scan{0}'.format(scan_id))
   
    dataset = DtuFitSparse(root_dir=data_dir, 
                        split="test", 
                        scan_id='scan%d'%scan_id, 
                        n_views=3,
                        set=0,
                        test_view_pair=[23, 24, 23])
    
    all_intrinsics = dataset.all_intrinsics #* [3, 4, 4]
    all_w2cs = dataset.all_render_w2cs_original #* [3, 4, 4]
    
    
    camera_path = interpolate_trajectory(cameras=all_w2cs[:, ...], num_views=240)
    n_images = len(camera_path)
    
    H, W = 640, 800

    # create tmp camera pose file for open3d
    camera_config = Path("video_poses")
    camera_config.mkdir(exist_ok=True, parents=True)

    for image_id in range(n_images):
        
        c2w = camera_path[image_id]
        w2c = np.linalg.inv(c2w)

        K = all_intrinsics[0].numpy().copy()
        # K[:2, :] *= 2.
     
        tmp_json = json.load(open('c1.json'))
        tmp_json["extrinsic"] = w2c.T.reshape(-1).tolist()
        
        tmp_json["intrinsic"]["intrinsic_matrix"] = K[:3,:3].T.reshape(-1).tolist()
        tmp_json["intrinsic"]["height"] = H 
        tmp_json["intrinsic"]["width"] = W 
        json.dump(tmp_json, open('video_poses/tmp%d.json'%(image_id), 'w'), indent=4)
    
    cmd = f"python render_trajectory_open3d.py {mesh} \"{out_path}\" {camera_config}"
    os.system(cmd)



scans = [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]
view_combinations = ['favorable', 'unfavorable']

data_dir = '/home/yourname/ssd/datasets/DTU_TEST'

for scan in scans:
    for vc in view_combinations:
        mesh_path = f'/home/yourname/ssd/UFORecon/render_files/uforecon_random/scan{scan}/{vc}/scan{scan}.ply'
        out_path = f'./rendering/uforecon_random/scan{str(scan)}/{vc}'

        print(out_path)
        Path(out_path).mkdir(exist_ok=True, parents=True)

        render_scan(scan, mesh_path, out_path)

