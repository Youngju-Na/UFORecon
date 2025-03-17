# UFORecon

import argparse
from re import I
from stat import UF_OPAQUE
from tqdm import tqdm
import math
import os
import os.path as path
import torch
from pytorch_lightning import loggers as pl_loggers
import sys
sys.path.append(path.dirname( path.dirname( path.abspath(__file__) ) ))

from torch.utils.data import DataLoader
from pytorch_lightning.loggers import WandbLogger
import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning.utilities.model_summary import ModelSummary
from pytorch_lightning.callbacks import ModelCheckpoint
from code1.model import UFORecon
from code1.dataset.dtu_train import MVSDataset
from code1.dataset.dtu_test_sparse import DtuFitSparse
from code1.dataset.general_fit import GeneralFit

import options

PI = math.pi
device = "cuda" if torch.cuda.is_available() else "cpu"

# -------------------------------- main function
if __name__ == "__main__":
    
    seed_everything(0, workers=True)    

    # -------------------------------- args for training and models ---------------------
    parser = argparse.ArgumentParser()

    parser.add_argument('--root_dir', dest='root_dir', type=str,
        help='directory of training dataset')

    #* training 
    parser.add_argument('--batch_size', dest='batch_size', type=int, default=2, help='batch size')
    parser.add_argument('--max_epochs', dest='max_epochs', type=int, default=16, help='max num of epochs')
    parser.add_argument('--val_only', dest='val_only', action="store_true", help='only validate')
    parser.add_argument('--uforecon_lr', dest='uforecon_lr', type=float, default=1.e-4, help='learning rate for uforecon')

    #* checkpoints
    parser.add_argument('--load_ckpt', dest='load_ckpt', type=str, default=False, help='load pretrained lightning ckpt')
    
    #* ray sampling
    parser.add_argument('--train_ray_num', dest='train_ray_num', type=int, default=1024, help='ray number in one image')
    parser.add_argument('--patch_size', dest='patch_size', type=int, default=48, help='patch size for training')
    parser.add_argument('--sW', type=int, default=1) #* change ray sampling stride
    parser.add_argument('--sH', type=int, default=1) #* change ray sampling stride
    parser.add_argument('--coarse_sample', dest='coarse_sample', type=int, default=64, help='number of coarse samples during training')
    parser.add_argument('--fine_sample', dest='fine_sample', type=int, default=64, help='number of fine samples during training')
    
    #* loss weights
    parser.add_argument('--weight_rgb', dest='weight_rgb', type=float, default=1.0)
    parser.add_argument('--weight_depth', dest='weight_depth', type=float, default=1.0)
    parser.add_argument('--logdir', default='./checkpoints/random_sample', help='the directory to save checkpoints/logs')

    # -------------------------------- args for testing --------------------------------
    parser.add_argument('--test_dir', dest='test_dir', type=str, help='directory of test dataset')
    parser.add_argument('--out_dir', dest='out_dir', type=str, help='directory of to save test result')
    parser.add_argument('--depth_dir', dest='depth_dir', type=str, help='directory of depth maps')
    parser.add_argument('--extract_geometry', dest='extract_geometry', action='store_true', help='if you only want to extract geometry')
    
    #* testing args
    parser.add_argument('--test_general', dest='test_general', action='store_true', help='test on custom dataset')
    parser.add_argument('--test_ray_num', dest='test_ray_num', type=int, default=1200)
    parser.add_argument('--test_sample_coarse', dest='test_sample_coarse', type=int, default=64)
    parser.add_argument('--test_sample_fine', dest='test_sample_fine', type=int, default=64)
    parser.add_argument('--test_coarse_only', dest='test_coarse_only', action="store_true", help='only use coarse samples during testing')
    parser.add_argument('--test_n_view', dest='test_n_view', type=int, default=3)
    parser.add_argument('--train_n_view', dest='train_n_view', type=int, default=5)
    parser.add_argument("--test_ref_view", type=int, nargs="+", default=[23, 24, 33, 22, 15, 34, 14, 32, 16, 35, 25])
    
    #* correlation modeling args
    parser.add_argument('--ndepths', type=str, default="48,32,8", help='ndepths')
    parser.add_argument('--depth_inter_r', type=str, default="4,2,1", help='depth_intervals_ratio')
    parser.add_argument('--grad_method', type=str, default="detach", choices=["detach", "undetach"], help='grad method')
    parser.add_argument('--share_cr', action='store_true', help='whether share the cost volume regularization')
    parser.add_argument('--cr_base_chs', type=str, default="8,8,8", help='cost regularization base channels')
    parser.add_argument('--numdepth', type=int, default=192, help='the number of depth values')
    
    #* ablation args
    parser.add_argument("--view_selection_type", type=str, default="random", choices=["random", "best"])
    parser.add_argument("--mvs_depth_guide", type=int, default=0, help='use mvs depth map as guidance')
    parser.add_argument("--volume_type", type=str, default="correlation", choices=["featuregrid", "correlation"])
    parser.add_argument('--volume_reso', dest='volume_reso', type=int, default=96, help="3D feature volume resolution") # set as 0 to disable
    parser.add_argument("--use_dir_srdf", action="store_true", help='use direction srdf')
    parser.add_argument("--depth_pos_encoding", action="store_true", help='use depth pos encoding')
    parser.add_argument("--explicit_similarity", action="store_true", help='use explicit similarity')
    parser.add_argument("--only_reference_frustum", action="store_true", help='use only the reference frustum view')
    
    parser.add_argument('--set', dest='set', type=int, default=0, help='two sets are provided by SparseNeuS')
    parser.add_argument('--debug', dest='debug', type=bool, default=False, help='debug mode')
    parser.add_argument('--test_scan', dest='test_scan', type=str, nargs="+", default=['5aa235f64a17b335eeaf9609', '5ba19a8a360c7c30c1c169df', '5adc6bd52430a05ecb2ffb85', '5bf7d63575c26f32dbf7413b'],)
    parser.add_argument('--dataset', dest='dataset', type=str, default='blendedmvs', help='dataset name')
    parser.add_argument('--use_mask', dest='use_mask', action='store_true', help='use mask')

    args = parser.parse_args()
    
    batch_size = args.batch_size
    num_workers = 1 if args.debug else 12
    devices = [0]
    
    
    #* load dataset
    if not args.extract_geometry:
        # training
        dtu_dataset_train = MVSDataset(            
                root_dir=args.root_dir,
                split="train",
                split_filepath="code1/dataset/dtu/lists/train.txt",
                pair_filepath="code1/dataset/dtu/dtu_pairs.txt",
                n_views=args.train_n_view,
                view_selection_type=args.view_selection_type,
                )

        dtu_dataset_val = MVSDataset(            
                root_dir=args.root_dir,
                split="test",
                split_filepath="code1/dataset/dtu/lists/test.txt",
                pair_filepath="code1/dataset/dtu/dtu_pairs.txt",
                n_views=args.test_n_view,
                test_ref_views = args.test_ref_view,  # only use view 23,
                view_selection_type=args.view_selection_type,
                )

        print("dtu_dataset_train:", len(dtu_dataset_train))
        print("dtu_dataset_val:", len(dtu_dataset_val))

        dataloader_train = DataLoader(dtu_dataset_train,
                                        batch_size=batch_size, 
                                        num_workers=num_workers, 
                                        shuffle=True) 
        dataloader_val = DataLoader(dtu_dataset_val,
                                        batch_size=batch_size, 
                                        num_workers=num_workers, 
                                        shuffle=False)  
    
    #! extract geometry
    else:
        dataloader_test = []
        # dtu, 15 test scenes
        if not args.test_general:
            for scan in [24, 37, 40, 55, 63, 65, 69, 83, 97, 105, 106, 110, 114, 118, 122]:
            # for scan in [65]:  
                dataset_tmp = DtuFitSparse(root_dir=args.test_dir, 
                                    split="test", 
                                    scan_id='scan%d'%scan, 
                                    n_views=args.test_n_view,
                                    set=args.set,
                                    test_view_pair=args.test_ref_view,
                                    depth_dir=args.depth_dir)
                dataloader_tmp = DataLoader(dataset_tmp,
                                                batch_size=1,
                                                num_workers=1,
                                                shuffle=False)
                dataloader_test.append(dataloader_tmp)
        else:
            # for scan in ['5aa235f64a17b335eeaf9609', '5ba19a8a360c7c30c1c169df', '5adc6bd52430a05ecb2ffb85', '5bf7d63575c26f32dbf7413b']: #* sculpture
            for scan in args.test_scan:
            # for scan in os.listdir(args.test_dir): #* ['general'] before
                dataset_tmp = GeneralFit(root_dir=args.test_dir,
                                    scan_id=scan,
                                    n_views=args.test_n_view,
                                    test_ref_view=args.test_ref_view, dataset=args.dataset, use_mask=args.use_mask)
                dataloader_tmp = DataLoader(dataset_tmp,
                                                batch_size=1, 
                                                num_workers=1, 
                                                shuffle=False)  
                dataloader_test.append(dataloader_tmp)

    # -------------------------------- lightning module -------------------------------
    
    
    print("---------------------------------------------------------------------------------------------")
    print("VIEW_SELECTION_TYPE:", args.view_selection_type, "MVS_DEPTH: ", args.mvs_depth_guide)
    print("---------------------------------------------------------------------------------------------")
    
    
    if args.load_ckpt:
        uforecon = UFORecon.load_from_checkpoint(checkpoint_path=args.load_ckpt, strict=True, args=args)
        print("Model loaded:", args.load_ckpt)
    else:
        uforecon = UFORecon(args)    
        
        
        

    tb_logger = pl_loggers.TensorBoardLogger("./%s" % args.logdir)

    checkpoint_callback = ModelCheckpoint(
        monitor='val/loss_depth_fine',
        dirpath=os.path.join(args.logdir, 'checkpoints'),
        filename='{epoch:02d}',
        save_top_k=15,
        mode='min',
    )

    # -------------------------------- trainer ---------------------------------------
    trainer = pl.Trainer(
        accelerator="gpu" if device=="cuda" else "cpu", 
        devices=devices,
        strategy = None,
        max_epochs=args.max_epochs,
        check_val_every_n_epoch=1, 
        logger=tb_logger,
        num_sanity_val_steps=0,
        callbacks=[checkpoint_callback],
        )
    
    
    print(ModelSummary(uforecon, max_depth=1))

    # -------------------------------- train or/and testing --------------------------------
    if not args.extract_geometry:
        if args.val_only:
            print("[only validation]")
            trainer.validate(uforecon, dataloader_train)
        else:
            print("[start training]")
            trainer.fit(uforecon, dataloader_train, dataloader_val)
    else:
        for dataloader_test1 in tqdm(dataloader_test):
            trainer.validate(uforecon, dataloader_test1) # model, dataloader 넣고 끝

    print("end")