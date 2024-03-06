# UFORecon

> This repository contains a code of UFORecon. \
> The project is led by Youngju Na and Suhyeon Ha.

<!-- ## Abstract

We propose a view-combination generalizable sparse-view surface reconstruction method by modeling the correlations across pairs of input images. Generalizable neural implicit surface reconstruction aims to obtain an accurate underlying geometry given a limited number of multi-view images from unseen scenes. However, existing baseline methods select only fixed reconstruction favorable views using predefined view-selection scores for both the training and testing phases. This constraint leads to impracticality, as it is not always possible to guarantee the availability of favorable combinations in real-world scenarios. We observe that previous methods output degenerate solutions under arbitrary and challenging view sets. Building upon this finding, we propose UFORecon, a robust view-combination generalizable surface reconstruction framework. To this end, we apply cross-view matching transformers to model interactions between source images and build correlation frustums to capture global correlations. In addition, we explicitly encode pairwise feature similarities as robust geometric priors. Our proposed framework largely outperforms previous methods not only in view-combination generalizability but also in the existing generalizable protocol trained with favorable view-combinations. -->


<p align="center">
  <img src="https://github.com/YoungjuNa-KR/CS479-team9/assets/45136186/a2a3e86c-d268-43a5-9f12-e97a78bbcd6c" alt="UFORecon Logo" width="600">
</p>

### Requirements

* python 3.8
* CUDA 11.x

```
conda create --name UFORecon python=3.8 pip
conda activate UFORecon

pip install -r requirements.txt
```

## Reproducing Sparse View Reconstruction on DTU

* Download pre-processed [DTU dataset](https://drive.google.com/file/d/1cMGgIAWQKpNyu20ntPAjq3ZWtJ-HXyb4/view?usp=sharing). The dataset is organized as follows:
```
root_directory
├──cameras
    ├── 00000000_cam.txt
    ├── 00000001_cam.txt
    └── ...  
├──pair.txt
├──scan24
├──scan37
      ├── image               
      │   ├── 000000.png       
      │   ├── 000001.png       
      │   └── ...                
      └── mask                   
          ├── 000.png   
          ├── 001.png
          └── ...                
```

Camera file ``cam.txt`` stores the camera parameters, which include extrinsic, intrinsic, minimum depth, and depth range interval:
```
extrinsic
E00 E01 E02 E03
E10 E11 E12 E13
E20 E21 E22 E23
E30 E31 E32 E33

intrinsic
K00 K01 K02
K10 K11 K12
K20 K21 K22

DEPTH_MIN DEPTH_INTERVAL
```

``pair.txt `` stores the view selection result. For each reference image, 10 best source views are stored in the file:
```
TOTAL_IMAGE_NUM
IMAGE_ID0                       # index of reference image 0 
10 ID0 SCORE0 ID1 SCORE1 ...    # 10 best source images for reference image 0 
IMAGE_ID1                       # index of reference image 1
10 ID0 SCORE0 ID1 SCORE1 ...    # 10 best source images for reference image 1 
...
```

### Evaluation and Meshing
```
bash script/eval_dtu.sh
bash script/tsdf_fusion.sh
bash script/clean_mesh.sh
bash script/eval_dtu.sh
```
Set `DATASET` as the root directory of the dataset, set `OUT_DIR` as the directory to store the rendered depth maps. `CKPT_FILE` is the path of the checkpoint file (default as our model pretrained on DTU). Run `bash eval_dtu.sh` on GPU. By Default, 3 images (`--test_n_view 3`) in image set 0 (`--set 0`) are used for testing.  

* For quantitative evaluation, download [SampleSet](http://roboimagedata.compute.dtu.dk/?page_id=36) and [Points](http://roboimagedata.compute.dtu.dk/?page_id=36) from DTU's website. Unzip them and place `Points` folder in `SampleSet/MVS Data/`. The structure looks like:
```
SampleSet
├──MVS Data
      └──Points
```

## Training on DTU
* Dataset link is provided by [UFORecon](https://github.com/IVRL/UFORecon)
* Download pre-processed [DTU's training set](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view) and [Depths_raw](https://virutalbuy-public.oss-cn-hangzhou.aliyuncs.com/share/cascade-stereo/CasMVSNet/dtu_data/dtu_train_hr/Depths_raw.zip). Then organize the dataset as follows:

```
root_directory
├──Cameras
├──Rectified
└──Depths_raw
```
* Train the model by running `bash train_dtu.sh` on GPU.
```
bash script/train_dtu.sh
```

## Acknowledgement
This project is based on [UFORecon](https://github.com/IVRL/UFORecon), [TransMVSNet](https://github.com/megvii-research/TransMVSNet), and [MatchNeRF](https://github.com/donydchen/matchnerf).
Thanks for their amazing work.



