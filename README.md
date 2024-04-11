# <img src="figs/free_ufo1.png" alt="UFORecon Logo" width="70"> UFORecon

This repository contains a official code of **UFORecon: Generalizable Sparse-View Surface Reconstruction from Arbitrary and Unfavorable Sets.** by Youngju Na, Woo Jae Kim, Kyu Beom Han, Suhyeon Ha, and Sung-Eui Yoon.

We will be presenting UFORecon at CVPR 2024 in Seattle. Check out the project page and paper!
### [Project Page](https://youngju-na.github.io/uforecon.github.io/) | [arXiv](https://arxiv.org/abs/2403.05086) 

## <img src="figs/free_ufo3.png" width="50"> Updates
- [x] release evaluation code
- [x] release pretrained weights

## <img src="figs/free_ufo3.png" width="50"> Overview
<p align="center">
  <table>
    <tr>
      <th align="center">VolRecon</th>
      <th align="center">Ours</th>
    </tr>
    <tr>
      <td align="center"><img src="./figs/scan65_volrecon.gif" alt="VolRecon" width="400"/></td>
      <td align="center"><img src="./figs/scan65_random.gif" alt="Ours(Random)" width="400"/></td>
    </tr>
  </table>
</p>


<!-- **Abstract:**
Generalizable neural implicit surface reconstruction aims to obtain an accurate underlying geometry given a limited number of multi-view images from unseen scenes. However, existing methods select only informative and relevant views using predefined scores for training and testing phases. This constraint renders the model impractical in real-world scenarios, where the availability of favorable combinations cannot always be ensured. We introduce and validate a view-combination score to indicate the effectiveness of the input view combination. We observe that previous methods output degenerate solutions under arbitrary and unfavorable sets. Building upon this finding, we propose UFORecon, a robust view-combination generalizable surface reconstruction framework. To achieve this, we apply cross-view matching transformers to model interactions between source images and build correlation frustums to capture global correlations. Additionally, we explicitly encode pairwise feature similarities as view-consistent priors. Our proposed framework significantly outperforms previous methods in terms of view-combination generalizability and also in the conventional generalizable protocol trained with favorable view-combinations. -->


## <img src="figs/free_ufo3.png" width="40">  Requirements

All experiments are tested on Ubuntu (20.04 or 22.04) with a single RTX 3090 or 4090. 
* python 3.10
* CUDA 11.x
```
conda create --name UFORecon python=3.10
conda activate UFORecon

pip install -r requirements.txt
```

## <img src="figs/free_ufo3.png" width="40"> Reproducing Sparse View Reconstruction on DTU

* Download pre-processed [DTU dataset following VolRecon](https://github.com/IVRL/VolRecon). The dataset is organized as follows:
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

### <img src="figs/free_ufo3.png" width="40"> Evaluation (unfavorable)


Set `DATASET` as the root directory of the dataset, set `OUT_DIR` as the directory to store the rendered depth maps. By default, 3 images (`--test_n_view 3`) of unfavorable camera configuration (cam id: 1, 16, 36) in the image set 0 (`--set 0`) are used for testing. 
For reproduction, use cam id: (23, 24, 33) for favorable set, and (1, 16, 36) for unfavorable set. All other combinations with different number of images are availble. 
```
bash script/eval_dtu_unfavorable.sh
bash script/tsdf_fusion.sh
bash script/clean_mesh.sh
bash script/eval_dtu.sh
```

* For quantitative evaluation, download [SampleSet](http://roboimagedata.compute.dtu.dk/?page_id=36) and [Points](http://roboimagedata.compute.dtu.dk/?page_id=36) from DTU's website. Unzip them and place `Points` folder in `SampleSet/MVS Data/`. The structure looks like:
```
SampleSet
├──MVS Data
      └──Points
```

## <img src="figs/free_ufo3.png" width="50"> Training UFORecon on DTU

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

Set `--view_selection_type` to `random` for applying random set training (default is `best` for training following the common protocol.)  

## <img src="figs/free_ufo3.png" width="50"> Citation 
If you find this project useful for your research, please cite: 

```
@misc{na2024uforecon,
      title={UFORecon: Generalizable Sparse-View Surface Reconstruction from Arbitrary and Unfavorable Sets}, 
      author={Youngju Na and Woo Jae Kim and Kyu Beom Han and Suhyeon Ha and Sung-Eui Yoon},
      journal={CVPR},
      year={2024}
}
```

## <img src="figs/free_ufo3.png" width="50">  Acknowledgement
This project is based on [VolRecon](https://github.com/IVRL/VolRecon), [TransMVSNet](https://github.com/megvii-research/TransMVSNet), and [MatchNeRF](https://github.com/donydchen/matchnerf).
Thanks for their amazing work.



