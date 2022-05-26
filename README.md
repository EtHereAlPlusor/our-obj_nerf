# Our-NeRF

## Installation
1. Create a virtual environment via `conda`.
    ```
    conda create -n our-nerf python=3.9
    conda activate our-nerf
    ```
2. Install `torch` and `torchvision`.
    ```
    conda install pytorch torchvision torchaudio cudatoolkit=11.3 -c pytorch
    ```
3. Install requirements.
    ```
    pip install -r requirements.txt
    ```

## Data Preparation
1. We evaluate our model on [KITTI-360](http://www.cvlibs.net/datasets/kitti-360/). You can download it from [here](http://www.cvlibs.net/datasets/kitti-360/download.php) and modify the roots in `./configs/our-nerf.yaml` accordingly. Here we show the structure of a test dataset as follow.
    ```
    ├── KITTI-360
      ├── 2013_05_28_drive_0000_sync
        ├── image_00
        ├── image_01
      ├── calibration
        ├── calib_cam_to_pose.txt
        ├── perspective.txt
      ├── data_2d_semantics
        ├── train
          ├── 2013_05_28_drive_0000_sync
            ├── image_00
              ├── instance
      ├── data_3d_bboxes
      ├── data_poses
        ├── 2013_05_28_drive_0000_sync
            ├── cam0_to_world.txt
            ├── poses.txt
    ```

    | file | Intro |
    | ------ | ------ |
    | `image_00/01` | stereo RGB images |
    | `data_poses` | system poses in a global Euclidean coordinate |
    | `calibration` | extrinsics and intrinsics of the perspective cameras |
    | `instance` | instance label in single-channel 16-bit PNG format. Each pixel value denotes the corresponding instanceID. |

## Training and Visualization
1. We provide the training code. Use the following command to train your own model and show a novel view apperance of the scene and object branches. Every 1000 iterations will cost about 1.5 min on a single NVIDIA GeForce RTX™ 3090.
    ```
    python our-nerf.py --cfg_file configs/our-nerf.yaml
    ```
    <img src="./images/scene.png" width="90%">
    <img src="./images/object.png" width="90%">
2. Use the following command to visualize novel view appearance after scene editing. 
    ```
    python our-nerf.py --cfg_file configs/our-nerf.yaml is_editing True
    ```
   Or you can turn the cfg `is_editing` into `True` in the config file `./configs/our-nerf.yaml` and use the following command to finish the task.
    ```
    python our-nerf.py --cfg_file configs/our-nerf.yaml
    ```
    <img src="./images/scene_editing.png" width="90%">

## Citation
Copyright © 2022, Zhejiang University. All rights reserved. We favor any positive inquiry, please contact `3190102060@zju.edu.cn`.