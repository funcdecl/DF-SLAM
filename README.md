## Official implementation of DF-SLAM: Dictionary Factors Representation for High-Fidelity Neural Implicit Dense Visual SLAM System
## Installation

First you have to make sure that you have all dependencies in place.
The simplest way to do so, is to use [anaconda](https://www.anaconda.com/). 

You can create an anaconda environment called `df_slam`. For linux, you need to install **libopenexr-dev** before creating the environment.
```bash
sudo apt-get install libopenexr-dev
    
conda env create -f environment.yaml
conda activate df_slam
```

## Run

### Replica
Download the data as below and the data is saved into the `./Datasets/Replica` folder.
```bash
bash scripts/download_replica.sh
```
and you can run DF_SLAM:
```bash
python -W ignore run.py configs/Replica/room0.yaml
```
The mesh for evaluation is saved as `$OUTPUT_FOLDER/mesh/final_mesh_eval_rec_culled.ply`, where the unseen and occluded regions are culled using all frames.


### ScanNet
Please follow the data downloading procedure on [ScanNet](http://www.scan-net.org/) website, and extract color/depth frames from the `.sens` file using this [code](https://github.com/ScanNet/ScanNet/blob/master/SensReader/python/reader.py).

<details>
  <summary>[Directory structure of ScanNet (click to expand)]</summary>
  
  DATAROOT is `./Datasets` by default. If a sequence (`sceneXXXX_XX`) is stored in other places, please change the `input_folder` path in the config file or in the command line.

```
  DATAROOT
  └── scannet
      └── scans
          └── scene0000_00
              └── frames
                  ├── color
                  │   ├── 0.jpg
                  │   ├── 1.jpg
                  │   ├── ...
                  │   └── ...
                  ├── depth
                  │   ├── 0.png
                  │   ├── 1.png
                  │   ├── ...
                  │   └── ...
                  ├── intrinsic
                  └── pose
                      ├── 0.txt
                      ├── 1.txt
                      ├── ...
                      └── ...

```
</details>

Once the data is downloaded and set up properly, you can run DF_SLAM:
```bash
python -W ignore run.py configs/ScanNet/scene0000.yaml
```
The final mesh is saved as `$OUTPUT_FOLDER/mesh/final_mesh_culled.ply`.

### TUM RGB-D
Download the data as below and the data is saved into the `./Datasets/TUM` folder.
```bash
bash scripts/download_tum.sh
```
and you can run DF_SLAM:
```bash
python -W ignore run.py configs/TUM_RGBD/freiburg1_desk.yaml
```
The final mesh is saved as `$OUTPUT_FOLDER/mesh/final_mesh_culled.ply`.

## Evaluation

### Average Trajectory Error
To evaluate the average trajectory error. Run the command below with the corresponding config file:
```bash
# An example for room0 of Replica
python src/tools/eval_ate.py configs/Replica/room0.yaml
```

### Reconstruction Error
To evaluate the reconstruction error, first download the ground truth Replica meshes and the files that determine the unseen regions.
```bash
bash scripts/download_replica_mesh.sh
```
Then run the `cull_mesh.py` with the following commands to exclude the unseen and occluded regions from evaluation.
```bash
# An example for room0 of Replica
# this code should create a culled mesh named 'room0_culled.ply'
GT_MESH=cull_replica_mesh/room0.ply
python src/tools/cull_mesh.py configs/Replica/room0.yaml --input_mesh $GT_MESH
```

Then run the command below. The 2D metric requires rendering of 1000 depth images, which will take some time. Use `-2d` to enable 2D metric. Use `-3d` to enable 3D metric.
```bash
# An example for room0 of Replica
OUTPUT_FOLDER=output/Replica/room0
GT_MESH=cull_replica_mesh/room0_culled.ply
python src/tools/eval_recon.py --rec_mesh $OUTPUT_FOLDER/mesh/final_mesh_eval_rec_culled.ply --gt_mesh $GT_MESH -2d -3d
```
