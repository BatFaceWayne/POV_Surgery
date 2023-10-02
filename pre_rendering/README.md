## Pre-rendering preparation
These module contains the utility fucntions required to render the synthetic egocentric imagees, as tranfsering SMPL-X body mesh to blender rendering coordinate system, preparing UV map for body model, and cauculating the egocentric(head mounted) camera trajectory.
### Usage
Go to the 'pre_rendering' folder:
```Shell
cd pre_rendering
```
#### Transfer SMPL-X body mesh to blender rendering coordinate system
```Shell
python transfer_pose.py
```
#### Prepare UV map for body model
```Shell
python prepare_uv.py
```
#### Calculate the egocentric camera trajectory
```Shell
python calculate_camera_trajectory.py
```