## Hand-body pose fusion
### Pre-requisites
We recommend using existing motion capture methods as [EasyMocap](https://github.com/zju3dv/EasyMocap) or motion generation methods to generate the SMPL-X body pose sequences. The input of this module is a folder with a list of SMPL-X body pose sequences in .pkl format. The output is a folder with a list of SMPL-X body meshes, named as 'xxxxx.ply', where the xxxxx is the frame name. 
You could donwload a sample data here.

Moreover, You should compete the grasp generation and refienment steps, with keyposes and interpolated poses in 'refined_subsamples' and 'refined_subsamples_interp' folders, respectively.
### Usage
Go to the 'pose_fusion' folder:
```Shell
cd pose_fusion
```
Change the path in 'transfer_model/__main__.py', then run:
```Shell
python -m transfer_model --exp-cfg config_files/smplh2smplx.yaml
```
## Acknowledgments
 Large amount of code is borrowed from:

[[SMPL-X](https://smpl-x.is.tue.mpg.de)]