import numpy as np
import scipy.io as sio
import shutil
import os
mesh_source = '/media/rui/My Passport/egobody_release/debug_transfer_SMPLX/original_smplx'
save_dict = sio.loadmat('/home/rui/Downloads/issue_smplx2smpl.mat')
target_1  = '/home/rui/Downloads/pose_processed_camera_weaer'
os.makedirs(target_1,exist_ok=True)

issue_interactee = save_dict['issue_camera_wearer']
for frame in issue_interactee:
    this_seq, this_frame = frame.split('/')
    if False:
        shutil.copyfile(os.path.join(mesh_source,this_seq,'',this_frame+'.ply'), os.path.join(target_1,this_seq+'^_^'+this_frame+'.ply'))
    else:
        shutil.copyfile(os.path.join(mesh_source,this_seq,'camera_wearer',this_frame.split('_')[-1]+'.ply'), os.path.join(target_1,this_seq+'^_^'+this_frame+'.ply'))

# shutil.copyfile(src, dst)