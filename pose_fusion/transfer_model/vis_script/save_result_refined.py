import os
import shutil

# base_refined_camera_wearer = '/home/rui/Downloads/finished_camera_wearer'
base_refined_interactee = '/home/rui/Downloads/finished_interatee'

# source_camera_wearer = '/home/rui/projects/virtual_human_ws/egobody_release/smpl_camera_wearer'
source_interatee = '/home/rui/projects/virtual_human_ws/egobody_release/smpl_interactee'

for this_camera_wearer_smpl in os.listdir(base_refined_interactee):
    if 'ply' in this_camera_wearer_smpl:
        continue
    this_rec = this_camera_wearer_smpl.split('^_^')[0]
    this_frame = this_camera_wearer_smpl.split('^_^')[1].split('.')[0]
    try:
        shutil.copyfile(os.path.join(base_refined_interactee,this_camera_wearer_smpl), os.path.join(source_interatee, this_rec,'body_idx_0','results',this_frame,'000.pkl'))
    except:
        shutil.copyfile(os.path.join(base_refined_interactee,this_camera_wearer_smpl), os.path.join(source_interatee, this_rec,'body_idx_1','results',this_frame,'000.pkl'))