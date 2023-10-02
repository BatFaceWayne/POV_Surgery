import os
import glob
import pickle
import os.path as osp
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
import torch
from smplx import build_layer
import scipy.io as sio
import trimesh
import pandas as pd
import numpy as np
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
base_path = '/media/rui/My Passport/egobody_release/smpl_camera_wearer_new'

model_path = '/home/rui/projects/sp2_ws/smplx_transfer/transfer_data/body_models'

finishedlist =os.listdir('/media/rui/My Passport/egobody_release/smpl_camera_wearer')
# traverse root directory, and list directories as dirs and files as files
for recording in os.listdir(base_path):
    if recording in finishedlist:
        continue
    try:
        df = pd.read_csv(os.path.join('/media/rui/My Passport/egobody_release', 'data_2022_info_final.csv'))  # todo
        recording_name_list = list(df['recording_name'])
        start_frame_list = list(df['start_frame'])
        end_frame_list = list(df['end_frame'])
        body_idx_fpv_list = list(df['body_idx_fpv'])
        gender_0_list = list(df['body_idx_0'])
        gender_1_list = list(df['body_idx_1'])

        body_idx_fpv_dict = dict(zip(recording_name_list, body_idx_fpv_list))
        gender_0_dict = dict(zip(recording_name_list, gender_0_list))
        gender_1_dict = dict(zip(recording_name_list, gender_1_list))
        start_frame_dict = dict(zip(recording_name_list, start_frame_list))
        end_frame_dict = dict(zip(recording_name_list, end_frame_list))

        ######## get body idx for camera wearer/second person
        interactee_idx = int(body_idx_fpv_dict[recording].split(' ')[0])
    except:
        df = pd.read_csv(os.path.join('/media/rui/My Passport/egobody_release', 'data_2021_info_final.csv'))  # todo
        recording_name_list = list(df['recording_name'])
        start_frame_list = list(df['start_frame'])
        end_frame_list = list(df['end_frame'])
        body_idx_fpv_list = list(df['body_idx_fpv'])
        gender_0_list = list(df['body_idx_0'])
        gender_1_list = list(df['body_idx_1'])

        body_idx_fpv_dict = dict(zip(recording_name_list, body_idx_fpv_list))
        gender_0_dict = dict(zip(recording_name_list, gender_0_list))
        gender_1_dict = dict(zip(recording_name_list, gender_1_list))
        start_frame_dict = dict(zip(recording_name_list, start_frame_list))
        end_frame_dict = dict(zip(recording_name_list, end_frame_list))

        ######## get body idx for camera wearer/second person
        interactee_idx = int(body_idx_fpv_dict[recording].split(' ')[0])

    camera_wearer_idx = 1 - interactee_idx
    ######### get gender for camera weearer/second person
    interactee_gender = body_idx_fpv_dict[recording].split(' ')[1]
    if camera_wearer_idx == 0:
        camera_wearer_gender = gender_0_dict[recording].split(' ')[1]
    elif camera_wearer_idx == 1:
        camera_wearer_gender = gender_1_dict[recording].split(' ')[1]
    interactee_gender = body_idx_fpv_dict[recording].split(' ')[1]
    # body_model_interatee = build_layer(model_path, model_type='smpl',
    #                          gender=interactee_gender, ext='pkl', use_compressed=False)
    # body_model_interatee = body_model_interatee.to(device)
    body_model_camera_wearer = build_layer(model_path, model_type='smpl',
                             gender=camera_wearer_gender, ext='pkl', use_compressed=False)
    body_model_camera_wearer = body_model_camera_wearer.to(device)
    recording_path = osp.join(base_path,recording)
    all_pkl_files = []
    for root, dirs, files in os.walk(recording_path):
        for file in files:
            if 'pkl' in file:
                all_pkl_files.append(osp.join(root,file))

    for i in tqdm(range(len(all_pkl_files))):
        this_old_name = all_pkl_files[i]
        new_dict = {}
        this_new_name = this_old_name.replace('smpl_camera_wearer_new','smpl_camera_wearer')
        old_dict = pickle.load(open(this_old_name,'rb'))
        new_dict = {}
        new_dict['body_pose'] = torch.tensor(R.from_rotvec(old_dict['body_pose'].reshape(23,3)).as_matrix()).float().unsqueeze(0).to(device)
        new_dict['global_orient'] = torch.tensor(R.from_rotvec(old_dict['global_orient']).as_matrix()).unsqueeze(
            0).float().to(device)
        new_dict['betas']= torch.tensor(old_dict['betas']).float().unsqueeze(0).to(device)
        output = body_model_camera_wearer(return_verts=True, **new_dict)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        body_corret = trimesh.load(
            '/media/rui/My Passport/egobody_release/debug_transfer_SMPLX/smpl_mesh/' + recording + '/camera_wearer/' + this_old_name.split('/')[-2] + '.ply',
            process=False)
        transl_vec = np.mean(np.asarray(body_corret.vertices), 0) - np.mean(vertices, 0)

        old_dict['transl'] = transl_vec

        os.makedirs(osp.dirname(this_new_name),exist_ok=True)
        with open(this_new_name, 'wb') as f:
            pickle.dump(old_dict, f)

