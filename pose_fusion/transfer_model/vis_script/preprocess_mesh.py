import os
import sys
import open3d as o3d
import fnmatch
rootPath = '../'
sys.path.append(rootPath)
import scipy.io as sio
import os.path as osp
import cv2
import numpy as np
import json
import trimesh
import argparse
# os.environ["PYOPENGL_PLATFORM"] = "egl"
# os.environ["PYOPENGL_PLATFORM"] = "osmesa"
import pyrender
import PIL.Image as pil_img
import pickle
import smplx
import torch
from tqdm import tqdm
from os.path import basename
from utils import *
import pandas as pd

def main(args):
    out_path = '/media/rui/My Passport/egobody_release/debug_transfer_SMPLX/original_smplx'
    for recording_name in os.listdir('/media/rui/My Passport/egobody_release/smplx_camera_wearer'):
        args.recording_name = recording_name
        print('[INFO] recording_name:', args.recording_name)
        print('[INFO] view: ', args.view)
        record_save_base = os.path.join(out_path, recording_name)
        if not os.path.exists(record_save_base):
            os.mkdir(record_save_base)
        fitting_root_interactee = osp.join(args.release_data_root, 'smplx_interactee', args.recording_name)
        fitting_root_camera_wearer = osp.join(args.release_data_root, 'smplx_camera_wearer', args.recording_name)
        calib_trans_dir = os.path.join(args.release_data_root, 'calibrations', args.recording_name)  # extrinsics
        camera_params_dir = os.path.join(args.release_data_root, 'kinect_cam_params')  # intrinsics
        color_dir = osp.join(args.release_data_root, 'kinect_color', args.recording_name, args.view)

        ########## load calibration from sub kinect to main kinect (between color cameras)
        # master: kinect 12, sub_1: kinect 11, sub_2: kinect 13, sub_3, kinect 14, sub_4: kinect 15
        if args.view == 'sub_1':
            trans_subtomain_path = osp.join(calib_trans_dir, 'cal_trans', 'kinect_11to12_color.json')
        elif args.view == 'sub_2':
            trans_subtomain_path = osp.join(calib_trans_dir, 'cal_trans', 'kinect_13to12_color.json')
        elif args.view == 'sub_3':
            trans_subtomain_path = osp.join(calib_trans_dir, 'cal_trans', 'kinect_14to12_color.json')
        elif args.view == 'sub_4':
            trans_subtomain_path = osp.join(calib_trans_dir, 'cal_trans', 'kinect_15to12_color.json')
        if args.view != 'master':
            if not os.path.exists(trans_subtomain_path):
                print('[ERROR] {} does not have the view of {}!'.format(args.recording_name, args.view))
                exit()
            with open(osp.join(trans_subtomain_path), 'r') as f:
                trans_subtomain = np.asarray(json.load(f)['trans'])
                trans_maintosub = np.linalg.inv(trans_subtomain)


        ################################################ read body idx info
        try:
            df = pd.read_csv(os.path.join(args.release_data_root, 'data_2022_info_final.csv'))  # todo
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
            interactee_idx = int(body_idx_fpv_dict[args.recording_name].split(' ')[0])
        except:
            df = pd.read_csv(os.path.join(args.release_data_root, 'data_2021_info_final.csv'))  # todo
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
            interactee_idx = int(body_idx_fpv_dict[args.recording_name].split(' ')[0])

        camera_wearer_idx = 1 - interactee_idx
        ######### get gender for camera weearer/second person
        interactee_gender = body_idx_fpv_dict[args.recording_name].split(' ')[1]
        if camera_wearer_idx == 0:
            camera_wearer_gender = gender_0_dict[args.recording_name].split(' ')[1]
        elif camera_wearer_idx == 1:
            camera_wearer_gender = gender_1_dict[args.recording_name].split(' ')[1]

        ###########################################




        ######## create smplx body models
        model_interactee = smplx.create(args.model_folder, model_type='smplx', gender=interactee_gender, ext='npz', num_pca_comps=args.num_pca_comps,
                                        create_global_orient=True, create_transl=True, create_body_pose=True, create_betas=True,
                                        create_left_hand_pose=True, create_right_hand_pose=True,
                                        create_expression=True, create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True)

        model_camera_wearer = smplx.create(args.model_folder, model_type='smplx', gender=camera_wearer_gender, ext='npz', num_pca_comps=args.num_pca_comps,
                                           create_global_orient=True, create_transl=True, create_body_pose=True, create_betas=True,
                                           create_left_hand_pose=True, create_right_hand_pose=True,
                                           create_expression=True, create_jaw_pose=True, create_leye_pose=True, create_reye_pose=True)
        out_name_interactee_list = []
        out_name_camerawearer_list = []
        for i_frame in tqdm(range(start_frame_dict[args.recording_name], end_frame_dict[args.recording_name]+1)[args.start::args.step]):
            frame_id = 'frame_{}'.format("%05d"%i_frame)
            if not osp.exists(osp.join(fitting_root_interactee, 'body_idx_{}'.format(interactee_idx), 'results', frame_id, '000.pkl')):
                print('interactee fitting {} do not exist!'.format(frame_id))
                continue
            if not osp.exists(osp.join(fitting_root_camera_wearer, 'body_idx_{}'.format(camera_wearer_idx), 'results', frame_id, '000.pkl')):
                print('camera wearer fitting {} do not exist!'.format(frame_id))
                continue


            ##### read interactee smplx params
            out_name_interactee_list.append(osp.join(fitting_root_interactee, 'body_idx_{}'.format(interactee_idx), 'results', frame_id, '000.pkl'))
            with open(osp.join(fitting_root_interactee, 'body_idx_{}'.format(interactee_idx), 'results', frame_id, '000.pkl'), 'rb') as f:
                param = pickle.load(f)
            torch_param = {}
            for key in param.keys():
                if key in ['pose_embedding', 'camera_rotation', 'camera_translation', 'gender']:
                    continue
                else:
                    torch_param[key] = torch.tensor(param[key])

            torch_param['left_hand_pose'] = None
            torch_param['right_hand_pose'] = None
            torch_param['jaw_pose'] = None
            torch_param['leye_pose'] = None
            torch_param['reye_pose'] = None
            torch_param['expression'] = None

            output = model_interactee(return_verts=True, **torch_param)
            vertices = output.vertices.detach().cpu().numpy().squeeze()

            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(model_interactee.faces)
            if not os.path.exists(os.path.join(record_save_base, 'interactee')):
                os.mkdir(os.path.join(record_save_base, 'interactee'))

            o3d.io.write_triangle_mesh(os.path.join(record_save_base, 'interactee', str(i_frame).zfill(5) + '.ply'), mesh)
            # body = trimesh.Trimesh(vertices, model_interactee.faces, process=False)
            out_name_camerawearer_list.append(osp.join(fitting_root_camera_wearer, 'body_idx_{}'.format(camera_wearer_idx), 'results', frame_id, '000.pkl'))
            with open(osp.join(fitting_root_camera_wearer, 'body_idx_{}'.format(camera_wearer_idx), 'results', frame_id, '000.pkl'), 'rb') as f:
                param = pickle.load(f)
            torch_param = {}
            for key in param.keys():
                if key in ['pose_embedding', 'camera_rotation', 'camera_translation', 'gender']:
                    continue
                else:
                    torch_param[key] = torch.tensor(param[key])
            torch_param['left_hand_pose'] = None
            torch_param['right_hand_pose'] = None
            torch_param['jaw_pose'] = None
            torch_param['leye_pose'] = None
            torch_param['reye_pose'] = None
            torch_param['expression'] = None
            output = model_camera_wearer(return_verts=True, **torch_param)
            vertices = output.vertices.detach().cpu().numpy().squeeze()
            mesh = o3d.geometry.TriangleMesh()
            mesh.vertices = o3d.utility.Vector3dVector(vertices)
            mesh.triangles = o3d.utility.Vector3iVector(model_interactee.faces)
            if not os.path.exists(os.path.join(record_save_base, 'camera_wearer')):
                os.mkdir(os.path.join(record_save_base, 'camera_wearer'))

            o3d.io.write_triangle_mesh(os.path.join(record_save_base, 'camera_wearer', str(i_frame).zfill(5) + '.ply'),
                                       mesh)
        sio.savemat(osp.join(record_save_base,'info.mat'),{'out_name_camerawearer_list':out_name_camerawearer_list,
                                                           'out_name_interactee_list':out_name_interactee_list,
                                                           'interactee_gender':interactee_gender,
                                                           'camera_wearer_gender':camera_wearer_gender})


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--release_data_root', type=str, default='/media/rui/My Passport/egobody_release')
    parser.add_argument('--recording_name', type=str, default='recording_20210907_S02_S01_01')
    parser.add_argument('--view', type=str, default='master', choices=['master', 'sub_1', 'sub_2', 'sub_3', 'sub_4'])
    parser.add_argument('--scene_name', type=str, default='seminar_g110')

    parser.add_argument('--save_undistorted_img', default='False', type=lambda x: x.lower() in ['true', '1'], help='save undistorted input image or not')

    parser.add_argument('--scale', type=int, default=2, help='the scale to downsample output rendering images')
    parser.add_argument('--start', type=int, default=0, help='id of the starting frame')
    parser.add_argument('--step', type=int, default=1, help='id of the starting frame')
    parser.add_argument('--model_folder', default='/home/rui/projects/sp2_ws/smplx_transfer/transfer_data/body_models/', type=str, help='')
    parser.add_argument('--num_pca_comps', type=int, default=12,help='')
    parser.add_argument('--rendering_mode', default='3d', type=str, choices=['body', '3d', 'both'], help='')

    args = parser.parse_args()
    main(args)
