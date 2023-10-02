import os
import sys

rootPath = '../'
sys.path.append(rootPath)
from smplx import build_layer
from scipy.spatial.transform import Rotation as R

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
    print('[INFO] recording_name:', args.recording_name)
    print('[INFO] view: ', args.view)

    fitting_root_interactee = osp.join(args.release_data_root, 'smpl_interactee', args.recording_name)
    fitting_root_camera_wearer = osp.join(args.release_data_root, 'smpl_camera_wearer', args.recording_name)
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
    camera_wearer_idx = 1 - interactee_idx
    ######### get gender for camera weearer/second person
    interactee_gender = body_idx_fpv_dict[args.recording_name].split(' ')[1]
    if camera_wearer_idx == 0:
        camera_wearer_gender = gender_0_dict[args.recording_name].split(' ')[1]
    elif camera_wearer_idx == 1:
        camera_wearer_gender = gender_1_dict[args.recording_name].split(' ')[1]

    ###########################################
    rendering_dir = os.path.join(args.release_data_root, 'renderings_kinect_img', args.recording_name)
    body_scene_rendering_dir = os.path.join(args.release_data_root, 'renderings_kinect_3d', args.recording_name)

    if args.rendering_mode == '3d' or args.rendering_mode == 'both':
        scene_dir = os.path.join(os.path.join(args.release_data_root, 'scene_mesh_4render'), args.scene_name)
        static_scene = trimesh.load(osp.join(scene_dir, args.scene_name + '.obj'))
        cam2world_dir = os.path.join(calib_trans_dir, 'cal_trans/kinect12_to_world')  # transformation from master camera to scene mesh
        with open(os.path.join(cam2world_dir, args.scene_name + '.json'), 'r') as f:
            trans = np.array(json.load(f)['trans'])
        trans = np.linalg.inv(trans)
        static_scene.apply_transform(trans)

        body_scene_rendering_dir = os.path.join(args.release_data_root, 'renderings_kinect_3d')
        if not osp.exists(body_scene_rendering_dir):
            os.mkdir(body_scene_rendering_dir)
        body_scene_rendering_dir = os.path.join(body_scene_rendering_dir, args.recording_name)
        if not osp.exists(body_scene_rendering_dir):
            os.mkdir(body_scene_rendering_dir)
        body_scene_rendering_dir = os.path.join(body_scene_rendering_dir, args.view)
        if not osp.exists(body_scene_rendering_dir):
            os.mkdir(body_scene_rendering_dir)

    if args.rendering_mode == 'body' or args.rendering_mode == 'both':
        rendering_dir = os.path.join(args.release_data_root, 'renderings_kinect_img')
        if not osp.exists(rendering_dir):
            os.mkdir(rendering_dir)
        rendering_dir = os.path.join(rendering_dir, args.recording_name)
        if not osp.exists(rendering_dir):
            os.mkdir(rendering_dir)
        rendering_dir = os.path.join(rendering_dir, args.view)
        if not osp.exists(rendering_dir):
            os.mkdir(rendering_dir)

    ########## read kinect color camera intrinsics
    with open(osp.join(camera_params_dir, 'kinect_{}'.format(args.view), 'Color.json'), 'r') as f:
        color_cam = json.load(f)
    [f_x, f_y] = color_cam['f']
    [c_x, c_y] = color_cam['c']

    ########## create render camera
    camera_pose = np.eye(4)
    camera_pose = np.array([1.0, -1.0, -1.0, 1.0]).reshape(-1, 1) * camera_pose
    camera = pyrender.camera.IntrinsicsCamera(
        fx=f_x, fy=f_y,
        cx=c_x, cy=c_y)
    light = pyrender.DirectionalLight(color=np.ones(3), intensity=4.0)

    material_interactee = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(1.0, 193 / 255, 193 / 255, 1.0)  # pink, interactee
    )

    material_camera_wearer = pyrender.MetallicRoughnessMaterial(
        metallicFactor=0.0,
        alphaMode='OPAQUE',
        baseColorFactor=(70 / 255, 130 / 255, 180 / 255, 1.0)  # blue, camera_wearer
    )

    ######## create smplx body models
    model_path = '/home/rui/projects/sp2_ws/smplx_transfer/transfer_data/body_models'
    model_interactee = build_layer(model_path, model_type='smpl',
                             gender=interactee_gender, ext='pkl', use_compressed=False)
    model_camera_wearer = build_layer(model_path, model_type='smpl',
                             gender=interactee_gender, ext='pkl', use_compressed=False)



    for i_frame in tqdm(range(start_frame_dict[args.recording_name], end_frame_dict[args.recording_name]+1)[args.start::args.step]):
        frame_id = 'frame_{}'.format("%05d"%i_frame)
        if not osp.exists(osp.join(fitting_root_interactee, 'body_idx_{}'.format(interactee_idx), 'results', frame_id, '000.pkl')):
            print('interactee fitting {} do not exist!'.format(frame_id))
            continue
        if not osp.exists(osp.join(fitting_root_camera_wearer, 'body_idx_{}'.format(camera_wearer_idx), 'results', frame_id, '000.pkl')):
            print('camera wearer fitting {} do not exist!'.format(frame_id))
            continue
        if not osp.exists(osp.join(color_dir, frame_id + '.jpg')):
            print('view {}, kinect color image {} do not exist!'.format(args.view, frame_id))
            continue

        ##### read interactee smplx params
        with open(osp.join(fitting_root_interactee, 'body_idx_{}'.format(interactee_idx), 'results', frame_id, '000.pkl'), 'rb') as f:
            param = pickle.load(f)
        torch_param = {}
        for key in param.keys():
            if key in ['pose_embedding', 'camera_rotation', 'camera_translation', 'gender']:
                continue
            else:
                if key == 'betas' or key == 'transl':
                    torch_param[key] = torch.tensor(param[key]).float().unsqueeze(0)
                elif key == 'body_pose':
                    torch_param[key] = torch.tensor(
                        R.from_rotvec(param[key].reshape(23, 3)).as_matrix()).float().unsqueeze(0)
                else:
                    torch_param[key] = torch.tensor(R.from_rotvec(param[key]).as_matrix()).unsqueeze(0).float()

        output = model_interactee(return_verts=True, **torch_param)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        body = trimesh.Trimesh(vertices, model_interactee.faces, process=False)
        if args.view != 'master':
            body.apply_transform(trans_maintosub)
        body_mesh_interactee = pyrender.Mesh.from_trimesh(body, material=material_interactee)

        ##### read camera wearer smplx params
        with open(osp.join(fitting_root_camera_wearer, 'body_idx_{}'.format(camera_wearer_idx), 'results', frame_id, '000.pkl'), 'rb') as f:
            param = pickle.load(f)
        torch_param = {}
        for key in param.keys():
            if key in ['pose_embedding', 'camera_rotation', 'camera_translation', 'gender']:
                continue
            else:
                if key == 'betas' or key == 'transl':
                    torch_param[key] = torch.tensor(param[key]).float().unsqueeze(0)
                elif key == 'body_pose':
                    torch_param[key] = torch.tensor(
                        R.from_rotvec(param[key].reshape(23, 3)).as_matrix()).float().unsqueeze(0)
                else:
                    torch_param[key] = torch.tensor(R.from_rotvec(param[key]).as_matrix()).unsqueeze(0).float()

        output = model_camera_wearer(return_verts=True, **torch_param)
        vertices = output.vertices.detach().cpu().numpy().squeeze()
        body = trimesh.Trimesh(vertices, model_camera_wearer.faces, process=False)
        if args.view != 'master':
            body.apply_transform(trans_maintosub)
        body_mesh_camera_wearer = pyrender.Mesh.from_trimesh(body, material=material_camera_wearer)

        ###### render on undistorted color image
        if args.rendering_mode == 'body' or args.rendering_mode == 'both':
            img = cv2.imread(os.path.join(color_dir, frame_id + '.jpg'))[:, :, ::-1]
            H, W, _ = img.shape
            img_undistort = cv2.undistort(img.copy(),
                                          np.asarray(color_cam['camera_mtx']),
                                          np.asarray(color_cam['k']))

            scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                                   ambient_light=(0.3, 0.3, 0.3))
            scene.add(camera, pose=camera_pose)
            scene.add(light, pose=camera_pose)
            scene.add(body_mesh_interactee, 'body_mesh_interactee')
            scene.add(body_mesh_camera_wearer, 'body_mesh_camera_wearer')

            r = pyrender.OffscreenRenderer(viewport_width=W,
                                           viewport_height=H,
                                           point_size=1.0)
            color, _ = r.render(scene, flags=pyrender.RenderFlags.RGBA)
            color = color.astype(np.float32)

            valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
            input_img = img_undistort
            output_img = (color[:, :, :-1] * valid_mask +
                          (1 - valid_mask) * input_img)

            output_img = pil_img.fromarray((output_img).astype(np.uint8))
            output_img.convert('RGB')
            output_img = output_img.resize((int(W / 2), int(H / 2)))
            output_img.save(os.path.join(rendering_dir, frame_id + '.jpg'))

            if args.save_undistorted_img:
                img_undistort = pil_img.fromarray(img_undistort.astype(np.uint8))
                img_undistort = img_undistort.resize((int(W / args.scale), int(H / args.scale)))
                img_undistort.save(os.path.join(rendering_dir, 'input_' + frame_id + '.jpg'))

        ###### render in 3d scene
        if args.rendering_mode == '3d' or args.rendering_mode == 'both':
            static_scene_mesh = pyrender.Mesh.from_trimesh(static_scene)

            scene = pyrender.Scene()
            scene.add(camera, pose=camera_pose)
            scene.add(light, pose=camera_pose)
            scene.add(static_scene_mesh, 'mesh')
            scene.add(body_mesh_interactee, 'body_mesh_interactee')
            scene.add(body_mesh_camera_wearer, 'body_mesh_camera_wearer')

            r = pyrender.OffscreenRenderer(viewport_width=1920,
                                           viewport_height=1080)
            color, _ = r.render(scene)
            color = color.astype(np.float32) / 255.0
            img = pil_img.fromarray((color * 255).astype(np.uint8))
            img = img.resize((int(1920 / args.scale), int(1080 / args.scale)))
            img.save(os.path.join(body_scene_rendering_dir, frame_id + '.jpg'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--release_data_root', type=str, default='/media/rui/My Passport/egobody_release')
    parser.add_argument('--recording_name', type=str, default='recording_20220215_S21_S22_01')
    parser.add_argument('--view', type=str, default='master', choices=['master', 'sub_1', 'sub_2', 'sub_3', 'sub_4'])
    parser.add_argument('--scene_name', type=str, default='cnb_dlab_0215')

    parser.add_argument('--save_undistorted_img', default='False', type=lambda x: x.lower() in ['true', '1'], help='save undistorted input image or not')

    parser.add_argument('--scale', type=int, default=2, help='the scale to downsample output rendering images')
    parser.add_argument('--start', type=int, default=0, help='id of the starting frame')
    parser.add_argument('--step', type=int, default=1, help='id of the starting frame')
    parser.add_argument('--model_folder', default='/home/rui/projects/sp2_ws/smplx_transfer/transfer_data/body_models/', type=str, help='')
    parser.add_argument('--num_pca_comps', type=int, default=12,help='')
    parser.add_argument('--rendering_mode', default='body', type=str, choices=['body', '3d', 'both'], help='')

    args = parser.parse_args()
    main(args)
