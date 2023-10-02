import pyrender
import numpy as np
import torch
import cv2
from os.path import join, dirname
from tqdm import tqdm
import scipy.io as sio
import trimesh
import sys
import os.path as os
import mano
from scipy.spatial.transform import Rotation as R
device = 'cuda'
batch_size = 1





with torch.no_grad():
    rh_mano = mano.load(model_path='/home/rui/projects/sp2_ws/GrabNet/mano/MANO_RIGHT.pkl',
                        model_type='mano',
                        num_pca_comps=45,
                        batch_size=1,
                        flat_hand_mean=True).to(device)
rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).to(device)

import pandas
import numpy as np
import pickle

material_hand = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=(255 / 255, 153 / 255, 255/255 ,1.0)  # pink, interactee
)

material_object = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=(255/255, 204/255, 153/255, 1.0)  # blue, camera_wearer
)

info_sheet = pandas.read_csv('/media/rui/mac_data/POV_surgery/info_batch_1.csv')

SCALPE_OFFSET = [0.04805371, 0 ,0]
DISKPLACER_OFFSET = [0, 0.34612157 ,0]
FRIEM_OFFSET = [0, 0.1145 ,0]

for i_seq, rec_name in enumerate(info_sheet['Sequence Name']):

    grasp_name, grasp_id = info_sheet['Grasp Sequence'][i_seq].split('/')
    rec_base_dir, this_rec_dir = rec_name.split('/')
    dataset_name = info_sheet['OUT_seq'][i_seq]
    this_mano_dir = join('/media/rui/mac_data/POV_surgery/temp_test/', dataset_name)
    os.makedirs(this_mano_dir,exist_ok=True)
    this_repro_dir = join('/media/rui/mac_data/POV_surgery/temp_test_rep/', dataset_name)
    os.makedirs(this_repro_dir,exist_ok=True)
    ROOT_DIR = '/media/rui/mac_data/Technopark_Recordings/'+ rec_base_dir +'/'+ this_rec_dir+ '/output/smplx'



    COLOR_NAME = dataset_name

    camera_dir = osp.join(ROOT_DIR, 'texture_rotate')
    camera_transl = sio.loadmat(join(camera_dir, 'transl_dict.mat'))
    camera_rot = sio.loadmat(join(camera_dir, 'rot_dict.mat'))
    hand_path = sio.loadmat(join(ROOT_DIR, 'hand_per_frame.mat'))['hand_path']
    hand_path = [temp.replace('_Hand.ply', '_MANO.pkl').strip() for temp in hand_path]
    camera_transl = camera_transl['transl_dict']

    camera_rot = camera_rot['rot_dict']
    BASE_DIR = join(ROOT_DIR, 'rotated_body_ply')
    inmat_this = '/home/rui/projects/sp2_ws/GrabNet/samples_near/'+grasp_name+'/' + grasp_id + '/generate.mat'
    inmat = sio.loadmat(inmat_this)
    this_rot_object_base = inmat['rotmat'][0]
    for i in tqdm(range(1, len(hand_path))):
        frame_anno = {}

        rot_unique = trimesh.transformations.rotation_matrix(
            np.radians(90), [1, 0, 0])
        rot_only = rot_unique[:3,:3]
        additional_dict = pickle.load(open(osp.join(ROOT_DIR,'s_additional_RT_pkl',str(i).zfill(5)+'.pkl'),'rb'))
        # indices = list(range(batch*batch_size, batch*batch_size+curr_batch_size))
        pb = pickle.load(open(hand_path[i], 'rb'))
        pc = pb.copy()

        for key_i in pb.keys():
            pb[key_i] = torch.from_numpy(pb[key_i]).float().to(device)

        this_hand = rh_mano(**pb)
        hand_vert = this_hand.vertices.detach().cpu().squeeze(0).numpy()

        transformed_mesh_shifted = hand_vert @ additional_dict['R'] @ rot_only.T + additional_dict['T'] @ rot_only.T

        mesh_hand = trimesh.Trimesh(vertices=transformed_mesh_shifted,
                               faces=rh_mano.faces)
        mesh_object = trimesh.load(join('/media/rui/mac_data/POV_surgery/tool_mesh', grasp_name.split('_')[0] +'.stl'))

        if 'diskplacer' in grasp_name:
            mesh_object.vertices = mesh_object.vertices * 0.001 - np.array(DISKPLACER_OFFSET)
        elif 'friem' in grasp_name:
            mesh_object.vertices = mesh_object.vertices * 0.001 - np.array(FRIEM_OFFSET)
        elif 'scalpel' in grasp_name:
            mesh_object.vertices = mesh_object.vertices * 0.001 - np.array(SCALPE_OFFSET)

        mesh_object.vertices = mesh_object.vertices @ this_rot_object_base.T
        mesh_object.vertices = mesh_object.vertices @  additional_dict['R'] @ rot_only.T + additional_dict['T'] @ rot_only.T

        this_rot = np.asarray(R.from_euler('xyz',camera_rot[i] / 180 * np.pi,degrees=False).as_matrix())
        ########
        frame_anno['mano'] = pc
        frame_anno['grab2world_R'] = additional_dict['R'] @ rot_only.T
        frame_anno['grab2world_T'] = additional_dict['T'] @ rot_only.T
        frame_anno['base_object_rot'] = this_rot_object_base
        frame_anno['cam_rot'] = this_rot # camera_rot[i]
        frame_anno['cam_transl'] = camera_transl[i]
        with open(join(this_mano_dir,str(i).zfill(5)+'.pkl'), 'wb') as f:
            pickle.dump(frame_anno, f)


        ########



        # mesh_hand.export('/home/rui/Downloads/smpl_sfdsdftest/debug_1000.ply')
        # mesh_object.vertices = mesh_object.vertices @  additional_dict['R'] + additional_dict['T']



        # mesh_hand.apply_transform(rot_unique)
        # mesh_object.apply_transform(rot_unique)


        # pred_vertices = pred_vertices - camera_transl[this_i]
        # pred_vertices = pred_vertices @ camera_rot[this_i]
        this_transl = camera_transl[i]

        if i%10 == 0:
            mesh1= pyrender.Mesh.from_trimesh(mesh_hand, material=material_hand)
            mesh2 = pyrender.Mesh.from_trimesh(mesh_object, material=material_object)
            scene = pyrender.Scene()
            scene.add(mesh1)
            scene.add(mesh2)
            # more_x = 10
            # camera = pyrender.PerspectiveCamera(yfov=np.pi / 3.0, aspectRatio=1.0)
            # camera = pyrender.camera.IntrinsicsCamera(1198.4395, 1198.4395, 1920/2, 175.2000, znear=0.05, zfar=100.0, name=None)
            # camera = pyrender.camera.IntrinsicsCamera(1198.4395 , 1198.4395, 960, 175.2 , znear=0.001, zfar=1000.0, name=None)
            camera = pyrender.camera.IntrinsicsCamera(1198.4395 , 1198.4395, 960, 175.2 , znear=0.001, zfar=1000.0, name=None)

            camera_pose = np.eye(4)
            camera_pose[:3, 3] = this_transl
            camera_pose[:3, :3] = this_rot

            scene.add(camera, pose=camera_pose)
            # light = pyrender.SpotLight(color=np.ones(3), intensity=3.0,
            #                            innerConeAngle=np.pi / 16.0,
            #                            outerConeAngle=np.pi / 6.0)
            # scene.add(light, pose=camera_pose)

            light = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2)
            light_pose = np.eye(4)

            light_pose[:3, 3] = np.array([0, -1, 1])
            scene.add(light, pose=light_pose)

            light_pose[:3, 3] = np.array([0, 1, 1])
            scene.add(light, pose=light_pose)

            light_pose[:3, 3] = np.array([1, 1, 2])
            scene.add(light, pose=light_pose)

            r = pyrender.OffscreenRenderer(1920, 1080)
            color, depth = r.render(scene)
            # color = color.astype(np.float32) / 255.0
            valid_mask = (depth > 0)[:, :, None]
            image_1 = cv2.imread('/media/rui/mac_data/POV_surgery/color/'+COLOR_NAME +'/' + str(i).zfill(5) + '.jpg')

            # image_1 = cv2.imread('/home/rui/projects/blender_ws/test_data/data/01/rgb/' + str(i).zfill(5) + '.jpg')
            color = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image_1)
            # color = (color[:, :, :3] * valid_mask * 0.5  + image_1[:, :, :3] * valid_mask * 0.5 + (1 - valid_mask) * image_1)

        # rendered_body_overlay_cropped = renderer(verts, camera_translation, img, overlay=True, my_transl =this_transl, my_rot = this_rot)
        # # rendered_body_overlay_cropped = renderer(verts, np.array([-100000,-10000,-10000]), img, overlay=True)
            img_fn_cropped = join(this_repro_dir, str(i).zfill(5)+ '.jpg')
            # os.makedirs(dirname(img_fn_cropped), exist_ok=True)
            cv2.imwrite(img_fn_cropped, color)




