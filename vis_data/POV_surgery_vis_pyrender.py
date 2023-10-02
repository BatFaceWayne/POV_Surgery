
import pyrender
import torch
import cv2
from os.path import join, dirname
from tqdm import tqdm
import os
import trimesh
import pandas
import numpy as np
import mano
import pickle
from scipy.spatial.transform import Rotation as R





device = 'cuda'
batch_size = 1
###################################################################################
DARASET_ROOT = '/home/ray/code_release/pov_surgery_dataset/POV_Surgery_data'
INFO_SHEET_PATH = join(DARASET_ROOT,'POV_Surgery_info.csv')
MANO_PATH = '../data/bodymodel/mano/MANO_RIGHT.pkl'
REPRO_DIR = '/home/ray/code_release/pov_surgery_dataset/temp_repro_pyrender'
###################################################################################
info_sheet = pandas.read_csv(INFO_SHEET_PATH)
SCALPE_OFFSET = [0.04805371, 0 ,0]
DISKPLACER_OFFSET = [0, 0.34612157 ,0]
FRIEM_OFFSET = [0, 0.1145 ,0]





with torch.no_grad():
    rh_mano = mano.load(model_path=MANO_PATH,
                        model_type='mano',
                        num_pca_comps=45,
                        batch_size=1,
                        emissiveFactor=1,
                        flat_hand_mean=True).to(device)
rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).to(device)




material_hand = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=(255 / 255, 153 / 255, 255/255 ,1.0)  # hand
)
material_object = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=(51/255, 255/255, 255/255, 1.0)  # object
)



for i_seq, rec_name in enumerate(info_sheet['Sequence Name']):


    grasp_name, grasp_id = info_sheet['Grasp Sequence'][i_seq].split('/')


    dataset_name = info_sheet['OUT_seq'][i_seq]
    this_mano_dir = join(DARASET_ROOT,'annotation', dataset_name)

    this_repro_dir = join(REPRO_DIR, dataset_name)
    os.makedirs(this_repro_dir,exist_ok=True)

    COLOR_NAME = dataset_name


    for i in tqdm(range(1, 5000)):
        this_pkl = join(this_mano_dir, str(i).zfill(5) + '.pkl')
        if not os.path.exists(this_pkl):
            continue
        img_fn_cropped = join(this_repro_dir, str(i).zfill(5) + '.jpg')
        if os.path.exists(img_fn_cropped):
            continue
        frame_anno = pickle.load(open(this_pkl,'rb'))

        rot_unique = trimesh.transformations.rotation_matrix(
            np.radians(90), [1, 0, 0])
        rot_only = rot_unique[:3,:3]

        hand_dict = frame_anno['mano']
        for key_i in hand_dict.keys():
            hand_dict[key_i] = torch.from_numpy(hand_dict[key_i]).float().to(device)

        this_hand = rh_mano(**hand_dict)
        hand_vert = this_hand.vertices.detach().cpu().squeeze(0).numpy()



        transformed_mesh_shifted = hand_vert @ frame_anno['grab2world_R'] + frame_anno['grab2world_T']

        mesh_hand = trimesh.Trimesh(vertices=transformed_mesh_shifted,
                               faces=rh_mano.faces)
        mesh_object = trimesh.load(join(DARASET_ROOT,'tool_mesh', grasp_name.split('_')[0] +'.stl'))

        if 'diskplacer' in grasp_name:
            mesh_object.vertices = mesh_object.vertices * 0.001 - np.array(DISKPLACER_OFFSET)
        elif 'friem' in grasp_name:
            mesh_object.vertices = mesh_object.vertices * 0.001 - np.array(FRIEM_OFFSET)
        elif 'scalpel' in grasp_name:
            mesh_object.vertices = mesh_object.vertices * 0.001 - np.array(SCALPE_OFFSET)

        mesh_object.vertices = mesh_object.vertices @ frame_anno['base_object_rot'].T
        mesh_object.vertices = mesh_object.vertices @  frame_anno['grab2world_R'] + frame_anno['grab2world_T']

        this_rot = frame_anno['cam_rot']


        this_transl = frame_anno['cam_transl']

        if True:
            mesh1 = pyrender.Mesh.from_trimesh(mesh_hand, material=material_hand)
            mesh2 = pyrender.Mesh.from_trimesh(mesh_object, material=material_object)
            scene = pyrender.Scene()
            scene.add(mesh1)
            scene.add(mesh2)

            camera = pyrender.camera.IntrinsicsCamera(1198.4395, 1198.4395, 960, 175.2, znear=0.001, zfar=1000.0,
                                                      name=None)

            camera_pose = np.eye(4)
            camera_pose[:3, 3] = this_transl
            camera_pose[:3, :3] = this_rot

            scene.add(camera, pose=camera_pose)

            pl = pyrender.PointLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            sl = pyrender.SpotLight(color=[1.0, 1.0, 1.0], intensity=2.0, innerConeAngle=0.05, outerConeAngle=0.5)
            dl = pyrender.DirectionalLight(color=[1.0, 1.0, 1.0], intensity=2.0)
            scene.add(pl, camera_pose)
            scene.add(sl, camera_pose)
            scene.add(dl, camera_pose)
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
            image_1 = cv2.imread(join(DARASET_ROOT,'color', COLOR_NAME, str(i).zfill(5) + '.jpg'))

            color = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image_1)

            img_fn_cropped = join(this_repro_dir, str(i).zfill(5) + '.jpg')
            cv2.imwrite(img_fn_cropped, color)


