################ CV2 repro style
import pyrender

import numpy as np
import torch

import cv2
from os.path import join, dirname
import os
from tqdm import tqdm



import trimesh

import matplotlib.pyplot as plt

import pandas
import numpy as np
import pickle
import mano
import pickle
from scipy.spatial.transform import Rotation as R
import scipy.io as sio
device = 'cuda'
batch_size = 1
TPID = [744,320,443,554,671]
###################################################################################
DARASET_ROOT = '/home/ray/code_release/pov_surgery_dataset/POV_Surgery_data'
INFO_SHEET_PATH = join(DARASET_ROOT,'POV_Surgery_info.csv')
MANO_PATH = '../data/bodymodel/mano/MANO_RIGHT.pkl'
REPRO_DIR = '/home/ray/code_release/pov_surgery_dataset/temp_repro_opencv'
###################################################################################
info_sheet = pandas.read_csv(INFO_SHEET_PATH)
SCALPE_OFFSET = [0.04805371, 0 ,0]
DISKPLACER_OFFSET = [0, 0.34612157 ,0]
FRIEM_OFFSET = [0, 0.1145 ,0]
def showObjJoints(image, kp, estIn=None, filename=None, upscale=1, lineThickness=3):
    '''
    Utility function for displaying object annotations
    :param imgIn: image on which annotation is shown
    :param gtIn: ground truth annotation
    :param estIn: estimated keypoints
    :param filename: dump image name
    :param upscale: scale factor
    :param lineThickness:
    :return:
    '''

    jointConns = [[0, 1, 3, 2, 0], [4, 5, 7, 6, 4], [0, 4], [1, 5], [2, 6], [3,7]]
    jointColsGt = (255,255,0)

    newCol = (jointColsGt[0] + jointColsGt[1] + jointColsGt[2]) / 3
    jointColsEst  = (0,255,0)
    color = np.ones(shape=(1080, 1920, 3), dtype=np.int16)
    color[:, :, 0] = image[:, :, 0]
    color[:, :, 1] = image[:, :, 1]
    color[:, :, 2] = image[:, :, 2]


    img = color.copy()
    index_p2d = np.ones((21, 3))
    index_p2d[:, 0] = kp[:, 1]
    index_p2d[:, 1] = kp[:, 0]
    est = index_p2d

    for i in range(len(jointConns)):
        for j in range(len(jointConns[i]) - 1):
            jntC = jointConns[i][j]
            jntN = jointConns[i][j + 1]
            cv2.line(img, (int(est[jntC,0]), int(est[jntC,1])), (int(est[jntN,0]), int(est[jntN,1])), jointColsEst, lineThickness)

    alpha = 0.7
    img_o = img*alpha + color*(1-alpha)
    return img_o

def vis_keypoints_with_skeleton(image, kp, fname='/home/rui/Downloads/inftest0.png'):
    color = np.ones(shape=(1080, 1920, 3), dtype=np.int16)
    color[:, :, 0] = image[:, :, 0]
    color[:, :, 1] = image[:, :, 1]
    color[:, :, 2] = image[:, :, 2]
    img = color
    # kp = kp[[0]]
    # kp[:, 0] = 512 - kp[:, 0]
    # kp[:, 1] = 512 - kp[:, 1]
    index_p2d = np.ones((21, 3))
    index_p2d[:, 0] = kp[:, 1]
    index_p2d[:, 1] = kp[:, 0]
    kps = index_p2d.T
    kp_thresh = 0.4
    alpha = 1
    # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
    kps_lines = [
        [0, 1],
        [1, 2],
        [2, 3],
        [3, 17],
        [0, 13],
        [13, 14],
        [14, 15],
        [15, 16],
        [0, 4],
        [4, 5],
        [5, 6],
        [6, 18],
        [0, 10],
        [10, 11],
        [11, 12],
        [12, 19],
        [0, 7],
        [7, 8],
        [8, 9],
        [9, 20]
    ]
    cmap = plt.get_cmap('rainbow')
    colors = [cmap(i) for i in np.linspace(0, 1, len(kps_lines) + 2)]
    colors = [(c[2] * 255, c[1] * 255, c[0] * 255) for c in colors]

    # Perform the drawing on a copy of the image, to allow for blending.
    kp_mask = np.copy(img)

    # Draw the keypoints.
    for l in range(len(kps_lines)):
        i1 = kps_lines[l][0]
        i2 = kps_lines[l][1]
        p1 = kps[0, i1].astype(np.int32), kps[1, i1].astype(np.int32)
        p2 = kps[0, i2].astype(np.int32), kps[1, i2].astype(np.int32)
        if kps[2, i1] > kp_thresh and kps[2, i2] > kp_thresh:
            cv2.line(
                kp_mask, p1, p2,
                color=colors[l], thickness=4, lineType=cv2.LINE_AA)
        if kps[2, i1] > kp_thresh:
            cv2.circle(
                kp_mask, p1,
                radius=6, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)
        if kps[2, i2] > kp_thresh:
            cv2.circle(
                kp_mask, p2,
                radius=6, color=colors[l], thickness=-1, lineType=cv2.LINE_AA)

    # Blend the keypoints.
    o_img = cv2.addWeighted(img, 1.0 - alpha, kp_mask, alpha, 0)
    # cv2.imwrite(fname, o_img)
    return o_img
    def vis_kp(self, image, kp, fname ='/home/rui/Downloads/inftest0.png'):

        # cv2.circle(img = color, (index_p2d[:, 0], index_p2d[:, 1]), 5, (0, 255, 0), -1)
        for temp_i in range(len(index_p2d[:, 0] )):
            cv2.circle(img=color, center=(index_p2d[temp_i,0], index_p2d[temp_i,1]), radius=5, color=(0, 255, 0), thickness=-1)

        # color[index_p2d[:, 1], index_p2d[:, 0], :] = 244

        cv2.imwrite(fname, color)

with torch.no_grad():
    rh_mano = mano.load(model_path=MANO_PATH,
                        model_type='mano',
                        num_pca_comps=45,
                        batch_size=1,
                        emissiveFactor=1,
                        flat_hand_mean=True).to(device)
rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).to(device)




material_interactee = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=(255 / 255, 153 / 255, 255/255 ,1.0)  # pink, interactee
)
material_camera_wearer = pyrender.MetallicRoughnessMaterial(
    metallicFactor=0.0,
    alphaMode='OPAQUE',
    baseColorFactor=(51/255, 255/255, 255/255, 1.0)  # blue, camera_wearer
)

import pickle

ALL_TRAIN = {}

for i_seq, rec_name in enumerate(info_sheet['Sequence Name']):

    grasp_name, grasp_id = info_sheet['Grasp Sequence'][i_seq].split('/')
    dataset_name = info_sheet['OUT_seq'][i_seq]
    i_start = info_sheet['start_frame'][i_seq]
    i_end = info_sheet['end_frame'][i_seq]
    this_mano_dir = join(DARASET_ROOT, 'annotation', dataset_name)
    this_repro_dir = join(REPRO_DIR, dataset_name)

    os.makedirs(this_repro_dir,exist_ok=True)




    COLOR_NAME = dataset_name


    for i in tqdm(range(i_start, i_end+1)):
        ALL_TRAIN[COLOR_NAME + '/' + str(i).zfill(5)] = {}

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
        hand_dict_save = frame_anno['mano'].copy()
        for key_i in hand_dict.keys():
            hand_dict[key_i] = torch.from_numpy(hand_dict[key_i]).float().to(device)

        this_hand = rh_mano(**hand_dict)
        hand_vert = this_hand.vertices.detach().cpu().squeeze(0).numpy()
        hand_kp_base = this_hand.joints.detach().cpu().float().numpy()[0]
        hand_finger = this_hand.vertices.detach().cpu().squeeze(0).float().numpy()[TPID,:]
        hand_kp = np.vstack((hand_kp_base, hand_finger))
        hand_kp = hand_kp @ frame_anno['grab2world_R'] + frame_anno['grab2world_T']


        transformed_mesh_shifted = hand_vert @ frame_anno['grab2world_R'] + frame_anno['grab2world_T']

        mesh_hand = trimesh.Trimesh(vertices=transformed_mesh_shifted,
                               faces=rh_mano.faces)


        mesh_object = trimesh.load(join(DARASET_ROOT, 'tool_mesh', grasp_name.split('_')[0] +'.stl'))
        control_point_mat = sio.loadmat(join(DARASET_ROOT,'tool_mesh','tool_control_points.mat'))


        if 'diskplacer' in grasp_name:
            mesh_object.vertices = mesh_object.vertices * 0.001 - np.array(DISKPLACER_OFFSET)
            tool_control_point = control_point_mat['diskplacer_kp']* 0.001 - np.array(DISKPLACER_OFFSET)
        elif 'friem' in grasp_name:
            mesh_object.vertices = mesh_object.vertices * 0.001 - np.array(FRIEM_OFFSET)
            tool_control_point = control_point_mat['friem_kp']* 0.001 - np.array(FRIEM_OFFSET)
        elif 'scalpel' in grasp_name:
            mesh_object.vertices = mesh_object.vertices * 0.001 - np.array(SCALPE_OFFSET)
            tool_control_point = control_point_mat['scalpel_kp']* 0.001 - np.array(SCALPE_OFFSET)

        mesh_object.vertices = mesh_object.vertices @ frame_anno['base_object_rot'].T
        mesh_object.vertices = mesh_object.vertices @  frame_anno['grab2world_R'] + frame_anno['grab2world_T']

        tool_control_point = tool_control_point @ frame_anno['base_object_rot'].T
        tool_control_point = tool_control_point @ frame_anno['grab2world_R'] + frame_anno['grab2world_T']


        this_rot = frame_anno['cam_rot']


        this_transl = frame_anno['cam_transl']

        if True :#i % 10 == 0:
            # temp_array = np.asarray(mesh_object.vertices)
            # temp_array[:,1] = temp_array[:,1]
            # temp_array[:, 2] =  temp_array[:, 2]
            # cur_pv2world_transform = holo_pv2world_trans_dict[holo_frame_id]
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = this_transl
            camera_pose[:3, :3] = this_rot
            # mesh_object.apply_transform(np.linalg.inv(camera_pose))
            # mesh_object.apply_transform(np.linalg.inv(camera_pose))
            hand_kp = hand_kp@ np.linalg.inv(camera_pose)[:3,:3].T + np.linalg.inv(camera_pose)[:3,3]

            tool_control_point = tool_control_point @ np.linalg.inv(camera_pose)[:3,:3].T + np.linalg.inv(camera_pose)[:3,3]
            # tool_control_point = tool_control_point @ np.linalg.inv(camera_pose)[:3,:3].T + np.linalg.inv(camera_pose)[:3,3]

            coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
            K = np.array([[1198.4395, 0.0000, 960.0000],[0.0000, 1198.4395, 175.2000],[0.0000, 0.0000, 1.0000]])

            temp_1 = (np.asarray(hand_kp).dot(coord_change_mat.T).dot(K.T)).T
            p2d = temp_1[0:2, :] / temp_1[[-1], :]

            temp_2 = (np.asarray(tool_control_point).dot(coord_change_mat.T).dot(K.T)).T
            new_p2d_object = temp_2[0:2, :] / temp_2[[-1], :]


            new_p2d = np.zeros_like(p2d.T)
            new_p2d[:,1] = np.clip(p2d[0,:] , 0, 1920 - 1)
            new_p2d[:, 0] =  np.clip(  p2d[1, :] , 0, 1080 - 1)

            new_p2d_object_ct = np.zeros_like(new_p2d_object.T)
            new_p2d_object_ct[:,1] = np.clip(new_p2d_object[0,:] , 0, 1920 - 1)
            new_p2d_object_ct[:, 0] =  np.clip(  new_p2d_object[1, :] , 0, 1080 - 1)

            image_size = [1920, 1080]
            ALL_TRAIN[COLOR_NAME + '/' + str(i).zfill(5)]['joints_uv'] = new_p2d
            ALL_TRAIN[COLOR_NAME + '/' + str(i).zfill(5)]['p2d'] = new_p2d_object_ct
            ALL_TRAIN[COLOR_NAME + '/' + str(i).zfill(5)]['frame_anno'] = this_pkl

            # image_1 = cv2.imread('/media/rui/Release_D/POV_surgery/color/'+COLOR_NAME +'/' + str(i).zfill(5) + '.jpg')
            image_1 = cv2.imread(join(DARASET_ROOT,'color', COLOR_NAME, str(i).zfill(5) + '.jpg'))
            index_p2d = new_p2d.astype(np.int32)
            index_p2d1 = new_p2d_object_ct.astype(np.int32)
            # color = image_1
            color = np.ones(shape=(1080, 1920, 3), dtype=np.int16)
            color[:, :, 0] = image_1[:, :, 0]
            color[:, :, 1] = image_1[:, :, 1]
            color[:, :, 2] = image_1[:, :, 2]
            color[ index_p2d[:,0]  ,index_p2d[:,1]] = 100
            color[index_p2d1[:, 0], index_p2d1[:, 1]] = 244

            color = vis_keypoints_with_skeleton(color, index_p2d)
            color = showObjJoints(color, index_p2d1)
            # for temp_i in range(len(index_p2d[:, 0])):
            #     cv2.circle(img=color, center=(index_p2d[temp_i, 1], index_p2d[temp_i, 0]), radius=5, color=(0, 255, 0),
            #                thickness=-1)
            # for temp_i in range(len(index_p2d1[:, 0])):
            #     cv2.circle(img=color, center=(index_p2d1[temp_i, 1], index_p2d1[temp_i, 0]), radius=5, color=(0, 255, 0),
            #                thickness=-1)
            # color = (color[:, :, :3] * valid_mask + (1 - valid_mask) * image_1)

            # color = (color[:, :, :3] * valid_mask * 0.5  + image_1[:, :, :3] * valid_mask * 0.5 + (1 - valid_mask) * image_1)

            img_fn_cropped = join(this_repro_dir, str(i).zfill(5)+ '.jpg')
            cv2.imwrite(img_fn_cropped, color)


            # imagePoints = np.clip(imagePoints,0,1920)

            # ax.plot([0,1920], [0,1920], 'r.')
            # ax.plot(p2d[0,:], p2d[1,:], 'r.')
            # num_points = points2D.shape[0]
            # for idasfa in range(len(mesh_object.vertices)):
            #     if p2d[0,idasfa]> 0 and p2d[0,idasfa] < 1920 and p2d[1,idasfa]>0 and p2d[1,idasfa]<1920:
            #         ax.plot([p2d[0,idasfa], p2d[1,idasfa]], color='r')

            # plt.show(block=False)
            # print(1)
            # 1
            # 1
