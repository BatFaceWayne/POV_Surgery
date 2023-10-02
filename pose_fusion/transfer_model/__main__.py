# -*- coding: utf-8 -*-

# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is
# holder of all proprietary rights on this computer program.
# You can only use this computer program if you have closed
# a license agreement with MPG or you get the right to use the computer
# program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and
# liable to prosecution.
#
# Copyright©2020 Max-Planck-Gesellschaft zur Förderung
# der Wissenschaften e.V. (MPG). acting on behalf of its Max Planck Institute
# for Intelligent Systems. All rights reserved.
#
# Contact: Vassilis Choutas, vassilis.choutas@tuebingen.mpg.de
import os

import os.path as osp
import pickle
import sys

import numpy as np
import open3d as o3d
import torch
import trimesh
from loguru import logger
from scipy.io import loadmat, savemat
from smplx import build_layer
from tqdm import tqdm

from .config import parse_args
from .data import build_dataloader
from .transfer_model import run_fitting
from .utils import read_deformation_transfer, np_mesh_to_o3d
import glob

def main() -> None:
    # os.environ['CUDA_LAUNCH_BLOCKING'] = "1"
    exp_cfg = parse_args()
    exp_cfg.batch_size = 500
    exp_cfg.body_model.gender = 'male'
    device = torch.device('cuda')
    if not torch.cuda.is_available():
        logger.error('CUDA is not available!')
        sys.exit(3)
    # ref_handle_name = '/home/rui/Downloads/disk_recon_handle.ply'
    # ref_whole_name = '/home/rui/Downloads/disk_recon.ply'
    # transformed_mesh_name = '/home/rui/projects/sp2_ws/GraspTTA/refined_subsamples/friem_subsample/00295/00000_Object.ply'

    refer_refence = '../grasp_refinement/refined_subsamples/diskplacer_subsamples/00001/'
    refer_refence_seq = refer_refence.replace('refined_subsamples', 'refined_subsamples_interp')


    transformed_mesh_name = glob.glob(osp.join(refer_refence, '*Object.ply'))[0]
    frames_info_mat = loadmat(osp.join(refer_refence_seq, 'interpolate.mat'))
    dict_to_save = {}

    base_source_dir = '/home/ray/Downloads/zju-ls-feng/output/smplx'

    # ref_handle = trimesh.load(
    #     ref_handle_name,
    #     process=False)
    #
    # ref_whole = trimesh.load(
    #     ref_whole_name,
    #     process=False)

    transformed_mesh = trimesh.load(transformed_mesh_name, process=False)
    # transformed_mesh_hat = get_alignMesh_as2(np.array(ref_handle.vertices),np.array(ref_whole.vertices),np.array(transformed_mesh.vertices))
    logger.remove()
    logger.add(lambda x: tqdm.write(x, end=''), level=exp_cfg.logger_level.upper(), colorize=True)
    exp_cfg.datasets.mesh_folder.data_folder = osp.join(base_source_dir, 'smpl_ply')
    frame_list = [int(temp.split('/')[-1].split('.')[0]) for temp in os.listdir(osp.join(base_source_dir, 'smpl_ply'))]
    output_folder = osp.expanduser(osp.expandvars(exp_cfg.output_folder))
    seg_30 = np.random.choice(frame_list, 29)
    seg_30.sort()
    # seg_30.append(max(frame_list))
    pointer = 0

    this_infer_num = 0
    this_local_index = 0
    all_refer_hand_path = []

    while this_local_index < len(frame_list):
        temp_i = frame_list[this_local_index]

        if temp_i < seg_30[pointer]:
            temp = frames_info_mat['save_source_list'][0][pointer]
            temp_path = osp.join(refer_refence, str(temp).zfill(5) + '_Hand.ply')
            all_refer_hand_path.append(temp_path)
            this_local_index = this_local_index + 1
        elif this_infer_num < frames_info_mat['interp_list'][0][pointer]:
            temp = frames_info_mat['save_source_list'][0][pointer]
            temp_t = frames_info_mat['save_target_list'][0][pointer]
            temp_path = osp.join(refer_refence_seq, str(temp).zfill(5) + '_' + str(temp_t).zfill(5) + '_' + str(
                this_infer_num) + '_Hand.ply')
            all_refer_hand_path.append(temp_path)
            this_infer_num = this_infer_num + 1
            this_local_index = this_local_index + 1
        elif temp_i > max(seg_30):
            temp = frames_info_mat['save_target_list'][0][pointer]
            temp_path = osp.join(refer_refence, str(temp).zfill(5) + '_Hand.ply')
            all_refer_hand_path.append(temp_path)
            this_local_index = this_local_index + 1

        else:
            this_infer_num = 0
            # if pointer
            pointer = pointer + 1

    savemat(os.path.join(base_source_dir, 'hand_per_frame.mat'), {'hand_path': all_refer_hand_path})
    all_refer_hand_path = np.array(all_refer_hand_path)
    logger.info(f'Saving output to: {output_folder}')
    os.makedirs(output_folder, exist_ok=True)

    model_path = exp_cfg.body_model.folder

    body_model = build_layer(model_path, **exp_cfg.body_model)
    logger.info(body_model)
    body_model = body_model.to(device=device)

    deformation_transfer_path = exp_cfg.get('deformation_transfer_path', '')
    def_matrix = read_deformation_transfer(deformation_transfer_path, device=device)

    # Read mask for valid vertex ids
    mask_ids_fname = osp.expandvars(exp_cfg.mask_ids_fname)
    mask_ids = None
    if osp.exists(mask_ids_fname):
        logger.info(f'Loading mask ids from: {mask_ids_fname}')
        mask_ids = np.load(mask_ids_fname)
        mask_ids = torch.from_numpy(mask_ids).to(device=device)
    else:
        logger.warning(f'Mask ids fname not found: {mask_ids_fname}')

    data_obj_dict = build_dataloader(exp_cfg)

    dataloader = data_obj_dict['dataloader']

    for ii, batch in enumerate(tqdm(dataloader)):
        for key in batch:
            if key == 'path':
                continue
            if torch.is_tensor(batch[key]):
                batch[key] = batch[key].to(device=device)
        paths = batch['paths']
        batch_frame_list = [int(temp.split('/')[-1].split('.')[0]) for temp in paths]
        seg_list = all_refer_hand_path[batch_frame_list]

        var_dict, additional_dict = run_fitting(exp_cfg, batch, body_model, def_matrix, mask_ids, segment_list=seg_list)

        for ii, path in enumerate(paths):
            out_dict = {}
            for key_i in var_dict.keys():
                if key_i == 'vertices' or key_i == 'faces':
                    continue
                if var_dict[key_i] is not None:
                    out_dict[key_i] = var_dict[key_i][ii].detach().cpu().numpy()
            addtional_RT = {}
            for key_i in additional_dict.keys():
                if key_i == 'R' or key_i == 'T':
                    if additional_dict[key_i] is not None:
                        addtional_RT[key_i] = additional_dict[key_i][ii].detach().cpu().numpy()
            # this_hand_temp = o3d.io.read_triangle_mesh(seg_list[ii])
            _, fname = osp.split(path)

            output_path_pkl = osp.join(base_source_dir, 'smplx_fitted_pkl', f'{osp.splitext(fname)[0]}.pkl')
            os.makedirs(osp.dirname(output_path_pkl), exist_ok=True)
            ################################################## change ###########################################################3
            with open(output_path_pkl, 'wb') as f:
                pickle.dump(out_dict, f)

            output_path_pkl_addtion = osp.join(base_source_dir, 's_additional_RT_pkl', f'{osp.splitext(fname)[0]}.pkl')
            os.makedirs(osp.dirname(output_path_pkl_addtion), exist_ok=True)
            with open(output_path_pkl_addtion, 'wb') as f:
                pickle.dump(addtional_RT, f)

            output_path = osp.join(base_source_dir, 'smplx_fitted_ply', f'{osp.splitext(fname)[0]}.ply')
            os.makedirs(osp.dirname(output_path), exist_ok=True)
            mesh = np_mesh_to_o3d(var_dict['vertices'][ii], var_dict['faces'])
            o3d.io.write_triangle_mesh(output_path, mesh)
            # this_hand_temp.vertices = o3d.utility.Vector3dVector(np.array(this_hand_temp.vertices) @
            #                                                      additional_dict['R'][ii].detach().cpu().numpy() + additional_dict['T'][ii].detach().cpu().numpy())
            transformed_mesh_shifted = np.array(transformed_mesh.vertices) @ additional_dict['R'][
                ii].detach().cpu().numpy() + additional_dict['T'][ii].detach().cpu().numpy()
            # output_path = osp.join(
            #     '/home/rui/Downloads/zju-ls-feng/output/smplx/smplx_fitted_ply_withObject', f'{osp.splitext(fname)[0]}.ply')
            # os.makedirs(osp.dirname(output_path), exist_ok=True)

            mesh1 = o3d.geometry.TriangleMesh()
            mesh1.vertices = o3d.utility.Vector3dVector(transformed_mesh_shifted)
            mesh1.triangles = o3d.utility.Vector3iVector(transformed_mesh.faces)
            mesh += mesh1
            # output_path = osp.join(
            #     base_source_dir,'smplx_fitted_ply_withObject', f'{osp.splitext(fname)[0]}.ply')
            # os.makedirs(osp.dirname(output_path), exist_ok=True)
            # o3d.io.write_triangle_mesh(output_path, mesh)
            # output_path = osp.join(
            #     base_source_dir,'s_hand_save', f'{osp.splitext(fname)[0]}.ply')
            # os.makedirs(osp.dirname(output_path), exist_ok=True)
            # o3d.io.write_triangle_mesh(output_path, this_hand_temp)

            output_path = osp.join(base_source_dir, 'smplx_fitted_ply_Object', f'{osp.splitext(fname)[0]}.ply')
            os.makedirs(osp.dirname(output_path), exist_ok=True)
            o3d.io.write_triangle_mesh(output_path, mesh1)
        del var_dict, additional_dict, batch
        torch.cuda.empty_cache()


if __name__ == '__main__':
    main()
