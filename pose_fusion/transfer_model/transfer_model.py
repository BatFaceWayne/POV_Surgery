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
from pampjpe_err import get_alignMesh
from typing import Optional, Dict, Callable
import sys
import open3d as o3d
import pickle
import numpy as np
import torch
import torch.nn as nn
import trimesh

from tqdm import tqdm

from loguru import logger
from .utils import get_vertices_per_edge

from .optimizers import build_optimizer, minimize
from .utils import (
    Tensor, batch_rodrigues, apply_deformation_transfer)
from .losses import build_loss
import scipy.io as sio

def summary_closure(gt_vertices, var_dict, body_model, mask_ids=None):
    param_dict = {}
    for key, var in var_dict.items():
        # Decode the axis-angles
        if 'pose' in key or 'orient' in key:
            param_dict[key] = batch_rodrigues(
                var.reshape(-1, 3)).reshape(len(var), -1, 3, 3)
        else:
            # Simply pass the variable
            param_dict[key] = var
    body_model_output = body_model(
        return_full_pose=True, get_skin=True, **param_dict)
    est_vertices = body_model_output['vertices']
    if mask_ids is not None:
        est_vertices = est_vertices[:, mask_ids]
        gt_vertices = gt_vertices[:, mask_ids]

    v2v = (est_vertices - gt_vertices).pow(2).sum(dim=-1).sqrt().mean()
    return {
        'Vertex-to-Vertex': v2v * 1000}


def build_model_forward_closure(
    body_model: nn.Module,
    var_dict: Dict[str, Tensor],
    per_part: bool = True,
    part_key: Optional[str] = None,
    jidx: Optional[int] = None,
    part: Optional[Tensor] = None
) -> Callable:
    if per_part:
        cond = part is not None and part_key is not None and jidx is not None
        assert cond, (
            'When per-part is True, "part", "part_key", "jidx" must not be'
            ' None.'
        )

        def model_forward():
            param_dict = {}
            for key, var in var_dict.items():
                if part_key == key:
                    param_dict[key] = batch_rodrigues(
                        var.reshape(-1, 3)).reshape(len(var), -1, 3, 3)
                    param_dict[key][:, jidx] = batch_rodrigues(
                        part.reshape(-1, 3)).reshape(-1, 3, 3)
                else:
                    # Decode the axis-angles
                    if 'pose' in key or 'orient' in key:
                        param_dict[key] = batch_rodrigues(
                            var.reshape(-1, 3)).reshape(len(var), -1, 3, 3)
                    else:
                        # Simply pass the variable
                        param_dict[key] = var

            return body_model(
                return_full_pose=True, get_skin=True, **param_dict)
    else:
        def model_forward():
            param_dict = {}
            for key, var in var_dict.items():
                # Decode the axis-angles
                if 'pose' in key or 'orient' in key:
                    param_dict[key] = batch_rodrigues(
                        var.reshape(-1, 3)).reshape(len(var), -1, 3, 3)
                else:
                    # Simply pass the variable
                    param_dict[key] = var

            return body_model(return_full_pose=True, get_skin=True,
                              **param_dict)
    return model_forward


def build_edge_closure(
    body_model: nn.Module,
    var_dict: Dict[str, Tensor],
    edge_loss: nn.Module,
    optimizer_dict,
    gt_vertices: Tensor,
    per_part: bool = True,
    part_key: Optional[str] = None,
    jidx: Optional[int] = None,
    part: Optional[Tensor] = None
) -> Callable:
    ''' Builds the closure for the edge objective
    '''
    optimizer = optimizer_dict['optimizer']
    create_graph = optimizer_dict['create_graph']

    if per_part:
        params_to_opt = [part]
    else:
        params_to_opt = [p for key, p in var_dict.items() if 'pose' in key]

    model_forward = build_model_forward_closure(
        body_model, var_dict, per_part=per_part, part_key=part_key,
        jidx=jidx, part=part)

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()

        body_model_output = model_forward()
        est_vertices = body_model_output['vertices']

        loss = edge_loss(est_vertices, gt_vertices)
        if backward:
            if create_graph:
                # Use this instead of .backward to avoid GPU memory leaks
                grads = torch.autograd.grad(
                    loss, params_to_opt, create_graph=True)
                torch.autograd.backward(
                    params_to_opt, grads, create_graph=True)
            else:
                loss.backward()

        return loss
    return closure


def build_vertex_closure(
    body_model: nn.Module,
    var_dict: Dict[str, Tensor],
    optimizer_dict,
    gt_vertices: Tensor,
    vertex_loss: nn.Module,
    mask_ids=None,
    per_part: bool = True,
    part_key: Optional[str] = None,
    jidx: Optional[int] = None,
    part: Optional[Tensor] = None,
    params_to_opt: Optional[Tensor] = None,
) -> Callable:
    ''' Builds the closure for the vertex objective
    '''
    optimizer = optimizer_dict['optimizer']
    create_graph = optimizer_dict['create_graph']

    model_forward = build_model_forward_closure(
        body_model, var_dict, per_part=per_part, part_key=part_key,
        jidx=jidx, part=part)

    if params_to_opt is None:
        params_to_opt = [p for key, p in var_dict.items()]

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()

        body_model_output = model_forward()
        est_vertices = body_model_output['vertices']

        loss = vertex_loss(
            est_vertices[:, mask_ids] if mask_ids is not None else
            est_vertices,
            gt_vertices[:, mask_ids] if mask_ids is not None else gt_vertices)
        if backward:
            if create_graph:
                # Use this instead of .backward to avoid GPU memory leaks
                grads = torch.autograd.grad(
                    loss, params_to_opt, create_graph=True)
                torch.autograd.backward(
                    params_to_opt, grads, create_graph=True)
            else:
                loss.backward()

        return loss
    return closure

def build_vertex_closure_rhand(
    body_model: nn.Module,
    var_dict: Dict[str, Tensor],
    var_dict_additional: Dict[str, Tensor],
    optimizer_dict,
    gt_vertices: Tensor,
    vertex_loss: nn.Module,
    mask_ids=None,
    per_part: bool = True,
    part_key: Optional[str] = None,
    jidx: Optional[int] = None,
    part: Optional[Tensor] = None,
    params_to_opt: Optional[Tensor] = None,
) -> Callable:
    ''' Builds the closure for the vertex objective
    '''
    optimizer = optimizer_dict['optimizer']
    create_graph = optimizer_dict['create_graph']

    model_forward = build_model_forward_closure(
        body_model, var_dict, per_part=per_part, part_key=part_key,
        jidx=jidx, part=part)

    if params_to_opt is None:
        params_to_opt = [p for key, p in var_dict.items()]

    def closure(backward=True):
        if backward:
            optimizer.zero_grad()

        body_model_output = model_forward()
        est_vertices = body_model_output['vertices']
        # gt_vertices = torch.bmm(gt_vertices,params_to_opt['R']) + params_to_opt['T'].tile(1,778,1)
        loss = vertex_loss(
            est_vertices[:, mask_ids.reshape(778)],
            torch.bmm(gt_vertices,params_to_opt[1]) + params_to_opt[2].tile(1,778,1))
        if backward:
            if create_graph:
                # Use this instead of .backward to avoid GPU memory leaks
                grads = torch.autograd.grad(
                    loss, params_to_opt, create_graph=True)
                torch.autograd.backward(
                    params_to_opt, grads, create_graph=True)
            else:
                loss.backward()

        return loss
    return closure


def get_variables(
    batch_size: int,
    body_model: nn.Module,
    dtype: torch.dtype = torch.float32
) -> Dict[str, Tensor]:
    var_dict = {}

    device = next(body_model.buffers()).device

    if (body_model.name() == 'SMPL' or body_model.name() == 'SMPL+H' or
            body_model.name() == 'SMPL-X'):
        var_dict.update({
            'transl': torch.zeros(
                [batch_size, 3], device=device, dtype=dtype),
            'global_orient': torch.zeros(
                [batch_size, 1, 3], device=device, dtype=dtype),
            'body_pose': torch.zeros(
                [batch_size, body_model.NUM_BODY_JOINTS, 3],
                device=device, dtype=dtype),
            'betas': torch.zeros([batch_size, body_model.num_betas],
                                 dtype=dtype, device=device),
        })

    if body_model.name() == 'SMPL+H' or body_model.name() == 'SMPL-X':
        var_dict.update(
            left_hand_pose=torch.zeros(
                [batch_size, body_model.NUM_HAND_JOINTS, 3], device=device,
                dtype=dtype),
            right_hand_pose=torch.zeros(
                [batch_size, body_model.NUM_HAND_JOINTS, 3], device=device,
                dtype=dtype),
        )

    if body_model.name() == 'SMPL-X':
        var_dict.update(
            jaw_pose=torch.zeros([batch_size, 1, 3],
                                 device=device, dtype=dtype),
            leye_pose=torch.zeros([batch_size, 1, 3],
                                  device=device, dtype=dtype),
            reye_pose=torch.zeros([batch_size, 1, 3],
                                  device=device, dtype=dtype),
            expression=torch.zeros(
                [batch_size, body_model.num_expression_coeffs],
                device=device, dtype=dtype),
        )
    # var_dict.update(
    #     additional_trans=torch.eye(
    #         4, device=device,
    #         dtype=dtype).unsqueeze(0).tile(batch_size,1,1),
    #
    # )
    # Toggle gradients to True
    for key, val in var_dict.items():
        val.requires_grad_(True)

    return var_dict


def get_variables_additional(
        batch_size: int,
        body_model: nn.Module,
        dtype: torch.dtype = torch.float32
) -> Dict[str, Tensor]:
    var_dict = {}

    device = next(body_model.buffers()).device



    var_dict.update(
        R=torch.eye(
            3, device=device,
            dtype=dtype).unsqueeze(0).tile(batch_size, 1, 1),
        T = torch.zeros(
        [batch_size, 1,3], device=device, dtype=dtype)

    )
    # Toggle gradients to True
    for key, val in var_dict.items():
        val.requires_grad_(True)

    return var_dict

def get_hand(seg_list):
    all_hand = []
    for i in range(len(seg_list)):
        ref_hand = trimesh.load(
            seg_list[i],
            process=False)
        S_1_original = np.expand_dims(np.array(ref_hand.vertices), 0)
        all_hand.append(S_1_original)
    all_hand = np.concatenate(all_hand,0)
    return all_hand


def run_fitting(
    exp_cfg,
    batch: Dict[str, Tensor],
    body_model: nn.Module,
    def_matrix: Tensor,
    mask_ids: Optional = None,
    segment_list = None
) -> Dict[str, Tensor]:
    ''' Runs fitting
    '''

    # all_path = batch['paths']
    vertices = batch['vertices']
    # faces = batch['faces']
    # frame_list = [int(temp.split('/')[-1].split('.')[0]) for temp in all_path]

    batch_size = len(vertices)
    dtype, device = vertices.dtype, vertices.device
    summary_steps = exp_cfg.get('summary_steps')
    interactive = exp_cfg.get('interactive')

    # Get the parameters from the model
    var_dict = get_variables(batch_size, body_model)
    additional_dict = get_variables_additional(batch_size, body_model)

    # Build the optimizer object for the current batch
    optim_cfg = exp_cfg.get('optim', {})

    def_vertices = vertices  # apply_deformation_transfer(def_matrix, vertices, faces)

    if mask_ids is None:
        f_sel = np.ones_like(body_model.faces[:, 0], dtype=np.bool_)
    else:
        f_per_v = [[] for _ in range(body_model.get_num_verts())]
        [f_per_v[vv].append(iff) for iff, ff in enumerate(body_model.faces)
         for vv in ff]
        f_sel = list(set(tuple(sum([f_per_v[vv] for vv in mask_ids], []))))
    vpe = get_vertices_per_edge(
        body_model.v_template.detach().cpu().numpy(), body_model.faces[f_sel])

    def log_closure():
        return summary_closure(def_vertices, var_dict, body_model,
                               mask_ids=mask_ids)

    edge_fitting_cfg = exp_cfg.get('edge_fitting', {})
    edge_loss = build_loss(type='vertex-edge', gt_edges=vpe, est_edges=vpe,
                           **edge_fitting_cfg)
    edge_loss = edge_loss.to(device=device)

    vertex_fitting_cfg = exp_cfg.get('vertex_fitting', {})
    vertex_loss = build_loss(**vertex_fitting_cfg)
    vertex_loss = vertex_loss.to(device=device)

    per_part = edge_fitting_cfg.get('per_part', True)
    logger.info(f'Per-part: {per_part}')
    # ref_hand_name = '/home/rui/projects/sp2_ws/GraspTTA/refined_subsamples/friem_subsample/00295/00000_Hand.ply'
    with open('../data/bodymodel/smplx_mano_flame_correspondences/MANO_SMPLX_vertex_ids.pkl', 'rb') as f:
        idxs_data = pickle.load(f)

    all_hand = get_hand(segment_list)

    # ref_hand = trimesh.load(
    #     ref_hand_name,
    #     process=False)
    # S_1_original = np.expand_dims(np.array(ref_hand.vertices),0)


    # re, S1_hat, S2 = get_alignMesh(S_1_original,S_2_orginal)
    # pcd1 = o3d.geometry.PointCloud()
    # pcd2 = o3d.geometry.PointCloud()
    # pcd1.points = o3d.utility.Vector3dVector(S1_hat[0])
    # pcd2.points = o3d.utility.Vector3dVector(S2[0])

    # Optimize edge-based loss to initialize pose


    if per_part:
        for key, var in tqdm(var_dict.items(), desc='Parts'):
            if 'pose' not in key:
                continue

            for jidx in tqdm(range(var.shape[1]), desc='Joints'):
                part = torch.zeros(
                    [batch_size, 3], dtype=dtype, device=device,
                    requires_grad=True)
                # Build the optimizer for the current part
                optimizer_dict = build_optimizer([part], optim_cfg)
                closure = build_edge_closure(
                    body_model, var_dict, edge_loss, optimizer_dict,
                    def_vertices, per_part=per_part, part_key=key, jidx=jidx,
                    part=part)

                minimize(optimizer_dict['optimizer'], closure,
                         params=[part],
                         summary_closure=log_closure,
                         summary_steps=summary_steps,
                         interactive=interactive,
                         **optim_cfg)
                with torch.no_grad():
                    var[:, jidx] = part
    else:
        optimizer_dict = build_optimizer(list(var_dict.values()), optim_cfg)
        closure = build_edge_closure(
            body_model, var_dict, edge_loss, optimizer_dict,
            def_vertices, per_part=per_part)

        minimize(optimizer_dict['optimizer'], closure,
                 params=var_dict.values(),
                 summary_closure=log_closure,
                 summary_steps=summary_steps,
                 interactive=interactive,
                 **optim_cfg)

    if 'translation' in var_dict:
        optimizer_dict = build_optimizer([var_dict['translation']], optim_cfg)
        closure = build_vertex_closure(
            body_model, var_dict,
            optimizer_dict,
            def_vertices,
            vertex_loss=vertex_loss,
            mask_ids=mask_ids,
            per_part=False,
            params_to_opt=[var_dict['translation']],
        )
        # Optimize translation
        minimize(optimizer_dict['optimizer'],
                 closure,
                 params=[var_dict['translation']],
                 summary_closure=log_closure,
                 summary_steps=summary_steps,
                 interactive=interactive,
                 **optim_cfg)

    #  Optimize all model parameters with vertex-based loss
    optimizer_dict = build_optimizer(list(var_dict.values()), optim_cfg)
    closure = build_vertex_closure(
        body_model, var_dict,
        optimizer_dict,
        def_vertices,
        vertex_loss=vertex_loss,
        per_part=False,
        mask_ids=mask_ids)
    minimize(optimizer_dict['optimizer'], closure,
             params=list(var_dict.values()),
             summary_closure=log_closure,
             summary_steps=summary_steps,
             interactive=interactive,
             **optim_cfg)
    # print(vert_loss)
    # transfer_mat_version_mano = sio.loadmat('/home/rui/projects/sp2_ws/GraspTTA/MANO_version.mat')
    optimizer_dict = build_optimizer([var_dict['right_hand_pose'],additional_dict['R'],additional_dict['T']], optim_cfg)
    all_hand = torch.from_numpy(all_hand).float().to(device)
    closure = build_vertex_closure_rhand(
        body_model, var_dict, additional_dict,
        optimizer_dict,
        all_hand,
        vertex_loss=vertex_loss,
        mask_ids=idxs_data['right_hand'],#[transfer_mat_version_mano['forward_transfer']-1],
        per_part=False,
        params_to_opt=[var_dict['right_hand_pose'],additional_dict['R'],additional_dict['T']],
    )
    minimize(optimizer_dict['optimizer'],
             closure,
             params=[var_dict['right_hand_pose'],additional_dict['R'],additional_dict['T']],
             summary_closure=log_closure,
             summary_steps=summary_steps,
             interactive=interactive,
             **optim_cfg)

    param_dict = {}
    for key, var in var_dict.items():
        # Decode the axis-angles
        if 'pose' in key or 'orient' in key:
            param_dict[key] = batch_rodrigues(
                var.reshape(-1, 3)).reshape(len(var), -1, 3, 3)
        else:
            # Simply pass the variable
            param_dict[key] = var
    param_dict['left_hand_pose'] = torch.zeros_like(param_dict['left_hand_pose']).to(device)
    body_model_output = body_model(
        return_full_pose=True, get_skin=True, **param_dict)
    # var_dict.update(body_model_output)

    var_dict['vertices'] = body_model_output.vertices.detach().float().cpu().numpy()
    var_dict['faces'] = body_model.faces
    var_dict['body_pose'] = var_dict['body_pose'].view(-1,1,63)
    var_dict['left_hand_pose'] = var_dict['left_hand_pose'].view(-1, 1,45)
    var_dict['right_hand_pose'] = var_dict['right_hand_pose'].view(-1, 1,45)

    del all_hand, param_dict, body_model_output, vertices
    torch.cuda.empty_cache()
    return var_dict, additional_dict
