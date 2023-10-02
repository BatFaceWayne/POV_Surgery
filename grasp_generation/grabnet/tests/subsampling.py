# -*- coding: utf-8 -*-
#
# Copyright (C) 2019 Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG),
# acting on behalf of its Max Planck Institute for Intelligent Systems and the
# Max Planck Institute for Biological Cybernetics. All rights reserved.
#
# Max-Planck-Gesellschaft zur Förderung der Wissenschaften e.V. (MPG) is holder of all proprietary rights
# on this computer program. You can only use this computer program if you have closed a license agreement
# with MPG or you get the right to use the computer program from someone who is authorized to grant you that right.
# Any use of the computer program without a valid license is prohibited and liable to prosecution.
# Contact: ps-license@tuebingen.mpg.de
#
from scipy.spatial.transform import Rotation as R
import sys
import scipy.io as sio
import trimesh
# from
sys.path.append('.')
sys.path.append('..')
import numpy as np
import torch
import os, time
import argparse

import mano

from grabnet.tools.utils import euler
from grabnet.tools.cfg_parser import Config
from grabnet.tests.tester import Tester

# from psbody.mesh import Mesh, MeshViewers
from psbody.mesh.colors import name_to_rgb
from grabnet.tools.train_tools import point2point_signed
from grabnet.tools.utils import aa2rotmat
from grabnet.tools.utils import makepath
from grabnet.tools.utils import to_cpu
from grabnet.tools.vis_tools import points_to_spheres
import open3d as o3d
from grabnet.tools.meshviewer import Mesh, MeshViewer, points2sphere

from bps_torch.bps import bps_torch


def get_meshes(dorig, coarse_net, refine_net, rh_model, save=False, save_dir=None, mat_name=None, this_zgen = None, idx =None, this_global_r = None, this_transl = None):
    with torch.no_grad():

        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # drec_cnet, zgen = coarse_net.sample_poses(dorig['bps_object'])
        drec_cnet, zgen = coarse_net.sample_poses_near_prior(dorig['bps_object'],prior=this_zgen)
        verts_rh_gen_cnet = rh_model(**drec_cnet).vertices
        zgen = zgen.cpu().float().numpy()

        _, h2o, _ = point2point_signed(verts_rh_gen_cnet, dorig['verts_object'].to(device))
        drec_cnet['trans_rhand_f'] = torch.from_numpy(np.repeat(this_transl.reshape(1,3),n_samples,0)).to(device)

        # import ipdb
        # ipdb.set_trace()
        # np.repeat(this_global_r.reshape(1, 3), 50, 0)
        drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(torch.from_numpy(np.repeat(this_global_r.reshape(1, 3), n_samples, 0))).view(-1, 3, 3).to(device)

        # drec_cnet['trans_rhand_f'] = drec_cnet['transl']
        # drec_cnet['global_orient_rhand_rotmat_f'] = aa2rotmat(drec_cnet['global_orient']).view(-1, 3, 3)
        # R.from_rotvec(np.repeat(this_global_r.reshape(1, 3), 50, 0)).
        drec_cnet['fpose_rhand_rotmat_f'] = aa2rotmat(drec_cnet['hand_pose']).view(-1, 15, 3, 3)
        drec_cnet['verts_object'] = dorig['verts_object'].to(device)
        drec_cnet['h2o_dist'] = h2o.abs()

        drec_rnet = refine_net(**drec_cnet)
        save_dict = {'zgen': zgen, 'rotmat': dorig['rotmat']}
        # drec_rnet['global_orient'] = this_global_r
        # drec_rnet['transl'] = this_global_r
        for this_key in drec_rnet.keys():
            save_dict[this_key] = drec_rnet[this_key].detach().cpu().numpy()
        out_1 = rh_model(**drec_rnet)
        verts_rh_gen_rnet = out_1.vertices
        save_dict['joints'] = out_1.joints.cpu().numpy()
        save_dict['vert'] = out_1.vertices.cpu().numpy()
        makepath(os.path.join(save_dir,mat_name,str(idx).zfill(5)))
        sio.savemat(os.path.join(save_dir,mat_name,str(idx).zfill(5),'generate.mat'), save_dict)
        # verts_rh_gen_rnet = verts_rh_gen_cnet

        gen_meshes = []
        for cId in range(0, len(dorig['bps_object'])):
            try:
                obj_mesh = dorig['mesh_object'][cId]
            except:
                obj_mesh = points2sphere(points=to_cpu(dorig['verts_object'][cId]), radius=0.002,
                                         vc=name_to_rgb['yellow'])
                print('points ')

            hand_mesh_gen_rnet = Mesh(vertices=to_cpu(verts_rh_gen_rnet[cId]), faces=rh_model.faces, vc=[245, 191, 177])

            if 'rotmat' in dorig:
                rotmat = dorig['rotmat'][cId].T
                obj_mesh = obj_mesh.rotate_vertices(rotmat)
                hand_mesh_gen_rnet.rotate_vertices(rotmat)

            gen_meshes.append([obj_mesh, hand_mesh_gen_rnet])
            if True:
                save_path = os.path.join(save_dir,mat_name,str(idx).zfill(5))
                # makepath(save_path)
                mesh_1 = o3d.geometry.TriangleMesh()
                mesh_1.vertices = o3d.utility.Vector3dVector(hand_mesh_gen_rnet.vertices)
                mesh_1.triangles =o3d.utility.Vector3iVector(hand_mesh_gen_rnet.faces)
                o3d.io.write_triangle_mesh(os.path.join(save_path ,str(cId).zfill(5)+'_Hand.ply'), mesh_1)
                mesh_2 = o3d.geometry.TriangleMesh()
                mesh_2.vertices = o3d.utility.Vector3dVector(obj_mesh.vertices)
                mesh_2.triangles =o3d.utility.Vector3iVector(obj_mesh.faces)
                o3d.io.write_triangle_mesh(os.path.join(save_path ,str(cId).zfill(5)+'_Object.ply'), mesh_2)

                # print(save_path + '/rh_mesh_gen_%d.obj' % cId)
                # hand_mesh_gen_rnet.export(filename=save_path + '/rh_mesh_gen_%d.ply' % cId)
                # obj_mesh.export(filename=save_path + '/obj_mesh_%d.ply' % cId)

        return gen_meshes


def grab_new_objs(grabnet, objs_path, rot=True, n_samples=10, scale=1., save_name='meshes', idx=0, this_rot = None, Z_GEN=None, this_global_r = None, this_transl = None):
    grabnet.coarse_net.eval()
    grabnet.refine_net.eval()

    rh_model = mano.load(model_path=grabnet.cfg.rhm_path,
                         model_type='mano',
                         num_pca_comps=45,
                         batch_size=n_samples,
                         flat_hand_mean=True).to(grabnet.device)

    grabnet.refine_net.rhm_train = rh_model

    grabnet.logger(f'################# \n'
                   f'Grabbing the object!'
                   )

    bps = bps_torch(custom_basis=grabnet.bps)

    if not isinstance(objs_path, list):
        objs_path = [objs_path]

    for new_obj in objs_path:

        # rand_rotdeg = np.random.random([n_samples, 3]) * np.array([360, 360, 360])

        # rand_rotmat = euler(rand_rotdeg)
        dorig = {'bps_object': [],
                 'verts_object': [],
                 'mesh_object': [],
                 'rotmat': []}

        for samples in range(n_samples):
            verts_obj, mesh_obj, rotmat = load_obj_verts(new_obj, this_rot, rndrotate=rot, scale=scale)

            bps_object = bps.encode(torch.tensor(verts_obj.astype(np.float32)), feature_type='dists')['dists']

            dorig['bps_object'].append(bps_object.to(grabnet.device))
            dorig['verts_object'].append(torch.from_numpy(verts_obj.astype(np.float32)).unsqueeze(0))
            dorig['mesh_object'].append(mesh_obj)
            dorig['rotmat'].append(rotmat)
            obj_name = os.path.basename(new_obj)

        dorig['bps_object'] = torch.cat(dorig['bps_object'])
        dorig['verts_object'] = torch.cat(dorig['verts_object'])

        save_dir = './samples_near'  # os.path.join(grabnet.cfg.work_dir, 'grab_new_objects')

        grabnet.logger(f'#################\n'
                       f'                   \n'
                       f'Saving results for the {obj_name.upper()}'
                       f'                      \n')

        gen_meshes = get_meshes(dorig=dorig,
                                coarse_net=grabnet.coarse_net,
                                refine_net=grabnet.refine_net,
                                rh_model=rh_model,
                                save=False,
                                save_dir=save_dir,
                                mat_name=save_name,
                                this_zgen = Z_GEN,
                                idx = idx,
                                this_global_r= this_global_r,
                                this_transl = this_transl
                                )

        torch.save(gen_meshes, os.path.join(save_dir,save_name,str(idx).zfill(5),'generate.pt') )


def load_obj_verts(mesh_path, rand_rotmat, rndrotate=True, scale=1., n_sample_verts=3000):
    np.random.seed(100)
    obj_mesh = Mesh(filename=mesh_path, vscale=scale)

    # if the object has no texture, make it yellow

    ## center and scale the object
    max_length = np.linalg.norm(obj_mesh.vertices, axis=1).max()
    if max_length > 1:
        re_scale = max_length / .08
        print(f'The object is very large, down-scaling by {re_scale} factor')
        obj_mesh.vertices[:] = obj_mesh.vertices / re_scale

    object_fullpts = obj_mesh.vertices
    maximum = object_fullpts.max(0, keepdims=True)
    minimum = object_fullpts.min(0, keepdims=True)

    offset = (maximum + minimum) / 2
    verts_obj = object_fullpts - offset
    obj_mesh.vertices[:] = verts_obj

    if rndrotate:
        obj_mesh.rotate_vertices(rand_rotmat)
    else:
        rand_rotmat = np.eye(3)
    #
    # while (obj_mesh.vertices.shape[0] < n_sample_verts):
    #     new_mesh = obj_mesh.subdivide()
    #     obj_mesh = Mesh(vertices=new_mesh.vertices,
    #                     faces=new_mesh.faces,
    #                     visual=new_mesh.visual)

    verts_obj = obj_mesh.vertices
    # verts_sample_id = np.random.choice(verts_obj.shape[0], n_sample_verts, replace=False)
    # verts_sampled = verts_obj[verts_sample_id]
    verts_sampled, _ = trimesh.sample.sample_surface_even(obj_mesh, n_sample_verts, radius=None)

    return verts_sampled, obj_mesh, rand_rotmat


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='GrabNet-Testing')

    parser.add_argument('--obj-path', required=True, type=str,
                        help='The path to the 3D object Mesh or Pointcloud')

    parser.add_argument('--rhm-path', required=True, type=str,
                        help='The path to the folder containing MANO_RIHGT model')

    parser.add_argument('--scale', default=1., type=float,
                        help='The scaling for the 3D object')

    parser.add_argument('--n-samples', default=10, type=int,
                        help='number of grasps to generate')

    parser.add_argument('--save_name', default='meshes', type=str,
                        help='name to save')

    args = parser.parse_args()



    # obj_path = args.obj_path
    rhm_path = args.rhm_path
    # scale = args.scale
    n_samples = 100


    cwd = os.getcwd()
    work_dir = cwd + '/logs'

    best_cnet = 'grabnet/models/coarsenet.pt'
    best_rnet = 'grabnet/models/refinenet.pt'
    bps_dir = 'grabnet/configs/bps.npz'

    source_dir = './OUT'
    obj_path = '/home/ray/Downloads/disk_recon_handle.ply'
    save_name = obj_path.split('/')[-1].split('.')[0]+'_subsamples'  # args.save_name
    # wanted_id = [16, 83, 58, 157, 436]  $ scalpel
    # wanted_id = [8, 295, 212, 193, 134]
    # wanted_id = [1,8,19,25,27,28,32,46,82,83,87,100,105,108,111,119,138,149,163,165,168,176,177,187,192,204,205,240,248,
    #                     250, 259, 261, 265, 276, 290, 293, 309, 313, 321, 341, 348, 359, 362, 364, 366, 390, 400, 402, 433, 439,
    #                     432, 454, 458, 475, 480, 499]
    wanted_id = [1,3]
    scale = 1
    info_mat_path = os.path.join(source_dir,'generate.mat')
    info_mat = sio.loadmat(info_mat_path)
    config = {
        'work_dir': work_dir,
        'best_cnet': best_cnet,
        'best_rnet': best_rnet,
        'bps_dir': bps_dir,
        'rhm_path': rhm_path

    }

    cfg = Config(**config)

    grabnet = Tester(cfg=cfg)
    for idx in wanted_id:
        this_rot = info_mat['rotmat'][idx]
        this_zgen = info_mat['zgen'][idx]
        this_global_r = info_mat['global_orient'][idx]
        this_transl = info_mat['transl'][idx]

        grab_new_objs(grabnet, obj_path, rot=True, n_samples=n_samples, scale=scale, save_name=save_name, idx=idx,Z_GEN=this_zgen,this_rot=this_rot, this_global_r=this_global_r, this_transl=this_transl)

