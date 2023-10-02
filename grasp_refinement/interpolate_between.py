import os
import sys

sys.path.append('.')
sys.path.append('..')
import time
import torch
import mano
from scipy.spatial.transform import Rotation as Ro
import argparse
from torch.utils.data import DataLoader
from dataset.HO3D_diversity_generation import HO3D_diversity
from network.affordanceNet_obman_mano_vertex import affordanceNet
from network.cmapnet_objhand import pointnet_reg
import numpy as np
import random
from utils import utils, utils_loss
import MANO.mano
import json
from utils.loss import TTT_loss
import trimesh
from metric.simulate import run_simulation
from scipy.io import loadmat, savemat
import open3d as o3d
import pickle

############## set up section ##################################################################
# get finger tip from vertices on mano hand
# TIP_IDS = {
#     'mano': {
#             'thumb':      744,
#             'index':      320,
#             'middle':     443,
#             'ring':           554,
#             'pinky':      671,
#         }
# }

TPID = [744, 320, 443, 554, 671]

# mano right hand location
MANO_PATH = '../data/bodymodel/mano/MANO_RIGHT.pkl'
# output folder location that contains all generated outputs
OUTPUT_BASE = './refined_subsamples_interp/'
OBJECT_LOCATION = '/home/ray/Downloads/Tools'
vhacd_exe = " optional path to vhacd executable"

### inmat for grabnet generated info
inmat = '../grasp_generation/samples_near/diskplacer_subsamples/00001/generate.mat'

###################################################################################################
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

    while (obj_mesh.vertices.shape[0] < n_sample_verts):
        new_mesh = obj_mesh.subdivide()
        obj_mesh = Mesh(vertices=new_mesh.vertices,
                        faces=new_mesh.faces,
                        visual=new_mesh.visual)

    verts_obj = obj_mesh.vertices
    # verts_sample_id = np.random.choice(verts_obj.shape[0], n_sample_verts, replace=False)
    # verts_sampled = verts_obj[verts_sample_id]
    verts_sampled, _ = trimesh.sample.sample_surface_even(obj_mesh, n_sample_verts, radius=None)

    return verts_sampled, obj_mesh, rand_rotmat


class Mesh(trimesh.Trimesh):

    def __init__(self,
                 filename=None,
                 vertices=None,
                 faces=None,
                 vc=None,
                 fc=None,
                 vscale=None,
                 radius=.002,
                 process=False,
                 visual=None,
                 wireframe=False,
                 smooth=False,
                 **kwargs):

        self.wireframe = wireframe
        self.smooth = smooth

        if filename is not None:
            mesh = trimesh.load(filename, process=process)
            vertices = mesh.vertices
            faces = mesh.faces
            visual = mesh.visual
        if vscale is not None:
            vertices = vertices * vscale

        if faces is None:
            mesh = points2sphere(vertices, radius=radius)
            vertices = mesh.vertices
            faces = mesh.faces
            visual = mesh.visual

        super(Mesh, self).__init__(vertices=vertices, faces=faces, process=process, visual=visual)

        if vc is not None:
            self.set_vertex_colors(vc)
        if fc is not None:
            self.set_face_colors(fc)

    def rotate_vertices(self, rxyz):
        visual = self.visual
        self.vertices[:] = np.array(self.vertices @ rxyz.T)
        self.visual = visual
        return self

    def colors_like(self, color, array, ids):

        color = np.array(color)

        if color.max() <= 1.:
            color = color * 255
        color = color.astype(np.int8)

        n_color = color.shape[0]
        n_ids = ids.shape[0]

        new_color = np.array(array)
        if n_color <= 4:
            new_color[ids, :n_color] = np.repeat(color[np.newaxis], n_ids, axis=0)
        else:
            new_color[ids, :] = color

        return new_color

    def set_vertex_colors(self, vc, vertex_ids=None):

        all_ids = np.arange(self.vertices.shape[0])
        if vertex_ids is None:
            vertex_ids = all_ids

        vertex_ids = all_ids[vertex_ids]
        new_vc = self.colors_like(vc, self.visual.vertex_colors, vertex_ids)
        self.visual.vertex_colors[:] = new_vc

    def set_face_colors(self, fc, face_ids=None):

        if face_ids is None:
            face_ids = np.arange(self.faces.shape[0])

        new_fc = self.colors_like(fc, self.visual.face_colors, face_ids)
        self.visual.face_colors[:] = new_fc

    @staticmethod
    def concatenate_meshes(meshes):
        return trimesh.util.concatenate(meshes)


def intersect_vox(obj_mesh, hand_mesh, pitch=0.5):
    '''
    Evaluating intersection between hand and object
    :param pitch: voxel size
    :return: intersection volume
    '''
    obj_vox = obj_mesh.voxelized(pitch=pitch)
    obj_points = obj_vox.points
    inside = hand_mesh.contains(obj_points)
    volume = inside.sum() * np.power(pitch, 3)
    return volume


def load_obj_verts_diskplacer(mesh_path, mesh_handle, rand_rotmat, rndrotate=True, scale=1., n_sample_verts=3000):
    np.random.seed(100)
    obj_mesh = Mesh(filename=mesh_path, vscale=scale)
    obj_handle = Mesh(filename=mesh_handle, vscale=scale)

    # if the object has no texture, make it yellow

    ## center and scale the object
    max_length = np.linalg.norm(obj_mesh.vertices, axis=1).max()
    if max_length > 1:
        re_scale = max_length / .08
        print(f'The object is very large, down-scaling by {re_scale} factor')
        obj_mesh.vertices[:] = obj_mesh.vertices / re_scale

    object_fullpts = obj_handle.vertices
    maximum = object_fullpts.max(0, keepdims=True)
    minimum = object_fullpts.min(0, keepdims=True)

    object_fullpts = obj_mesh.vertices

    offset = (maximum + minimum) / 2
    verts_obj = object_fullpts - offset
    obj_mesh.vertices[:] = verts_obj

    if rndrotate:
        obj_mesh.rotate_vertices(rand_rotmat)
    else:
        rand_rotmat = np.eye(3)

    # while (obj_mesh.vertices.shape[0]<n_sample_verts):
    #     new_mesh = obj_mesh.subdivide()
    #     obj_mesh = Mesh(vertices=new_mesh.vertices,
    #                     faces = new_mesh.faces,
    #                     visual = new_mesh.visual)

    verts_obj = obj_mesh.vertices
    # verts_sample_id = np.random.choice(verts_obj.shape[0], n_sample_verts, replace=False)
    # verts_sampled = verts_obj[verts_sample_id]
    verts_sampled, _ = trimesh.sample.sample_surface_even(obj_mesh, n_sample_verts, radius=None)

    return verts_sampled, obj_mesh, rand_rotmat


def mesh_vert_int_exts(obj1_mesh, obj2_verts):
    inside = obj1_mesh.ray.contains_points(obj2_verts)
    sign = (inside.astype(int) * 2) - 1
    return sign


def show_pcd(list_1):
    a = []
    import open3d as o3d
    for i in list_1:
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(i)
        a.append(pcd)

    o3d.visualization.draw_geometries(a)


# def enforce_contact_finger:

FRIEM_SELECTION = [8, 18, 23, 25, 50, 69, 85, 134, 136, 137, 138, 149, 161, 162, 168, 176, 179, 193, 194, 198, 204, 212,
                   214, 242, 252, 253, 254, 264, 277,
                   286, 295, 308, 318, 321, 326, 331, 342, 347, 365, 381, 384, 392, 393, 397, 400, 464, 468, 478, 492,
                   493]

DISKPLACER_SELECTION = [1, 8, 19, 25, 27, 28, 32, 46, 82, 83, 87, 100, 105, 108, 111, 119, 138, 149, 163, 165, 168, 176,
                        177, 187, 192, 204, 205, 240, 248,
                        250, 259, 261, 265, 276, 290, 293, 309, 313, 321, 341, 348, 359, 362, 364, 366, 390, 400, 402,
                        433, 439,
                        432, 454, 458, 475, 480, 499]

SCALEPL_SELECTION = [1, 7, 16, 41, 58, 81, 83, 90, 98, 110, 157, 169, 170, 176, 194, 201, 215, 216, 250, 252, 257, 259,
                     269,
                     341, 387, 393, 422, 436, 474, 485, 491, 496]


def main(args, model, cmap_model, device, rh_mano, rh_faces,using_contactnet=False):
    '''
    Generate diverse grasps for object index with args.obj_id in out-of-domain HO3D object models
    '''
    if using_contactnet:
        model.eval()
        cmap_model.eval()
    rh_mano.eval()
    # for batch_idx, (obj_id, obj_pc, origin_verts, origin_faces) in enumerate(eval_loader):
    if True:
        # if obj_id.item() != args.obj_id:
        #     continue
        # obj_xyz = obj_pc.permute(0, 2, 1)[:, :, :3].squeeze(0).cpu().numpy()  # [3000, 3]
        # origin_verts = origin_verts.squeeze(0).numpy()  # [N, 3]
        # recon_params, R_list, trans_list, r_list = [], [], [], []

        OUT_dir = os.path.join(OUTPUT_BASE, inmat.split('/')[-3], inmat.split('/')[-2])
        os.makedirs(OUT_dir, exist_ok=True)
        all_generated = loadmat(inmat)
        all_order_list = []

        interp_list = []
        # all_valid = loadmat( os.path.join( OUT_dir.replace('refined_subsamples_interp','refined_subsamples'), 'valid.mat'))
        # all_valid = all_valid['all_valid']
        all_valid = [int(temp.split('.')[0].split('_')[0]) for temp in
                     os.listdir(OUT_dir.replace('refined_subsamples_interp', 'refined_subsamples')) if
                     '.ply' in temp and 'Hand' in temp]
        selection_list = np.random.choice(all_valid, 30)
        save_source_list = []
        save_target_list = []
        for i in range(len(selection_list) - 1):

            frames_to_interpolate = [6, 7, 8, 9, 10, 11, 12]
            this_frames_interp = np.random.choice(frames_to_interpolate, 1)[0]
            # generate random rotation

            save_source_list.append(selection_list[i])
            save_target_list.append(selection_list[i + 1])
            interp_list.append(this_frames_interp)
            for temp_j in range(this_frames_interp):

                source_frame = selection_list[i]
                target_frame = selection_list[i + 1]

                if 'friem' in inmat:
                    verts_obj, mesh_obj, rotmat = load_obj_verts(
                        '../data/TOOLS_Release/Friem_original.ply',
                        all_generated['rotmat'][source_frame] @ Ro.from_euler('z', 0,
                                                                            degrees=True).as_matrix(),
                        rndrotate=True,
                        scale=0.001)
                elif 'diskplacer' in inmat:
                    verts_obj, mesh_obj, rotmat = load_obj_verts_diskplacer('../data/TOOLS_Release/DiskPlacer.stl',
                                                                            '../data/TOOLS_Release/DiskPlacer_handle.ply',
                                                                            all_generated['rotmat'][
                                                                                source_frame] @ Ro.from_euler('z', 0,
                                                                                                            degrees=True).as_matrix(),
                                                                            rndrotate=True,
                                                                            scale=0.001)
                elif 'scalpel' in inmat:
                    verts_obj, mesh_obj, rotmat = load_obj_verts('../data/TOOLS_Release/Scalpel.stl',
                                                                 all_generated['rotmat'][source_frame] @ Ro.from_euler(
                                                                     'z', 0,
                                                                     degrees=True).as_matrix(),
                                                                 rndrotate=True,
                                                                 scale=0.001)
                else:
                    print('WRONG')

                obj_mesh_verts = verts_obj
                # verts_obj = verts_obj @ np.linalg.inv(all_generated['rotmat'][index_temp])
                obj_pc_TTT = np.concatenate((verts_obj, np.ones((3000, 1)) * 0.2248), 1)
                obj_pc_TTT = torch.from_numpy(obj_pc_TTT).permute(1, 0).view(1, 4, 3000).float().to(device)

                this_global_orient = all_generated['global_orient'][[source_frame], :]
                # this_fullpose = all_generated['fullpose'][[index_temp], :]
                this_joints = all_generated['joints'][[source_frame], :]
                this_joints = torch.from_numpy(this_joints).float().to(device)
                target_joints = all_generated['joints'][[target_frame], :]
                target_joints = torch.from_numpy(target_joints).float().to(device)
                this_hand_pose = all_generated['hand_pose'][[source_frame], :]
                this_transl = all_generated['transl'][[source_frame], :]
                temp_hand = Mesh(filename=inmat[:-12] + str(source_frame).zfill(5) + '_Hand.ply')
                temp_hand_v = temp_hand.vertices
                temp_hand_v = temp_hand_v @ rotmat.T
                this_vert = torch.from_numpy(temp_hand_v).float().to(device)
                temp_hand = Mesh(filename=inmat[:-12] + str(target_frame).zfill(5) + '_Hand.ply')
                temp_hand_v = temp_hand.vertices
                temp_hand_v = temp_hand_v @ rotmat.T
                target_vert = torch.from_numpy(temp_hand_v).float().to(device)
                #       this_hand_beta = np.array([ 1.1150e-07,  8.3223e-07, -1.2874e-06, -2.2899e-06,  9.6721e-06,
                # 1.4173e-06,  2.0028e-07,  4.4639e-06, -1.0583e-06, -6.9350e-07])
                this_hand_beta = np.array([[0, 0, 0, 0, 0, 0, 0, 0, 0, 0]])
                recon_param = np.concatenate((this_hand_beta, this_global_orient, this_hand_pose, this_transl), 1)
                # recon_param = torch.from_numpy(recon_param).view(1, 61).float().to(device)
                recon_param = torch.from_numpy(this_hand_pose).view(1, 45).float().to(device)

                recon_param = torch.autograd.Variable(recon_param, requires_grad=True)
                optimizer = torch.optim.SGD([recon_param], lr=1e-6, momentum=0.8)
                # optimizer = torch.optim.SGD([recon_param], lr=0, momentum=0.8)
                # optimizer = torch.optim.Adam([recon_param], lr=1e-4)
                l2loss = torch.nn.MSELoss()

                source_full_joints = torch.cat((this_joints, this_vert[TPID, :].unsqueeze(0)), 1)
                target_full_joints = torch.cat((target_joints, target_vert[TPID, :].unsqueeze(0)), 1)

                gt_full_joints = source_full_joints + (target_full_joints - source_full_joints) * (
                            temp_j + 1) / this_frames_interp

                for j in range(300):  # non-learning based optimization steps
                    optimizer.zero_grad()

                    recon_mano = rh_mano(betas=torch.from_numpy(this_hand_beta).float().to(device),
                                         global_orient=torch.from_numpy(this_global_orient).float().to(device),
                                         hand_pose=recon_param, transl=torch.from_numpy(this_transl).float().to(device))

                    recon_xyz = recon_mano.vertices.float().to(device)  # [B,778,3], hand vertices
                    recon_joints = recon_mano.joints.to(device)

                    obj_nn_dist_affordance, _ = utils_loss.get_NN(obj_pc_TTT.permute(0, 2, 1)[:, :, :3], recon_xyz)
                    cmap_affordance = utils.get_pseudo_cmap(obj_nn_dist_affordance)  # [B,3000]

                    # predict target cmap by ContactNet
                    if using_contactnet:
                        recon_cmap = cmap_model(obj_pc_TTT[:, :3, :], recon_xyz.permute(0, 2, 1).contiguous())  # [B,3000]
                        recon_cmap = (recon_cmap / torch.max(recon_cmap, dim=1)[0]).detach()
                    else:
                        recon_cmap = torch.zeros((1, 3000)).to(device)

                    penetr_loss, consistency_loss, contact_loss, finger_contact_loss = TTT_loss(recon_xyz, rh_faces,
                                                                                                obj_pc_TTT[:, :3,
                                                                                                :].permute(0, 2,
                                                                                                           1).contiguous(),
                                                                                                cmap_affordance,
                                                                                                recon_cmap)
                    kp_weight = 100
                    new_joint = torch.cat((recon_joints, recon_xyz[:, TPID, :]), 1)

                    kp_loss = l2loss(new_joint * 1000, gt_full_joints * 1000)
                    loss = kp_weight * kp_loss

                    loss.backward()
                    optimizer.step()
                    # if j == 0 or j == 299:
                    #     print("Object sample {}, pose {}, iter {}, "
                    #           "penetration loss {:9.5f}, "
                    #           "kp_loss loss {:9.5f}, "
                    #           "contact loss {:9.5f}".format(batch_idx, i, j,
                    #                                         penetr_loss.item(), kp_loss.item(), contact_loss.item()))

                for j in range(300):  # non-learning based optimization steps
                    optimizer.zero_grad()

                    recon_mano = rh_mano(betas=torch.from_numpy(this_hand_beta).float().to(device),
                                         global_orient=torch.from_numpy(this_global_orient).float().to(device),
                                         hand_pose=recon_param, transl=torch.from_numpy(this_transl).float().to(device))
                    #
                    # recon_mano = rh_mano(betas=recon_param[:, :10], global_orient=recon_param[:, 10:13],
                    #                      hand_pose=recon_param[:, 13:58], transl=recon_param[:, 58:])
                    recon_xyz = recon_mano.vertices.float().to(device)  # [B,778,3], hand vertices
                    recon_joints = recon_mano.joints.to(device)

                    obj_nn_dist_affordance, _ = utils_loss.get_NN(obj_pc_TTT.permute(0, 2, 1)[:, :, :3], recon_xyz)
                    cmap_affordance = utils.get_pseudo_cmap(obj_nn_dist_affordance)  # [B,3000]

                    # predict target cmap by ContactNet
                    if using_contactnet:
                        recon_cmap = cmap_model(obj_pc_TTT[:, :3, :], recon_xyz.permute(0, 2, 1).contiguous())  # [B,3000]
                        recon_cmap = (recon_cmap / torch.max(recon_cmap, dim=1)[0]).detach()
                    else:
                        recon_cmap = torch.zeros((1, 3000)).to(device)

                    penetr_loss, consistency_loss, contact_loss, finger_contact_loss = TTT_loss(recon_xyz, rh_faces,
                                                                                                obj_pc_TTT[:, :3,
                                                                                                :].permute(0, 2,
                                                                                                           1).contiguous(),
                                                                                                cmap_affordance,
                                                                                                recon_cmap)
                    kp_weight = 10

                    new_joint = torch.cat((recon_joints, recon_xyz[:, TPID, :]), 1)
                    gt_joint = torch.cat((this_joints, this_vert[TPID, :].unsqueeze(0)), 1)
                    kp_loss = l2loss(new_joint * 100, gt_full_joints * 100)
                    # kp_loss = l2loss(recon_xyz*1000, this_vert*1000)
                    # loss = 0 * contact_loss + 0 * consistency_loss + 0 * penetr_loss + kp_weight * kp_loss + finger_contact_loss * 0
                    if 'diskplacer' in inmat:
                        PENE_TRA = 0.05
                        loss = 60 * contact_loss + 0 * consistency_loss + 300 * penetr_loss + kp_weight * kp_loss + finger_contact_loss * 0
                    elif 'friem' in inmat:
                        PENE_TRA = 0.01
                        loss = 20 * contact_loss + 10 * consistency_loss + 300 * penetr_loss + kp_weight * kp_loss + finger_contact_loss * 0
                    else:
                        PENE_TRA = 0.01
                        loss = 100 * contact_loss + 0 * consistency_loss + 30 * penetr_loss + kp_weight * kp_loss + finger_contact_loss * 0

                    # loss = 20 * contact_loss + 0 * consistency_loss + 15 * penetr_loss + kp_weight * kp_loss + finger_contact_loss * 0   ###diskplacer
                    # loss = 20 * contact_loss + 0 * consistency_loss + 300 * penetr_loss + kp_weight * kp_loss + finger_contact_loss*0   ### Friem
                    # loss = 60 * contact_loss + 0 * consistency_loss + 300 * penetr_loss + kp_weight * kp_loss + finger_contact_loss*0   ### scalpel
                    loss.backward()
                    optimizer.step()
                    if j == 0 or j == 299:
                        print("iter {}, "
                              "penetration loss {:9.5f}, "
                              "kp_loss loss {:9.5f}, "
                              "contact loss {:9.5f}".format(  j,
                                                            penetr_loss.item(), kp_loss.item(), contact_loss.item()))

                # evaluate grasp

                obj_mesh = mesh_obj

                final_mano = rh_mano(betas=torch.from_numpy(this_hand_beta).float().to(device),
                                     global_orient=torch.from_numpy(this_global_orient).float().to(device),
                                     hand_pose=recon_param, transl=torch.from_numpy(this_transl).float().to(device))
                save_hand_dict = {}
                save_hand_dict['betas'] = this_hand_beta
                save_hand_dict['global_orient'] = this_global_orient
                save_hand_dict['hand_pose'] = recon_param.detach().cpu().numpy()
                save_hand_dict['transl'] = this_transl

                final_mano_verts = final_mano.vertices.squeeze(0).detach().cpu().numpy()  # [778, 3]
                all_order_list.append(final_mano.joints[0, 3, :].detach().cpu().numpy())

                mesh_1 = o3d.geometry.TriangleMesh()
                mesh_1.vertices = o3d.utility.Vector3dVector(final_mano_verts)
                mesh_1.triangles = o3d.utility.Vector3iVector(rh_faces.squeeze(0).detach().cpu().numpy())
                o3d.io.write_triangle_mesh(
                    os.path.join(OUT_dir, str(source_frame).zfill(5) + '_' + str(target_frame).zfill(5) + '_'
                                 + str(temp_j) + '_Hand.ply'), mesh_1)
                output_path_pkl_addtion = os.path.join(OUT_dir,
                                                       str(source_frame).zfill(5) + '_' + str(target_frame).zfill(
                                                           5) + '_'
                                                       + str(temp_j) + '_MANO.pkl')

                with open(output_path_pkl_addtion, 'wb') as f:
                    pickle.dump(save_hand_dict, f)

                try:
                    simu_disp = run_simulation(final_mano_verts, rh_faces.reshape((-1, 3)),
                                               obj_mesh_verts,
                                               obj_mesh.faces,
                                               vhacd_exe=vhacd_exe, sample_idx=i)
                except:
                    print('NO simulation displacement avaiableo')
                    simu_disp = 0.00010
                # save_flag = (penetr_vol < args.penetr_vol_thre) and (simu_disp < args.simu_disp_thre) and sample_contact
                print('generate id: {}, penetr vol: {}, simu disp: {}, contact: {}, save flag: {}'
                      .format(i, 0, simu_disp, 0, True))

        temp_list = [temp[2] for temp in all_order_list]
        # print(np.array(np.argsort(temp_list)))
        file = open(os.path.join(OUT_dir, "order.txt"), "w+")
        # save_source_list.append(selection_list[i])
        # save_target_list.append(selection_list[i + 1])
        # interp_list.append(this_frames_interp)
        save_dict = {}
        save_dict['save_source_list'] = save_source_list
        save_dict['save_target_list'] = save_target_list
        save_dict['interp_list'] = interp_list

        savemat(os.path.join(OUT_dir, 'interpolate.mat'), save_dict)

        # Saving the 2D array in a text file
        content = str(np.array(np.argsort(temp_list)))
        file.write(content)
        file.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    '''experiment setting'''
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch_size", type=int, default=1)
    parser.add_argument("--use_cuda", type=int, default=1)
    parser.add_argument("--dataloader_workers", type=int, default=32)
    '''affordance network information'''
    parser.add_argument("--affordance_model_path", type=str, default='checkpoints/model_affordance_best_full.pth')
    parser.add_argument("--encoder_layer_sizes", type=list, default=[1024, 512, 256])
    parser.add_argument("--decoder_layer_sizes", type=list, default=[1024, 256, 61])
    parser.add_argument("--latent_size", type=int, default=64)
    parser.add_argument("--obj_inchannel", type=int, default=4)
    parser.add_argument("--condition_size", type=int, default=1024)
    '''cmap network information'''
    parser.add_argument("--cmap_model_path", type=str, default='checkpoints/model_cmap_best.pth')
    '''Generated graps information'''
    parser.add_argument("--obj_id", type=int, default=6)
    # You can change the two thresholds to save the graps you want
    parser.add_argument("--penetr_vol_thre", type=float, default=9e-6)  # 4cm^3
    parser.add_argument("--simu_disp_thre", type=float, default=0.03)  # 3cm
    parser.add_argument("--num_grasp", type=int, default=100)  # number of grasps you want to generate
    args = parser.parse_args()
    assert args.obj_id in [3, 4, 6, 10, 11, 19, 21, 25, 35, 37, 99]

    # device
    use_cuda = args.use_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    print("using device", device)

    # network
    affordance_model = affordanceNet(obj_inchannel=args.obj_inchannel,
                                     cvae_encoder_sizes=args.encoder_layer_sizes,
                                     cvae_latent_size=args.latent_size,
                                     cvae_decoder_sizes=args.decoder_layer_sizes,
                                     cvae_condition_size=args.condition_size)  # GraspCVAE
    cmap_model = pointnet_reg(with_rgb=False)  # ContactNet
    using_contactnet = False
    # load pre-trained model
    if using_contactnet:
        checkpoint_affordance = torch.load(args.affordance_model_path, map_location=torch.device('cpu'))['network']
        affordance_model.load_state_dict(checkpoint_affordance)
        affordance_model = affordance_model.to(device)
        checkpoint_cmap = torch.load(args.cmap_model_path, map_location=torch.device('cpu'))['network']
        cmap_model.load_state_dict(checkpoint_cmap)
        cmap_model = cmap_model.to(device)
    else:
        affordance_model = None
        cmap_model = None

    # dataset
    # dataset = HO3D_diversity()
    # dataloader = DataLoader(dataset=dataset, batch_size=1, shuffle=False, num_workers=1)
    # mano hand model
    with torch.no_grad():
        rh_mano = mano.load(model_path=MANO_PATH,
                            model_type='mano',
                            num_pca_comps=45,
                            batch_size=1,
                            flat_hand_mean=True).to(device)
    rh_faces = torch.from_numpy(rh_mano.faces.astype(np.int32)).view(1, -1, 3).to(device)  # [1, 1538, 3], face indexes

    main(args, affordance_model, cmap_model, device, rh_mano, rh_faces, using_contactnet=using_contactnet)
