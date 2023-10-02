import torch
import torch.nn as nn
from torch.nn import functional as F
from nets.backbone import FPN
from nets.transformer import Transformer
from nets.regressor import Regressor
from utils.mano import MANO
from config import cfg
import cv2

import scipy.io as sio

import open3d as o3d
import matplotlib.pyplot as plt
import math
from random import  *
import numpy as np
jointsMapManoToSimple = [0, 13, 14, 15, 16,
                                 1, 2, 3, 17,
                                 4, 5, 6, 18,
                                 10, 11, 12, 19,
                                 7, 8, 9, 20]
class Model(nn.Module):
    def __init__(self, backbone, FIT, SET, regressor):
        super(Model, self).__init__()
        self.backbone = backbone
        self.FIT = FIT
        self.SET = SET
        self.regressor = regressor
        # self.

    def vis_keypoints_with_skeleton(self, image, kp, fname ='/home/rui/Downloads/inftest0.png'):
        color = np.ones(shape=(256, 256, 3), dtype=np.int16)
        color[:, :, 0] = image[:, :, 2]
        color[:, :, 1] = image[:, :, 1]
        color[:, :, 2] = image[:, :, 0]
        img = color
        # kp = kp[[0]]
        # kp[:, 0] = 512 - kp[:, 0]
        # kp[:, 1] = 512 - kp[:, 1]
        index_p2d = np.ones((21,3))
        index_p2d[:, 0] = np.clip(kp[:, 0], 0, 512 - 1)
        index_p2d[:, 1] = np.clip(kp[:, 1], 0, 512 - 1)
        kps = index_p2d.T
        kp_thresh = 0.4
        alpha = 1
        # Convert from plt 0-1 RGBA colors to 0-255 BGR colors for opencv.
        # kps_lines = [
        #         [0, 1],
        #     [1, 2],
        #     [2, 3],
        #     [3, 17],
        #     [0, 13],
        #     [13, 14],
        #     [14, 15],
        #     [15, 16],
        #     [0, 4],
        #     [4, 5],
        #     [5, 6],
        #     [6, 18],
        #     [0, 10],
        #     [10, 11],
        #     [11, 12],
        #     [12, 19],
        #     [0, 7],
        #     [7, 8],
        #     [8, 9],
        #     [9, 20]
        #          ]

        kps_lines = [[0, 1],
                     [1, 2],
                     [2, 3],
                     [3, 4],
                     [0, 5],
                     [5, 6],
                     [6, 7],
                     [7, 8],
                     [0, 9],
                     [9, 10],
                     [10, 11],
                     [11, 12],
                     [0, 13],
                     [13, 14],
                     [14, 15],
                     [15, 16],
                     [0, 17],
                     [17, 18],
                     [18, 19],
                     [19, 20]
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
        cv2.imwrite(fname, o_img)

    def compute_similarity_transform(self, S1, S2):
        """
        Computes a similarity transform (sR, t) that takes
        a set of 3D points S1 (3 x N) closest to a set of 3D points S2,
        where R is an 3x3 rotation matrix, t 3x1 translation, s scale.
        i.e. solves the orthogonal Procrutes problem.
        """
        transposed = False
        if S1.shape[0] != 3 and S1.shape[0] != 2:
            S1 = S1.T
            S2 = S2.T
            transposed = True
        assert (S2.shape[1] == S1.shape[1])

        # 1. Remove mean.
        mu1 = S1.mean(axis=1, keepdims=True)
        mu2 = S2.mean(axis=1, keepdims=True)
        X1 = S1 - mu1
        X2 = S2 - mu2

        # 2. Compute variance of X1 used for scale.
        var1 = np.sum(X1 ** 2)

        # 3. The outer product of X1 and X2.
        K = X1.dot(X2.T)

        # 4. Solution that Maximizes trace(R'K) is R=U*V', where U, V are
        # singular vectors of K.
        U, s, Vh = np.linalg.svd(K)
        V = Vh.T
        # Construct Z that fixes the orientation of R to get det(R)=1.
        Z = np.eye(U.shape[0])
        Z[-1, -1] *= np.sign(np.linalg.det(U.dot(V.T)))
        # Construct R.
        R = V.dot(Z.dot(U.T))

        # 5. Recover scale.
        scale = np.trace(R.dot(K)) / var1

        # 6. Recover translation.
        t = mu2 - scale * (R.dot(mu1))

        # 7. Error:
        S1_hat = scale * R.dot(S1) + t

        if transposed:
            S1_hat = S1_hat.T

        return S1_hat

    def recover_joints(self, joints2d, bbox):
        bbox = bbox.reshape(2, 2)
        joints2d = joints2d * (bbox[1, :] - bbox[0, :]) + bbox[0, :]
        return joints2d

    # def process_joints(self, joints2d, bbox):
    #     bbox = bbox.reshape(2, 2)
    #     joints2d = joints2d * (bbox[1, :] - bbox[0, :]) + bbox[0, :]
    #     joints2d = (joints2d - bbox[0, :]) * (bbox[1, :] - bbox[0, :]) + bbox[0, :]
    #     return joints2d
    
    def forward(self, inputs, targets, meta_info, mode,this_name=None):
        p_feats, s_feats = self.backbone(inputs['img']) # primary, secondary feats
        feats = self.FIT(s_feats, p_feats)
        feats = self.SET(feats, feats)
        key_l = ['all_addition_g', 'all_addition_t_no_transl', 'rot_aug', 'joints2d', 'mano_param', 'bbox_hand']
        for key in key_l:
            targets[key] = targets[key].float()
        targets['joints2d'] = targets['joints2d'][:,jointsMapManoToSimple,:]
        # targets['joints2d'] = targets['joints2d'][:, jointsMapManoToSimple, :]
        all_hand_j = []
        for i_in_b in range(targets['bbox_hand'].shape[0]):
            hand_box_temp = targets['bbox_hand'][i_in_b].detach().cpu().numpy()
            hand_joints_temp = self.recover_joints(targets['joints2d'][i_in_b].detach().cpu().numpy(), hand_box_temp)
            hand_joints_temp = hand_joints_temp / 256
            hand_joints_temp = np.expand_dims(hand_joints_temp, axis=0)
            all_hand_j.append(hand_joints_temp)

        all_hand_j = np.concatenate(all_hand_j, 0)
        targets['joints2d'] = torch.from_numpy(all_hand_j).float().cuda()
        # targets['joints3d'] = targets['joints3d'][:,jointsMapManoToSimple,:]
        # jointsMapManoToSimple

        gt_mano_params = targets
        # if mode == 'train':
        #     gt_mano_params = torch.cat([targets['mano_pose'], targets['mano_shape']], dim=1)
        # else:
        #     gt_mano_params = None
        pred_mano_results, gt_mano_results, preds_joints_img = self.regressor(feats, gt_mano_params)
       
        if mode == 'train':

            epch_this, step_this = this_name.split('_')
            # loss functions
            loss = {}
            # loss['mano_verts'] = 0 * F.mse_loss(pred_mano_results['verts3d'], gt_mano_results['verts3d'])

            loss['mano_verts'] = cfg.lambda_mano_verts * F.mse_loss(pred_mano_results['verts3d'], gt_mano_results['verts3d'])
            loss['mano_joints'] = cfg.lambda_mano_joints * F.mse_loss(pred_mano_results['joints3d'], gt_mano_results['joints3d'])
            # loss['mano_pose'] = cfg.lambda_mano_pose * F.mse_loss(pred_mano_results['mano_pose'], gt_mano_results['mano_pose'])
            loss['mano_pose'] = 0 * F.mse_loss(pred_mano_results['mano_pose'], gt_mano_results['mano_pose'])
            # loss['mano_shape'] = cfg.lambda_mano_shape * F.mse_loss(pred_mano_results['mano_shape'], gt_mano_results['mano_shape'])
            loss['mano_shape'] = cfg.lambda_mano_shape * F.mse_loss(pred_mano_results['mano_shape'],
                                                                   torch.zeros_like(pred_mano_results['mano_shape']))
            loss['joints_img'] = cfg.lambda_joints_img * F.mse_loss(preds_joints_img[0], targets['joints2d'])
            # loss['joints_img'] = 0 * F.mse_loss(preds_joints_img[0], targets['joints2d'])


            # for i_in_b in range(targets['bbox_hand'].shape[0]):
            #     hand_temp = preds_joints_img[0][i_in_b].detach().cpu().numpy()
            #     # hand_box_temp = targets['bbox_hand'][0].detach().cpu().numpy()
            #     hand_joints_temp = hand_temp * 256 #self.recover_joints(hand_temp, hand_box_temp)
            #     img_temp = inputs['img'][i_in_b].permute(1, 2, 0).detach().cpu().numpy() * 255
            #     gt_temp = targets['joints2d'][i_in_b].detach().cpu().numpy()
            #     gt_temp = gt_temp * 256 #self.recover_joints(gt_temp, hand_box_temp)
            #     img_temp = img_temp.astype(np.int16)
            #     import os
            #     save_snapshot = True
            #     # which_mode, epoch_n, batch_n = snapshot_name.split('_')
            #     save_dir_temp = os.path.join(
            #         '/home/rui/projects/sp2_ws/HandOccNet/debg', 'train',
            #         epch_this)
            #     os.makedirs(save_dir_temp, exist_ok=True)
            #     # save_ply_name = os.path.join(save_dir_temp, mode)
            #
            #     self.vis_keypoints_with_skeleton(img_temp, gt_temp,
            #                                      fname=os.path.join(save_dir_temp, step_this + targets['seqName'][i_in_b]+'_' +targets['id'][i_in_b]+'_gt.png'))
            #     self.vis_keypoints_with_skeleton(img_temp, hand_joints_temp,
            #                                      fname=os.path.join(save_dir_temp, step_this + targets['seqName'][i_in_b]+'_' +targets['id'][i_in_b]+ '_pred.png'))


            if randint(1, 100) == 1:
                hand_temp = preds_joints_img[0][0].detach().cpu().numpy()
                # hand_box_temp = targets['bbox_hand'][0].detach().cpu().numpy()
                hand_joints_temp = hand_temp * 256 #self.recover_joints(hand_temp, hand_box_temp)
                img_temp = inputs['img'][0].permute(1, 2, 0).detach().cpu().numpy() * 255
                gt_temp = targets['joints2d'][0].detach().cpu().numpy()
                gt_temp = gt_temp * 256 #self.recover_joints(gt_temp, hand_box_temp)
                img_temp = img_temp.astype(np.int16)
                import os
                save_snapshot = True
                # which_mode, epoch_n, batch_n = snapshot_name.split('_')
                save_dir_temp = os.path.join(
                    '../debg', 'train',
                    epch_this)
                os.makedirs(save_dir_temp, exist_ok=True)
                # save_ply_name = os.path.join(save_dir_temp, mode)

                self.vis_keypoints_with_skeleton(img_temp, gt_temp,
                                                 fname=os.path.join(save_dir_temp, step_this + '_gt.png'))
                self.vis_keypoints_with_skeleton(img_temp, hand_joints_temp,
                                                 fname=os.path.join(save_dir_temp, step_this + '_pred.png'))

                vert_gt_o = gt_mano_results['verts3d']
                vert_pred_o = pred_mano_results['verts3d']
                a = vert_gt_o.detach().cpu().numpy()[0]
                b = sio.loadmat(os.path.join(cfg.root_root,'data','rh_face.mat'))['rh_face']
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(a)
                mesh.triangles = o3d.utility.Vector3iVector(b)
                o3d.io.write_triangle_mesh(os.path.join(
                    '../debg/train', epch_this, step_this + '_gt.ply'), mesh)
                c = vert_pred_o.detach().cpu().numpy()[0]

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(c)
                mesh.triangles = o3d.utility.Vector3iVector(b)
                o3d.io.write_triangle_mesh(os.path.join(
                '../debg/train' , epch_this, step_this +'_pred.ply'
                ), mesh)

                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector( gt_mano_results['joints3d'][0].cpu().numpy())
                # o3d.io.write_point_cloud(os.path.join(
                # '/home/rui/projects/sp2_ws/HandOccNet/debg/train' , epch_this, step_this +'J_gt.ply'
                # ), pcd)
                # pcd1 = o3d.geometry.PointCloud()
                # pcd1.points = o3d.utility.Vector3dVector(pred_mano_results['joints3d'][0].clone().detach().cpu().numpy())
                # o3d.io.write_point_cloud(os.path.join(
                # '/home/rui/projects/sp2_ws/HandOccNet/debg/train' , epch_this, step_this +'J_pred.ply'
                # ), pcd1)
            return loss
        elif mode == 'my_val':
            epch_this = '0'
            step_this = this_name
            # loss functions
            loss = {}
            v2v = torch.sqrt(((pred_mano_results['verts3d'] - gt_mano_results['verts3d']) ** 2).sum(dim=-1))
            mesh3d_loss = v2v.mean(dim=-1)
            loss['mano_verts'] = 1000 * mesh3d_loss.sum()

            # loss['mano_verts'] = F.mse_loss(pred_mano_results['verts3d'], gt_mano_results['verts3d']) * 1000
            error_per_joint = torch.sqrt(((pred_mano_results['joints3d'] - gt_mano_results['joints3d']) ** 2).sum(dim=-1))
            joints3d_loss = error_per_joint.mean(dim=-1)
            loss['mano_joints'] = 1000 * joints3d_loss.sum()

            # targets['scale_img']
            # loss['mano_verts'] = cfg.lambda_mano_verts * F.mse_loss(pred_mano_results['verts3d'], gt_mano_results['verts3d'])
            # loss['mano_joints'] = F.mse_loss(pred_mano_results['joints3d'], gt_mano_results['joints3d']) * 1000
            error_2d_joint = torch.sqrt(((preds_joints_img[0] * targets['scale_img'] - targets['joints2d'] * targets['scale_img']) ** 2).sum(dim=-1))
            e2d = error_2d_joint.mean(dim=-1).sum()
            loss['joints_img'] = e2d

            joint_out1 = self.compute_similarity_transform(pred_mano_results['joints3d'][0].detach().cpu().numpy(),
                                gt_mano_results['joints3d'][0].detach().cpu().numpy())

            pa_j3d = np.sqrt(((joint_out1 - gt_mano_results['joints3d'][0].detach().cpu().numpy()) ** 2).sum(1))
            pa_j3d_mean = pa_j3d.mean() * 1000
            loss['j3d_pa'] = torch.tensor(pa_j3d_mean).float().cuda()

            vo = self.compute_similarity_transform(pred_mano_results['verts3d'][0].detach().cpu().numpy(),
                                gt_mano_results['verts3d'][0].detach().cpu().numpy())
            pa_v3d = np.sqrt(((vo - gt_mano_results['verts3d'][0].detach().cpu().numpy()) ** 2).sum(1))
            pa_v3d_mean = pa_v3d.mean() * 1000
            loss['v3d_pa'] = torch.tensor(pa_v3d_mean).float().cuda()


            # loss['joints_img'] = cfg.lambda_joints_img * F.mse_loss(preds_joints_img[0] * 256, targets['joints2d'] * 256)

            # if randint(1, 100) == 1:
            if int(step_this) % 10 ==0:
                hand_temp = preds_joints_img[0][0].detach().cpu().numpy()
                # hand_box_temp = targets['bbox_hand'][0].detach().cpu().numpy()
                hand_joints_temp = hand_temp * 256 #self.recover_joints(hand_temp, hand_box_temp)
                img_temp = inputs['img'][0].permute(1, 2, 0).detach().cpu().numpy() * 255
                gt_temp = targets['joints2d'][0].detach().cpu().numpy()
                gt_temp = gt_temp * 256 #self.recover_joints(gt_temp, hand_box_temp)
                img_temp = img_temp.astype(np.int16)
                import os
                save_snapshot = True
                # which_mode, epoch_n, batch_n = snapshot_name.split('_')
                save_dir_temp = os.path.join(
                    '../debg', 'val',
                    '0')
                os.makedirs(save_dir_temp, exist_ok=True)
                # save_ply_name = os.path.join(save_dir_temp, mode)

                self.vis_keypoints_with_skeleton(img_temp, gt_temp,
                                                 fname=os.path.join(save_dir_temp, step_this + '_gt.png'))
                self.vis_keypoints_with_skeleton(img_temp, hand_joints_temp,
                                                 fname=os.path.join(save_dir_temp, step_this + '_pred.png'))

                resized = cv2.resize(img_temp, (224, 224), interpolation=cv2.INTER_LINEAR)



                vert_gt_o = gt_mano_results['verts3d']
                vert_pred_o = pred_mano_results['verts3d']
                a = vert_gt_o.detach().cpu().numpy()[0]
                b = sio.loadmat(os.path.join(cfg.root_root,'data','rh_face.mat'))['rh_face']
                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(a)
                mesh.triangles = o3d.utility.Vector3iVector(b)
                o3d.io.write_triangle_mesh(os.path.join(
                    '../debg/val', epch_this, step_this + '_gt.ply'), mesh)
                c = vert_pred_o.detach().cpu().numpy()[0]

                mesh = o3d.geometry.TriangleMesh()
                mesh.vertices = o3d.utility.Vector3dVector(c)
                mesh.triangles = o3d.utility.Vector3iVector(b)
                o3d.io.write_triangle_mesh(os.path.join(
                '../debg/val' , epch_this, step_this +'_pred.ply'
                ), mesh)

                # pcd = o3d.geometry.PointCloud()
                # pcd.points = o3d.utility.Vector3dVector( gt_mano_results['joints3d'][0].cpu().numpy())
                # o3d.io.write_point_cloud(os.path.join(
                # '/home/rui/projects/sp2_ws/HandOccNet/debg/train' , epch_this, step_this +'J_gt.ply'
                # ), pcd)
                # pcd1 = o3d.geometry.PointCloud()
                # pcd1.points = o3d.utility.Vector3dVector(pred_mano_results['joints3d'][0].clone().detach().cpu().numpy())
                # o3d.io.write_point_cloud(os.path.join(
                # '/home/rui/projects/sp2_ws/HandOccNet/debg/train' , epch_this, step_this +'J_pred.ply'
                # ), pcd1)
            return loss

        else:
            # test output
            out = {}
            out['joints_coord_cam'] = pred_mano_results['joints3d']
            out['mesh_coord_cam'] = pred_mano_results['verts3d']
            return out

def init_weights(m):
    if type(m) == nn.ConvTranspose2d:
        nn.init.normal_(m.weight,std=0.001)
    elif type(m) == nn.Conv2d:
        nn.init.normal_(m.weight,std=0.001)
        nn.init.constant_(m.bias, 0)
    elif type(m) == nn.BatchNorm2d:
        nn.init.constant_(m.weight,1)
        nn.init.constant_(m.bias,0)
    elif type(m) == nn.Linear:
        nn.init.normal_(m.weight,std=0.01)
        nn.init.constant_(m.bias,0)

def get_model(mode):
    backbone = FPN(pretrained=True)
    FIT = Transformer(injection=True) # feature injecting transformer
    SET = Transformer(injection=False) # self enhancing transformer
    regressor = Regressor()
    
    if mode == 'train':
        FIT.apply(init_weights)
        SET.apply(init_weights)
        regressor.apply(init_weights)
        
    model = Model(backbone, FIT, SET, regressor)
    
    return model