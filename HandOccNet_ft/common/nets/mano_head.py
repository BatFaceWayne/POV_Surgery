import torch
from torch import nn
from torch.nn import functional as F
from utils.mano import MANO
import numpy as np
mano = MANO()
import trimesh
jointsMapManoToSimple = [0, 13, 14, 15, 16,
                                 1, 2, 3, 17,
                                 4, 5, 6, 18,
                                 10, 11, 12, 19,
                                 7, 8, 9, 20]
from random import *
def batch_rodrigues(theta):
    # theta N x 3
    l1norm = torch.norm(theta + 1e-8, p=2, dim=1)
    angle = torch.unsqueeze(l1norm, -1)
    normalized = torch.div(theta, angle)
    angle = angle * 0.5
    v_cos = torch.cos(angle)
    v_sin = torch.sin(angle)
    quat = torch.cat([v_cos, v_sin * normalized], dim=1)

    return quat2mat(quat)


def quat2mat(quat):
    """Convert quaternion coefficients to rotation matrix.
    """
    norm_quat = quat
    norm_quat = norm_quat / norm_quat.norm(p=2, dim=1, keepdim=True)
    w, x, y, z = norm_quat[:, 0], norm_quat[:, 1], norm_quat[:, 2], norm_quat[:, 3]

    B = quat.size(0)

    w2, x2, y2, z2 = w.pow(2), x.pow(2), y.pow(2), z.pow(2)
    wx, wy, wz = w * x, w * y, w * z
    xy, xz, yz = x * y, x * z, y * z

    rotMat = torch.stack([w2 + x2 - y2 - z2, 2 * xy - 2 * wz, 2 * wy + 2 * xz,
                          2 * wz + 2 * xy, w2 - x2 + y2 - z2, 2 * yz - 2 * wx,
                          2 * xz - 2 * wy, 2 * wx + 2 * yz, w2 - x2 - y2 + z2], dim=1).view(B, 3, 3)
    return rotMat


def quat2aa(quaternion):
    """Convert quaternion vector to angle axis of rotation."""
    if not torch.is_tensor(quaternion):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(quaternion)))

    if not quaternion.shape[-1] == 4:
        raise ValueError("Input must be a tensor of shape Nx4 or 4. Got {}"
                         .format(quaternion.shape))
    # unpack input and compute conversion
    q1 = quaternion[..., 1]
    q2 = quaternion[..., 2]
    q3 = quaternion[..., 3]
    sin_squared_theta = q1 * q1 + q2 * q2 + q3 * q3

    sin_theta = torch.sqrt(sin_squared_theta)
    cos_theta = quaternion[..., 0]
    two_theta = 2.0 * torch.where(
        cos_theta < 0.0,
        torch.atan2(-sin_theta, -cos_theta),
        torch.atan2(sin_theta, cos_theta))

    k_pos = two_theta / sin_theta
    k_neg = 2.0 * torch.ones_like(sin_theta)
    k = torch.where(sin_squared_theta > 0.0, k_pos, k_neg)

    angle_axis = torch.zeros_like(quaternion)[..., :3]
    angle_axis[..., 0] += q1 * k
    angle_axis[..., 1] += q2 * k
    angle_axis[..., 2] += q3 * k
    return angle_axis


def mat2quat(rotation_matrix, eps=1e-6):
    """Convert 3x4 rotation matrix to 4d quaternion vector"""
    if not torch.is_tensor(rotation_matrix):
        raise TypeError("Input type is not a torch.Tensor. Got {}".format(
            type(rotation_matrix)))

    if len(rotation_matrix.shape) > 3:
        raise ValueError(
            "Input size must be a three dimensional tensor. Got {}".format(
                rotation_matrix.shape))
    if not rotation_matrix.shape[-2:] == (3, 4):
        raise ValueError(
            "Input size must be a N x 3 x 4  tensor. Got {}".format(
                rotation_matrix.shape))

    rmat_t = torch.transpose(rotation_matrix, 1, 2)

    mask_d2 = rmat_t[:, 2, 2] < eps

    mask_d0_d1 = rmat_t[:, 0, 0] > rmat_t[:, 1, 1]
    mask_d0_nd1 = rmat_t[:, 0, 0] < -rmat_t[:, 1, 1]

    t0 = 1 + rmat_t[:, 0, 0] - rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q0 = torch.stack([rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      t0, rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2]], -1)
    t0_rep = t0.repeat(4, 1).t()

    t1 = 1 - rmat_t[:, 0, 0] + rmat_t[:, 1, 1] - rmat_t[:, 2, 2]
    q1 = torch.stack([rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] + rmat_t[:, 1, 0],
                      t1, rmat_t[:, 1, 2] + rmat_t[:, 2, 1]], -1)
    t1_rep = t1.repeat(4, 1).t()

    t2 = 1 - rmat_t[:, 0, 0] - rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q2 = torch.stack([rmat_t[:, 0, 1] - rmat_t[:, 1, 0],
                      rmat_t[:, 2, 0] + rmat_t[:, 0, 2],
                      rmat_t[:, 1, 2] + rmat_t[:, 2, 1], t2], -1)
    t2_rep = t2.repeat(4, 1).t()

    t3 = 1 + rmat_t[:, 0, 0] + rmat_t[:, 1, 1] + rmat_t[:, 2, 2]
    q3 = torch.stack([t3, rmat_t[:, 1, 2] - rmat_t[:, 2, 1],
                      rmat_t[:, 2, 0] - rmat_t[:, 0, 2],
                      rmat_t[:, 0, 1] - rmat_t[:, 1, 0]], -1)
    t3_rep = t3.repeat(4, 1).t()

    mask_c0 = mask_d2 * mask_d0_d1
    mask_c1 = mask_d2 * ~mask_d0_d1
    mask_c2 = ~mask_d2 * mask_d0_nd1
    mask_c3 = ~mask_d2 * ~mask_d0_nd1
    mask_c0 = mask_c0.view(-1, 1).type_as(q0)
    mask_c1 = mask_c1.view(-1, 1).type_as(q1)
    mask_c2 = mask_c2.view(-1, 1).type_as(q2)
    mask_c3 = mask_c3.view(-1, 1).type_as(q3)

    q = q0 * mask_c0 + q1 * mask_c1 + q2 * mask_c2 + q3 * mask_c3
    q /= torch.sqrt(t0_rep * mask_c0 + t1_rep * mask_c1 +  # noqa
                    t2_rep * mask_c2 + t3_rep * mask_c3)  # noqa
    q *= 0.5
    return q


def rot6d2mat(x):
    """Convert 6D rotation representation to 3x3 rotation matrix.
    Based on Zhou et al., "On the Continuity of Rotation Representations in Neural Networks", CVPR 2019
    """
    a1 = x[:, 0:3]
    a2 = x[:, 3:6]
    b1 = F.normalize(a1)
    b2 = F.normalize(a2 - torch.einsum('bi,bi->b', b1, a2).unsqueeze(-1) * b1)
    b3 = torch.cross(b1, b2, dim=1)
    return torch.stack((b1, b2, b3), dim=-1)


def mat2aa(rotation_matrix):
    """Convert 3x4 rotation matrix to Rodrigues vector"""

    def convert_points_to_homogeneous(points):
        if not torch.is_tensor(points):
            raise TypeError("Input type is not a torch.Tensor. Got {}".format(
                type(points)))
        if len(points.shape) < 2:
            raise ValueError("Input must be at least a 2D tensor. Got {}".format(
                points.shape))

        return F.pad(points, (0, 1), "constant", 1.0)

    if rotation_matrix.shape[1:] == (3, 3):
        rotation_matrix = convert_points_to_homogeneous(rotation_matrix)
    quaternion = mat2quat(rotation_matrix)
    aa = quat2aa(quaternion)
    aa[torch.isnan(aa)] = 0.0
    return aa


class mano_regHead(nn.Module):
    def __init__(self, mano_layer=mano.layer, feature_size=1024, mano_neurons=[1024, 512]):
        super(mano_regHead, self).__init__()

        # 6D representation of rotation matrix
        self.pose6d_size = 16 * 6
        self.mano_pose_size = 16 * 3

        # Base Regression Layers
        mano_base_neurons = [feature_size] + mano_neurons
        base_layers = []
        for layer_idx, (inp_neurons, out_neurons) in enumerate(
                zip(mano_base_neurons[:-1], mano_base_neurons[1:])):
            base_layers.append(nn.Linear(inp_neurons, out_neurons))
            base_layers.append(nn.LeakyReLU(inplace=True))
        self.mano_base_layer = nn.Sequential(*base_layers)
        # Pose layers
        self.pose_reg = nn.Linear(mano_base_neurons[-1], self.pose6d_size)
        # Shape layers
        self.shape_reg = nn.Linear(mano_base_neurons[-1], 10)

        self.mano_layer = mano.get_layer()
        self.mano_layer_gt = mano.get_layer_gt()

    def forward(self, features, GT_mano_params=None):
        # for key, val in enumerate(GT_mano_params):
        #     GT_mano_params[key] = GT_mano_params[key].float()
        gt_mano_params = GT_mano_params['mano_param']
        mano_features = self.mano_base_layer(features)
        pred_mano_pose_6d = self.pose_reg(mano_features)
        
        pred_mano_pose_rotmat = rot6d2mat(pred_mano_pose_6d.view(-1, 6)).view(-1, 16, 3, 3).contiguous()
        pred_mano_shape = self.shape_reg(mano_features)
        # pred_mano_shape = torch.zeros_like(gt_mano_params[:, self.mano_pose_size:]).float().cuda()

        pred_mano_pose = mat2aa(pred_mano_pose_rotmat.view(-1, 3, 3)).contiguous().view(-1, self.mano_pose_size)
        pred_verts, pred_joints = self.mano_layer(th_pose_coeffs=pred_mano_pose, th_betas=pred_mano_shape)

        pred_verts /= 1000
        pred_joints /= 1000

# vis
        rot_unique = trimesh.transformations.rotation_matrix(
            np.radians(-90), [0, 1, 0])
        rot_only = rot_unique[:3,:3]
        rot_only = torch.from_numpy(rot_only).float().cuda()
        if gt_mano_params is not None:

            gt_mano_shape = gt_mano_params[:, self.mano_pose_size:].float()
            # gt_mano_shape = torch.zeros_like(gt_mano_params[:, self.mano_pose_size:]).float().cuda()

            gt_mano_pose = gt_mano_params[:, :self.mano_pose_size].contiguous().float()
            gt_mano_pose_rotmat = batch_rodrigues(gt_mano_pose.view(-1, 3)).view(-1, 16, 3, 3)
            gt_verts, gt_joints = self.mano_layer_gt(th_pose_coeffs=gt_mano_pose, th_betas=gt_mano_shape)

            gt_verts /= 1000
            gt_joints /= 1000

            coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
            # gt_joints = gt_joints @ rot_only.T


            gts_new =  gt_joints @ GT_mano_params['all_addition_g'] + GT_mano_params['all_addition_t_no_transl']
            coord_change_mat = torch.from_numpy(coord_change_mat).float().cuda()

            gts_new = gts_new @ coord_change_mat.T  #rot_only
            # gts_new = torch.bmm(gts_new, rot_aug.T)
            gts_new = torch.einsum('bij,bjk->bik', gts_new, torch.transpose(GT_mano_params['rot_aug'], 1, 2))

            # gts_new = gts_new[:, jointsMapManoToSimple,:]
            gts_new_perl = gts_new[:,[0],:]

            gt_verts = gt_verts  @ GT_mano_params['all_addition_g'] + GT_mano_params['all_addition_t_no_transl']
            gt_verts = gt_verts @ coord_change_mat.T #@ rot_only
            gt_verts   =  torch.einsum('bij,bjk->bik', gt_verts, torch.transpose(GT_mano_params['rot_aug'], 1, 2))
            pred_new_pelv = pred_joints[:,[0],:]
            vert_gt_o = gt_verts - gts_new_perl
            gt_mano_results = {
                "verts3d": gt_verts - gts_new_perl,
                "joints3d": gts_new - gts_new_perl,
                "mano_shape": gt_mano_shape,
                "mano_pose": gt_mano_pose_rotmat}

        else:
            gt_mano_results = None

        pred_mano_results = {
            "verts3d": pred_verts - pred_new_pelv,
            "joints3d": pred_joints - pred_new_pelv,
            "mano_shape": pred_mano_shape,
            "mano_pose": pred_mano_pose_rotmat,
            "mano_pose_aa": pred_mano_pose}
        vert_pred_o = pred_verts - pred_new_pelv
        # if randint(1, 100) == 1:
        #     import scipy.io as sio
        #
        #     import open3d as o3d
        #
        #     a = vert_gt_o.detach().cpu().numpy()[0]
        #     b = sio.loadmat('/home/rui/Downloads/rh_face.mat')['rh_face']
        #     mesh = o3d.geometry.TriangleMesh()
        #     mesh.vertices = o3d.utility.Vector3dVector(a)
        #     mesh.triangles = o3d.utility.Vector3iVector(b)
        #     o3d.io.write_triangle_mesh(
        #         '/home/rui/projects/sp2_ws/HandOccNet/debg/1' + '_gt.ply', mesh)
        #     c = vert_pred_o.detach().cpu().numpy()[0]
        #
        #     mesh = o3d.geometry.TriangleMesh()
        #     mesh.vertices = o3d.utility.Vector3dVector(c)
        #     mesh.triangles = o3d.utility.Vector3iVector(b)
        #     o3d.io.write_triangle_mesh(
        #         '/home/rui/projects/sp2_ws/HandOccNet/debg/1' + '_pred.ply', mesh)

        return pred_mano_results, gt_mano_results
