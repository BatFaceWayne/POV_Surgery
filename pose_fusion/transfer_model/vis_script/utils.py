import numpy as np
import cv2
import ast
import torch
import torchgeometry as tgm
import torch.nn as nn
import torch.nn.functional as F

class ContinousRotReprDecoder(nn.Module):
    def __init__(self):
        super(ContinousRotReprDecoder, self).__init__()

    def forward(self, module_input):
        reshaped_input = module_input.view(-1, 3, 2)
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def decode(module_input):
        reshaped_input = module_input.view(-1, 3, 2)
        b1 = F.normalize(reshaped_input[:, :, 0], dim=1)
        dot_prod = torch.sum(b1 * reshaped_input[:, :, 1], dim=1, keepdim=True)
        b2 = F.normalize(reshaped_input[:, :, 1] - dot_prod * b1, dim=-1)
        b3 = torch.cross(b1, b2, dim=1)
        return torch.stack([b1, b2, b3], dim=-1)


    @staticmethod
    def matrot2aa(pose_matrot):  # input: [bs, 3, 3]
        '''
        :param pose_matrot: Nx1xnum_jointsx9
        :return: Nx1xnum_jointsx3
        '''
        homogen_matrot = F.pad(pose_matrot.view(-1, 3, 3), [0,1])  # [bs, 3, 4], float
        pose = tgm.rotation_matrix_to_angle_axis(homogen_matrot).view(-1, 3).contiguous()
        return pose

    @staticmethod
    def aa2matrot(pose):
        '''
        :param Nx1xnum_jointsx3
        :return: pose_matrot: Nx1xnum_jointsx9
        '''
        pose_body_matrot = tgm.angle_axis_to_rotation_matrix(pose.reshape(-1, 3))[:, :3, :3].contiguous()
        return pose_body_matrot


def row(A):
    return A.reshape((1, -1))

def col(A):
    return A.reshape((-1, 1))

def unproject_depth_image(depth_image, cam):
    us = np.arange(depth_image.size) % depth_image.shape[1]  # (217088,)  [0,1,2,...,640, ..., 0,1,2,...,640]
    vs = np.arange(depth_image.size) // depth_image.shape[1]  # (217088,)  [0,0,...,0, ..., 576,576,...,576]
    ds = depth_image.ravel()  # (217088,) return flatten depth_image (still the same memory, not a copy)
    uvd = np.array(np.vstack((us.ravel(), vs.ravel(), ds.ravel())).T)  # [576*640, 3]
    # undistort depth map
    xy_undistorted_camspace = cv2.undistortPoints(np.asarray(uvd[:, :2].reshape((1, -1, 2)).copy()), # intrinsic: Distortion coefficients (1x5): k   Camera matrix (3x3): camera_mtx
                                                  np.asarray(cam['camera_mtx']), np.asarray(cam['k']))  # [217088, 1, 2]  camera_mtx (3x3): [f_x, 0, c_x, 0, f_y, c_y, 0,0,0 ]
    # unproject to 3d points in depth cam coord
    xyz_camera_space = np.hstack((xy_undistorted_camspace.squeeze(), col(uvd[:, 2])))  # [217088, 3]
    xyz_camera_space[:, :2] *= col(xyz_camera_space[:, 2])  # scale x,y by z, --> 3d coordinates in depth camera coordinate
    return xyz_camera_space   # [576*640, 3]


def points_coord_trans(xyz_source_coord, trans_mtx):
    # trans_mtx: sourceCoord_2_targetCoord, same as trans in open3d pcd.transform(trans)
    xyz_target_coord = xyz_source_coord.dot(trans_mtx[:3, :3].transpose())  # [N, 3]
    xyz_target_coord = xyz_target_coord + row(trans_mtx[:3, 3])
    return xyz_target_coord


def projectPoints(v, cam):
    v = v.reshape((-1, 3)).copy()
    # return cv2.projectPoints(v, np.asarray(cam['R']), np.asarray(cam['T']), np.asarray(cam['camera_mtx']),
    #                          np.asarray(cam['k']))[0].squeeze()
    return cv2.projectPoints(v, np.asarray([[0.0,0.0,0.0]]), np.asarray([0.0,0.0,0.0]), np.asarray(cam['camera_mtx']),
                             np.asarray(cam['k']))[0].squeeze()


def get_valid_idx(points_color_coord, color_cam, TH=1e-2):
    # 3D points --> 2D coordinates in color image
    uvs = projectPoints(points_color_coord, color_cam)  # [n_depth_points, 2]
    uvs = np.round(uvs).astype(int)
    valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < 1080)  # [n_depth_points], true/false
    valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < 1920)
    valid_idx = np.logical_and(valid_x, valid_y)  # [n_depth_points], true/false
    valid_idx = np.logical_and(valid_idx, points_color_coord[:, 2] > TH)
    uvs = uvs[valid_idx == True]  # valid 2d coords in color img of 3d depth points
    # todo: use human mask here if mask in color img space
    return valid_idx, uvs

def get_valid_idx_2(points_color_coord, color_cam, mask_2, TH=1e-2):
    # 3D points --> 2D coordinates in color image
    uvs = projectPoints(points_color_coord, color_cam)  # [n_depth_points, 2]
    uvs = np.round(uvs).astype(int)
    valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < 1080)  # [n_depth_points], true/false
    valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < 1920)
    valid_idx = np.logical_and(valid_x, valid_y)  # [n_depth_points], true/false
    valid_idx = np.logical_and(valid_idx, points_color_coord[:, 2] > TH)
    # use human mask here if mask in color img space
    valid_idx_copy = valid_idx.copy()
    valid_idx_copy[valid_idx_copy == True] = (mask_2[uvs[valid_idx == True][:, 1], uvs[valid_idx == True][:, 0]] == True)

    uvs = uvs[valid_idx_copy == True]  # valid 2d coords in color img of 3d depth points
    return valid_idx_copy, uvs


def get_valid_idx_3(points_color_coord, color_cam, mask_color, TH=1e-2):
    # 3D points --> 2D coordinates in color image
    uvs = projectPoints(points_color_coord, color_cam)  # [n_depth_points, 2]
    uvs = np.round(uvs).astype(int)
    valid_x = np.logical_and(uvs[:, 1] >= 0, uvs[:, 1] < 1080)  # [n_depth_points], true/false
    valid_y = np.logical_and(uvs[:, 0] >= 0, uvs[:, 0] < 1920)
    valid_idx = np.logical_and(valid_x, valid_y)  # [n_depth_points], true/false
    valid_idx = np.logical_and(valid_idx, points_color_coord[:, 2] > TH)
    # use human mask here if mask in color img space, mask_color: 0-belong to body, 1-bg
    valid_idx_copy = valid_idx.copy()
    valid_idx_copy[valid_idx_copy == True] = (mask_color[uvs[valid_idx == True][:, 1], uvs[valid_idx == True][:, 0]] == False)

    uvs = uvs[valid_idx_copy == True]  # valid 2d coords in color img of 3d depth points
    return valid_idx_copy, uvs


def load_pv_data(csv_path):
    with open(csv_path) as f:
        lines = f.readlines()

    # The first line contains info about the intrinsics.
    # The following lines (one per frame) contain timestamp, focal length and transform PVtoWorld
    n_frames = len(lines) - 1
    frame_timestamps = np.zeros(n_frames, dtype=np.longlong)
    focal_lengths = np.zeros((n_frames, 2))
    pv2world_transforms = np.zeros((n_frames, 4, 4))

    intrinsics_ox, intrinsics_oy, \
        intrinsics_width, intrinsics_height = ast.literal_eval(lines[0])

    for i_frame, frame in enumerate(lines[1:]):
        # Row format is
        # timestamp, focal length (2), transform PVtoWorld (4x4)
        frame = frame.split(',')
        frame_timestamps[i_frame] = int(frame[0])
        focal_lengths[i_frame, 0] = float(frame[1])
        focal_lengths[i_frame, 1] = float(frame[2])
        pv2world_transforms[i_frame] = np.array(frame[3:20]).astype(float).reshape((4, 4))

    return (frame_timestamps, focal_lengths, pv2world_transforms,
            intrinsics_ox, intrinsics_oy, intrinsics_width, intrinsics_height)


def load_head_hand_eye_data(csv_path):
    # joint_count = HandJointIndex.Count.value

    data = np.loadtxt(csv_path, delimiter=',')

    n_frames = len(data)
    timestamps = np.zeros(n_frames)
    head_transs = np.zeros((n_frames, 3))

    # left_hand_transs = np.zeros((n_frames, joint_count, 3))
    # left_hand_transs_available = np.ones(n_frames, dtype=bool)
    # right_hand_transs = np.zeros((n_frames, joint_count, 3))
    # right_hand_transs_available = np.ones(n_frames, dtype=bool)

    # origin (vector, homog) + direction (vector, homog) + distance (scalar)
    gaze_data = np.zeros((n_frames, 9))
    gaze_available = np.ones(n_frames, dtype=bool)

    for i_frame, frame in enumerate(data):
        timestamps[i_frame] = frame[0]
        # head
        head_transs[i_frame, :] = frame[1:17].reshape((4, 4))[:3, 3]
        # # left hand
        # left_hand_transs_available[i_frame] = (frame[17] == 1)
        # left_start_id = 18
        # for i_j in range(joint_count):
        #     j_start_id = left_start_id + 16 * i_j
        #     j_trans = frame[j_start_id:j_start_id + 16].reshape((4, 4))[:3, 3]
        #     left_hand_transs[i_frame, i_j, :] = j_trans
        # # right hand
        # right_hand_transs_available[i_frame] = (
        #     frame[left_start_id + joint_count * 4 * 4] == 1)
        # right_start_id = left_start_id + joint_count * 4 * 4 + 1
        # for i_j in range(joint_count):
        #     j_start_id = right_start_id + 16 * i_j
        #     j_trans = frame[j_start_id:j_start_id + 16].reshape((4, 4))[:3, 3]
        #     right_hand_transs[i_frame, i_j, :] = j_trans

        # assert(j_start_id + 16 == 851)
        gaze_available[i_frame] = (frame[851] == 1)
        gaze_data[i_frame, :4] = frame[852:856]
        gaze_data[i_frame, 4:8] = frame[856:860]
        gaze_data[i_frame, 8] = frame[860]

    # return (timestamps, head_transs, left_hand_transs, left_hand_transs_available,
    #         right_hand_transs, right_hand_transs_available, gaze_data, gaze_available)
    return (timestamps, head_transs, gaze_data, gaze_available)


def get_eye_gaze_point(gaze_data):
    origin_homog = gaze_data[:4]
    direction_homog = gaze_data[4:8]
    direction_homog = direction_homog / np.linalg.norm(direction_homog)
    # if no distance was recorded, set 1m by default
    dist = gaze_data[8] if gaze_data[8] > 0.0 else 1.0
    point = origin_homog + direction_homog * dist

    return point[:3], origin_homog, direction_homog, dist



def match_timestamp(target, all_timestamps):
    return np.argmin([abs(x - target) for x in all_timestamps])


def convert_to_6D_all(x_batch):
    xr_mat = ContinousRotReprDecoder.aa2matrot(x_batch)  # return [:,3,3]
    xr_repr = xr_mat[:, :, :-1].reshape([-1, 6])
    return xr_repr


def convert_to_3D_rot(x_batch):
    '''
    input: [transl, 6d rotation, local params]
    convert global rotation from 6D continuous representation to Eular angle
    '''
    xt = x_batch[:,:3]   # (reconstructed) normalized global translation
    xr = x_batch[:,3:9]  # (reconstructed) 6D rotation vector
    xb = x_batch[:,9:]   # pose $ shape parameters

    xr_mat = ContinousRotReprDecoder.decode(xr)  # [bs,3,3]
    xr_aa = ContinousRotReprDecoder.matrot2aa(xr_mat) # return [:,3]
    return torch.cat([xt, xr_aa, xb], dim=-1)


def gen_body_mesh_v1(body_params, smplx_model, vposer_model, output_mode='verts'):
    # body_params: [T, 3+6+10+32/126 (+180:hands)]
    bs = body_params.shape[0]
    body_params_dict = {}
    body_params_dict['transl'] = body_params[:, 0:3]  # [T, 3]
    body_params_dict['global_orient'] = body_params[:, 3:6]  # [T, 3]
    body_params_dict['betas'] = body_params[:, 6:16]
    body_params_dict['body_pose'] = vposer_model.decode(body_params[:, 16:48], output_type='aa').view(bs, -1)
    body_params_dict['left_hand_pose'] = body_params[:, 48:60]
    body_params_dict['right_hand_pose'] = body_params[:, 60:]

    smplx_output = smplx_model(return_verts=True, **body_params_dict)  # generated human body mesh
    if output_mode == 'verts':
        body_verts = smplx_output.vertices  # [bs, n_body_vert, 3]
        return body_verts
    elif output_mode == 'joints':
        body_joints = smplx_output.joints
        return body_joints
    elif output_mode == 'both':
        body_verts = smplx_output.vertices
        body_joints = smplx_output.joints
        return body_verts, body_joints


@torch.no_grad()
def sample_scene_points(model, smpl_output, scene_vertices, scene_normals=None, n_upsample=2, max_queries=10000):
    points = scene_vertices.clone()
    # remove points that are well outside the SMPL bounding box
    bb_min = smpl_output.vertices.min(1).values.reshape(1, 3)
    bb_max = smpl_output.vertices.max(1).values.reshape(1, 3)

    # # ########## visualize
    # import trimesh
    # trimesh.PointCloud(scene_vertices.detach().cpu().numpy()[0]).export('tmp_scene_vertices.ply')
    # trimesh.PointCloud(smpl_output.vertices.detach().cpu().numpy()[0]).export('tmp_smpl_vertices.ply')
    # # from coap import attach_coap
    # # attach_coap(model, device=points.device)
    # model.coap.extract_mesh(smpl_output)[0].export('tmp_coap_vertices.ply')


    inds = (scene_vertices >= bb_min).all(-1) & (scene_vertices <= bb_max).all(-1)
    if not inds.any():
        return None
    points = scene_vertices[inds]
    model.coap.eval()
    colliding_inds = (model.coap.query(points.reshape((1, -1, 3)), smpl_output) > 0.5).reshape(-1)
    # import pdb; pdb.set_trace()
    model.coap.detach_cache()  # detach variables to enable differentiable pass in the opt. loop
    if not colliding_inds.any():
        return None
    points = points[colliding_inds.reshape(-1)]

    if scene_normals is not None and points.size(0) > 0:  # sample extra points if normals are available
        norms = scene_normals[inds][colliding_inds]

        offsets = 0.5 * torch.normal(0.05, 0.05, size=(points.shape[0] * n_upsample, 1), device=norms.device).abs()
        verts, norms = points.repeat(n_upsample, 1), norms.repeat(n_upsample, 1)
        points = torch.cat((points, (verts - offsets * norms).reshape(-1, 3)), dim=0)

    if points.shape[0] > max_queries:
        points = points[torch.randperm(points.size(0), device=points.device)[:max_queries]]

    return points.float().reshape(1, -1, 3)  # add batch dimension


def draw_gaze_heatmap_2d(H=1080, W=1920, holo_gaze_point2d_dict=None, holo_frame_id=None):
    gaze_heatmap = np.zeros([H, W])
    # color: (1080, 1920, 3)
    us = np.arange(H * W) % W
    vs = np.arange(H * W) // W
    gaze_u = int(holo_gaze_point2d_dict[holo_frame_id][0])
    gaze_v = int(holo_gaze_point2d_dict[holo_frame_id][1])
    gaze_visible = False
    if gaze_u < 1920 and gaze_u > 0 and gaze_v < 1080 and gaze_u > 0:
        gaze_visible = True
        d = (us - gaze_u) ** 2 + (vs - gaze_v) ** 2
        d = d ** 0.5
        d[d > 150] = 150
        # assert np.min(d) == 0
        d = d / np.max(d)  # in [0,1]
        d = 1 - d
        gaze_heatmap = d.reshape([H, W])

    gaze_heatmap = np.uint8(255 * gaze_heatmap)
    gaze_heatmap = cv2.applyColorMap(gaze_heatmap, cv2.COLORMAP_JET)
    # turn into red headmap
    gaze_heatmap[:, :, -1] = 255
    gaze_heatmap[:, :, 0] = 0
    gaze_heatmap[:, :, 1] = 0

    gaze_heatmap = gaze_heatmap[:, :, ::-1]
    gaze_heatmap = cv2.cvtColor(gaze_heatmap, cv2.COLOR_RGB2RGBA)
    if gaze_visible:
        gaze_heatmap[:, :, -1] = d.reshape([H, W]) * 255  # set alpha by distance from gaze
    else:
        gaze_heatmap[:, :, -1] = 0
    gaze_heatmap[:, :, -1] = gaze_heatmap[:, :, -1] * 0.7  # numpy array
    return gaze_heatmap