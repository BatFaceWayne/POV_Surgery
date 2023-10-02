import os
import pickle
from scipy.spatial.transform import Rotation as R
# original_dict = pickle.load(open('/Users/wangrui/Downloads/debug_SMPL/00261.pkl','rb'))
import open3d as o3d
import numpy as np
import scipy
from scipy.io import savemat
from slerp_utils import quaternion_from_matrix, quaternion_slerp, quaternion_matrix


## 889 right eye 333 left eye  626 forehead
def quat_correct(quat):
    """ Converts quaternion to minimize Euclidean distance from previous quaternion (wxyz order) """
    for q in range(1, quat.shape[0]):
        if np.linalg.norm(quat[q-1] - quat[q], axis=0) > np.linalg.norm(quat[q-1] + quat[q], axis=0):
            quat[q] = -quat[q]
    return quat


def quat_smooth(quat, ratio = 0.3):
    """ Converts quaternion to minimize Euclidean distance from previous quaternion (wxyz order) """
    for q in range(1, quat.shape[0]):
        quat[q] = quaternion_slerp(quat[q-1], quat[q], ratio)
    return quat


def smooth_pose_mat(pose, ratio = 0.3):
    quats_all = []
    for j in range(pose.shape[1]):
        quats = []
        for i in range(pose.shape[0]):
            R = pose[i,j,:,:]
            quats.append(quaternion_from_matrix(R))
        quats = quat_correct(np.array(quats))
        quats = quat_smooth(quats, ratio = ratio)
        quats_all.append(np.array([quaternion_matrix(i)[:3,:3] for i in quats]))

    quats_all = np.stack(quats_all, axis=1)
    return quats_all

def smooth_camera(pose, ratio = 0.3):
    from scipy.spatial.transform import Rotation as R
    pose_quat = [R.from_euler('xyz',pose[temp_i,:],degrees=True).as_quat() for temp_i in range(len(pose[:,0]))]
    quats_all = []



    quats = quat_correct(np.array(pose_quat))
    quats = quat_smooth(quats, ratio = ratio)


    # quats_all = np.stack(quats_all, axis=1)
    return R.from_quat(quats).as_euler('xyz',degrees=True)

base_ply = '/home/ray/Downloads/zju-ls-feng/output/smplx/rotated_body_ply'
out_camera = base_ply.replace('rotated_body_ply','texture_rotate')
os.makedirs(out_camera,exist_ok=True)
max_1 = 0
for this_ply in os.listdir(base_ply):
    if '.ply' not in this_ply:
        continue
    temp = int(this_ply.split('.')[0])
    if temp > max_1:
        max_1 = temp
transl_dict = np.zeros((max_1+1,3)) #{}
rot_dict = np.zeros((max_1+1,3)) #{}
grav_trans = None
grav_rot = None
def rotation_matrix_from_vectors(vec1, vec2):
    """ Find the rotation matrix that aligns vec1 to vec2
    :param vec1: A 3d "source" vector
    :param vec2: A 3d "destination" vector
    :return mat: A transform matrix (3x3) which when applied to vec1, aligns it with vec2.
    """
    a, b = (vec1 / np.linalg.norm(vec1)).reshape(3), (vec2 / np.linalg.norm(vec2)).reshape(3)
    v = np.cross(a, b)
    c = np.dot(a, b)
    s = np.linalg.norm(v)
    kmat = np.array([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    rotation_matrix = np.eye(3) + kmat + kmat.dot(kmat) * ((1 - c) / (s ** 2))
    return rotation_matrix
# transl_dict = [] #{}
# rot_dict = [] #{}



for this_ply in os.listdir(base_ply):
    if '.ply' not in this_ply:
        continue
    # all_dict[this_ply] = {}

    mesh = o3d.io.read_triangle_mesh(os.path.join(base_ply, this_ply))
    # print(mesh)
    vert = np.asarray(mesh.vertices)
    forehead = vert[626,:]
    head_top = vert[8153,:]
    left_eye = -vert[9427,:]+ forehead
    right_eye = -vert[10049,:] + forehead
    original_cmaera_rot = np.array([0,0,-1])
    # new_position = -(vert[333,:]+vert[889,:] - 2*head_top)/2
    new_position = -np.cross(left_eye,right_eye )
    # rotation_matrix = new_position.reshape(3,1) @ original_cmaera_rot.reshape(1,3)
    rotation_matrix = rotation_matrix_from_vectors(original_cmaera_rot,new_position)
    if 'capture_01_07_22' in base_ply:
        rotation_matrix = rotation_matrix @ R.from_euler('xyz',[-30,0,0],degrees=True).as_matrix()
    elif 'capture_31_10_22' in base_ply:
        rotation_matrix = rotation_matrix @ R.from_euler('xyz',[15,0,0],degrees=True).as_matrix()
    elif 'caputre_16_12_22' in base_ply and 'david_diskplacer' not in base_ply :
        rotation_matrix = rotation_matrix @ R.from_euler('xyz',[20,0,0],degrees=True).as_matrix()
    elif 'caputre_16_12_22' in base_ply and '_aug' in base_ply and 'david_diskplacer_aug' not in base_ply:
        rotation_matrix = rotation_matrix @ R.from_euler('xyz', [15, 0, 0], degrees=True).as_matrix()
    elif 'caputre_16_12_22' in base_ply and 'david_diskplacer_aug' in base_ply:
        rotation_matrix = rotation_matrix @ R.from_euler('xyz', [7, 0, 0], degrees=True).as_matrix()
    elif 'capture_23_01_23' in base_ply:
        rotation_matrix = rotation_matrix @ R.from_euler('xyz', [-10, 0, 0], degrees=True).as_matrix()
    else:
        rotation_matrix = rotation_matrix @ R.from_euler('xyz', [0, 0, 0], degrees=True).as_matrix()
    r = R.from_matrix(rotation_matrix).as_euler('xyz',degrees=True)
    # def func1(x):
    #     a_t = R.from_euler('xyz',[r[0],x, r[1]],degrees=True).as_matrix()
    #     new_ar = a_t @ np.array([0,1,0])
    #
    #
    #     return new_ar @ (vert[9427,:] - vert[10049,:])

    def func1(x):
        a_t = R.from_euler('xyz',[0,0 ,x[0]],degrees=True).as_matrix()
        new_ar = R.from_euler('xyz',[r[0],r[1] ,r[2]],degrees=True).as_matrix() @ a_t @ R.from_euler('xyz',[0,0 ,-90],degrees=True).as_matrix() @np.array([0,1,0])


        return new_ar @ (vert[9427,:] - vert[10049,:])
    y_original = [0,1,0]
    # print(forehead)
    # print(list(r))
    a = {'hello': 'world'}

    temp = scipy.optimize.minimize(func1, 0, args=(), method='L-BFGS-B', jac=None, bounds=None, tol=None, callback=None, options={'disp': None, 'maxcor': 100, 'ftol': 2.220446049250313e-09, 'gtol': 1e-05, 'eps': 1e-08, 'maxfun': 15000, 'maxiter': 15000, 'iprint': - 1, 'maxls': 20, 'finite_diff_rel_step': None})
  ################################
    # if temp['fun']>0:    # rui seq
    #     result_y = temp['x'][0]-90
    # else:
    #     result_y = temp['x'][0]-90 + 180
    # print(r[0],result_y,r[2])
    ################################

    result_temp = R.from_euler('xyz',[r[0],r[1] ,r[2]],degrees=True).as_matrix() @ R.from_euler('xyz',[0,0 ,temp['x'][0]],degrees=True).as_matrix()
    result_temp = R.from_matrix(result_temp).as_euler('xyz',degrees=True)
    # if grav_trans is None:
    grav_trans = forehead
    grav_rot = np.array([result_temp[0],result_temp[1],result_temp[2]])
    # else:
    #     grav_trans = grav_trans * 0.5 + 0.5 * forehead
    #     grav_rot = grav_rot*0.5 +  0.5 * np.array([result_temp[0],result_temp[1],result_temp[2]])

    this_ply = int(this_ply.split('.')[0])
    transl_dict[this_ply,:] = grav_trans
    # print(result_temp)
    rot_dict[this_ply,:] = grav_rot#np.array([result_temp[0],result_temp[1],result_temp[2]])

savemat(os.path.join(out_camera,'transl_dict_raw.mat'),{'transl_dict':transl_dict})
smooth_camera(rot_dict)
savemat(os.path.join(out_camera,'rot_dict_raw.mat'),{'rot_dict':rot_dict})
# a = R.from_rotvec(original_dict['global_orient'])
# r = R.from_euler('zyx',[0, 90, 0], degrees=True)
# OUT= R.from_matrix(a.as_matrix() @ r.as_matrix())
# original_dict['global_orient'] = OUT.as_rotvec()
# with open('/Users/wangrui/Downloads/debug_SMPL/00261_debug.pkl', 'wb') as handle:
#     pickle.dump(original_dict, handle, protocol=pickle.HIGHEST_PROTOCOL)