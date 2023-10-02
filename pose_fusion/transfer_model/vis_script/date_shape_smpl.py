import pickle
import os
import os.path as osp
import pandas as pd
import glob
from tqdm import tqdm
import numpy as np

data_release_root = '/media/rui/mac_data/Technopark_Recordings/capture_14_7_22_tools/marc_friem_720_1/output/smplx/smplx_fitted_pkl'
# # fitting_root = osp.join(data_release_root, 'smpl_interactee')
# fitting_root = osp.join(data_release_root, 'smpl_camera_wearer')  # todo

# df = pd.read_csv(os.path.join(data_release_root, 'data_info_release.csv'))
# recording_name_list = os.listdir('/home/rui/projects/virtual_human_ws/egobody_release/smpl_camera_wearer')

# for recording_name in tqdm(recording_name_list):
# smpl_interactee_dir = glob.glob(osp.join(fitting_root, recording_name, 'body_idx_*', 'results'))[0]
# pkl_list = glob.glob(osp.join(smpl_interactee_dir, 'frame_*'))
# pkl_list = sorted(pkl_list)
for pkl_path in os.listdir(data_release_root):
    if '.pkl' not in pkl_path:
        continue
    path = os.path.join(data_release_root, pkl_path)
    with open(path, 'rb') as f:
        param = pickle.load(f)
    for key in param.keys():
        param[key] = np.float32(param[key])
        if param[key].shape[0] != 1:
            param[key] = param[key].reshape(1, param[key].shape[0])
    with open(path, 'wb') as result_file:
        pickle.dump(param, result_file, protocol=2)