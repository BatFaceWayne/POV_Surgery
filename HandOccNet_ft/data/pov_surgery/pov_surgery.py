import os
import os.path as osp
import numpy as np
import torch
from PIL import Image, ImageFilter
import cv2
from torchvision.transforms import functional
import random
import json
import math
import copy
from pycocotools.coco import COCO
from config import cfg
from utils.preprocessing import load_img, get_bbox, process_bbox, generate_patch_image, augmentation
from utils.transforms import world2cam, cam2pixel, pixel2cam, rigid_align, transform_joint_to_other_db
from utils.vis import vis_keypoints, vis_mesh, save_obj, vis_keypoints_with_skeleton
from utils.mano import MANO
import pickle
import data.pov_surgery.datautil as dataset_util
mano = MANO()


class POVSURGERY(torch.utils.data.Dataset):
    def __init__(self, transform, data_split):
        self.transform = transform
        self.data_split = data_split
        # self.root_dir = osp.join('..', 'data', 'HO3D', 'data')
        # self.annot_path = osp.join(self.root_dir, 'annotations')
        # self.root_dir = osp.join('/media/rui/data/SP_2_data/HO3D_v2/HO3D_v2')
        self.root_joint_idx = 0
        self.hue = 0.15
        self.contrast = 0.5
        self.brightness = 0.5
        self.saturation = 0.5
        self.blur_radius = 0.5
        self.scale_jittering = 0.3
        self.center_jittering = 0.15
        self.max_rot = 3.1415926

        self.coord_change_mat = np.array([[1., 0., 0.], [0, -1., 0.], [0., 0., -1.]], dtype=np.float32)
        self.inp_res = 256



        if self.data_split == 'train':
            self.mode = 'train'
            self.base_info = pickle.load(open('/home/ray/code_release/pov_surgery_dataset/POV_Surgery_data/handoccnet_train/2d_repro_ho3d_style_hocc_cleaned.pkl', 'rb'))
            self.set_list = list(self.base_info.keys())
        elif self.data_split == 'validation':
            self.mode = 'validation'
            self.base_info = pickle.load(open('/home/ray/code_release/pov_surgery_dataset/POV_Surgery_data/handoccnet_train/2d_repro_ho3d_style_test_cleaned.pkl', 'rb'))
            self.set_list = list(self.base_info.keys())
        else:
            self.mode = 'demo'
            self.base_info = pickle.load(open('/media/rui/mac_data/POV_surgery/demo_idx_selected.pkl', 'rb'))
            temp_list =  self.base_info['imgname']
            temp_anno = self.base_info['annoname']
            temp_o = []
            for temp_i in range(len(temp_list)):
                temp_o.append(os.path.join(temp_anno[temp_i].split('ego_eval')[0],'images','6', temp_list[temp_i]))
            self.set_list = temp_o



        # if self.data_split != 'train':
        #     self.eval_result = [[], []]  # [pred_joints_list, pred_verts_list]
        self.joints_name = (
        'Wrist', 'Index_1', 'Index_2', 'Index_3', 'Middle_1', 'Middle_2', 'Middle_3', 'Pinky_1', 'Pinky_2', 'Pinky_3',
        'Ring_1', 'Ring_2', 'Ring_3', 'Thumb_1', 'Thumb_2', 'Thumb_3', 'Thumb_4', 'Index_4', 'Middle_4', 'Ring_4',
        'Pinly_4')

    # def load_data(self):
    #     db = COCO(osp.join(self.annot_path, "HO3D_{}_data.json".format(self.data_split)))
    #
    #     datalist = []
    #     for aid in db.anns.keys():
    #         ann = db.anns[aid]
    #         image_id = ann['image_id']
    #         img = db.loadImgs(image_id)[0]
    #         img_path = osp.join(self.root_dir, self.data_split, img['file_name'])
    #         img_shape = (img['height'], img['width'])
    #         if self.data_split == 'train':
    #             joints_coord_cam = np.array(ann['joints_coord_cam'], dtype=np.float32)  # meter
    #             cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
    #             joints_coord_img = cam2pixel(joints_coord_cam, cam_param['focal'], cam_param['princpt'])
    #             bbox = get_bbox(joints_coord_img[:, :2], np.ones_like(joints_coord_img[:, 0]), expansion_factor=1.5)
    #             bbox = process_bbox(bbox, img['width'], img['height'], expansion_factor=1.0)
    #             if bbox is None:
    #                 continue
    #
    #             mano_pose = np.array(ann['mano_param']['pose'], dtype=np.float32)
    #             mano_shape = np.array(ann['mano_param']['shape'], dtype=np.float32)
    #
    #             data = {"img_path": img_path, "img_shape": img_shape, "joints_coord_cam": joints_coord_cam,
    #                     "joints_coord_img": joints_coord_img,
    #                     "bbox": bbox, "cam_param": cam_param, "mano_pose": mano_pose, "mano_shape": mano_shape}
    #         else:
    #             root_joint_cam = np.array(ann['root_joint_cam'], dtype=np.float32)
    #             cam_param = {k: np.array(v, dtype=np.float32) for k, v in ann['cam_param'].items()}
    #             bbox = np.array(ann['bbox'], dtype=np.float32)
    #             bbox = process_bbox(bbox, img['width'], img['height'], expansion_factor=1.5)
    #
    #             data = {"img_path": img_path, "img_shape": img_shape, "root_joint_cam": root_joint_cam,
    #                     "bbox": bbox, "cam_param": cam_param}
    #
    #         datalist.append(data)
    #
    #     return datalist
    def data_aug_val(self, img, mano_param, joints_uv, K, gray, p2d):
        crop_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_hand, img.size)

        # Randomly jitter center
        # center_offsets = (self.center_jittering * scale * np.random.uniform(low=-1, high=1, size=2))
        # center = center + center_offsets

        # Scale jittering
        # scale_jittering = self.scale_jittering * np.random.randn() + 1
        # scale_jittering = np.clip(scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
        # scale = scale * scale_jittering

        rot = 1e-9#np.random.uniform(low=-self.max_rot, high=self.max_rot)
        affinetrans, post_rot_trans, rot_mat = dataset_util.get_affine_transform(center, scale,
                                                                                 [self.inp_res, self.inp_res], rot=rot,
                                                                                 K=K)
        # Change mano from openGL coordinates to normal coordinates
        # mano_param[:3] = dataset_util.rotation_angle(mano_param[:3], rot_mat, coord_change_mat=self.coord_change_mat)

        joints_uv = dataset_util.transform_coords(joints_uv, affinetrans)  # hand landmark trans
        K = post_rot_trans.dot(K)

        p2d = dataset_util.transform_coords(p2d, affinetrans)  # obj landmark trans
        # get hand bbox and normalize landmarks to [0,1]
        bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.1)
        joints_uv = dataset_util.normalize_joints(joints_uv, bbox_hand)

        # get obj bbox and normalize landmarks to [0,1]
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        p2d = dataset_util.normalize_joints(p2d, bbox_obj)

        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))

        # Img blurring and color jitter
        # blur_radius = random.random() * self.blur_radius
        # img = img.filter(ImageFilter.GaussianBlur(blur_radius))
        # img = dataset_util.color_jitter(img, brightness=self.brightness,
        #                                 saturation=self.saturation, hue=self.hue, contrast=self.contrast)

        # Generate object mask: gray segLabel transform and crop
        gray = dataset_util.transform_img(gray, affinetrans, [self.inp_res, self.inp_res])
        # gray = gray.crop((0, 0, self.inp_res, self.inp_res))
        # gray = dataset_util.get_mask_ROI(gray, bbox_obj)
        # # Generate object mask
        # gray = np.asarray(gray.resize((32, 32), Image.NEAREST))
        # obj_mask = np.ma.getmaskarray(np.ma.masked_not_equal(gray, 0)).astype(int)
        # obj_mask = torch.from_numpy(obj_mask)
        obj_mask = torch.from_numpy(np.zeros(3))

        return img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj, rot_mat

    def motion_blur(self, img, blur, kernel_size=3):
        kernel_1 = np.zeros((kernel_size, kernel_size))
        kernel_1[kernel_size // 2, :] = 1 / kernel_size
        kernel_2 = kernel_1.T
        kernel_3 = np.eye(kernel_size) / kernel_size
        kernel_4 = np.flip(kernel_3, 0)
        kernels = [kernel_1, kernel_2, kernel_3, kernel_4]
        blur_img = cv2.filter2D(img, -1, kernels[blur])
        return blur_img
    def data_aug(self, img, mano_param, joints_uv, K, gray, p2d):
        crop_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.5)
        crop_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.5)
        center, scale = dataset_util.fuse_bbox(crop_hand, crop_hand, img.size)

        # Randomly jitter center
        center_offsets = (self.center_jittering * scale * np.random.uniform(low=-1, high=1, size=2))
        center = center + center_offsets

        # Scale jittering
        scale_jittering = self.scale_jittering * np.random.randn() + 1
        scale_jittering = np.clip(scale_jittering, 1 - self.scale_jittering, 1 + self.scale_jittering)
        scale = scale * scale_jittering

        rot = 1e-9
        # rot = np.random.uniform(low=-self.max_rot, high=self.max_rot)
        affinetrans, post_rot_trans, rot_mat = dataset_util.get_affine_transform(center, scale,
                                                                                 [self.inp_res, self.inp_res], rot=rot,
                                                                                 K=K)
        # Change mano from openGL coordinates to normal coordinates
        # mano_param[:3] = dataset_util.rotation_angle(mano_param[:3], rot_mat, coord_change_mat=self.coord_change_mat)

        joints_uv = dataset_util.transform_coords(joints_uv, affinetrans)  # hand landmark trans
        K = post_rot_trans.dot(K)

        p2d = dataset_util.transform_coords(p2d, affinetrans)  # obj landmark trans
        # get hand bbox and normalize landmarks to [0,1]
        bbox_hand = dataset_util.get_bbox_joints(joints_uv, bbox_factor=1.1)
        joints_uv = dataset_util.normalize_joints(joints_uv, bbox_hand)

        # get obj bbox and normalize landmarks to [0,1]
        bbox_obj = dataset_util.get_bbox_joints(p2d, bbox_factor=1.0)
        p2d = dataset_util.normalize_joints(p2d, bbox_obj)

        # Transform and crop
        img = dataset_util.transform_img(img, affinetrans, [self.inp_res, self.inp_res])
        img = img.crop((0, 0, self.inp_res, self.inp_res))

        # Img blurring and color jitter
        blur_radius = random.random() * self.blur_radius
        img = img.filter(ImageFilter.GaussianBlur(blur_radius))

        BLUR_SIZES = [3, 5, 7]
        blur = np.random.randint(4)
        blur_size = np.random.randint(len(BLUR_SIZES))
        blur_size = BLUR_SIZES[blur_size]
        open_cv_image = np.array(img)
        rgb_img = open_cv_image[:, :, ::-1].copy()
        rgb_img = self.motion_blur(rgb_img, blur, blur_size)
        img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2RGB)
        img = Image.fromarray(img)

        img = dataset_util.color_jitter(img, brightness=self.brightness,
                                        saturation=self.saturation, hue=self.hue, contrast=self.contrast)

        # Generate object mask: gray segLabel transform and crop
        gray = dataset_util.transform_img(gray, affinetrans, [self.inp_res, self.inp_res])
        gray = gray.crop((0, 0, self.inp_res, self.inp_res))
        gray = dataset_util.get_mask_ROI(gray, bbox_obj)
        # Generate object mask
        gray = np.asarray(gray.resize((32, 32), Image.NEAREST))
        obj_mask = np.ma.getmaskarray(np.ma.masked_not_equal(gray, 0)).astype(int)
        obj_mask = torch.from_numpy(obj_mask)

        return img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj, rot_mat
    # def __len__(self):
    #     return len(self.set_list)
    def __len__(self):
        return len(self.set_list)
    def __getitem__(self, idx):
        sample = {}
        BASE_DATA = '/home/ray/code_release/pov_surgery_dataset/POV_Surgery_data'
        if self.mode != 'demo':
            seqName, id = self.set_list[idx].split("/")
            img = Image.open(os.path.join(BASE_DATA,'color', seqName, id + '.jpg')).convert("RGB")
            frame_anno = pickle.load(open(os.path.join(BASE_DATA,'annotation', seqName, id + '.pkl'), 'rb'))

            sample["seqName"] = seqName
            sample["id"] = id
            if self.mode == 'train':
                joints_uv_temp = self.base_info[self.set_list[idx]]['joints_uv']
                p2d_temp = self.base_info[self.set_list[idx]]['p2d']
                K = np.array([[1198.4395, 0.0000, 960.0000], [0.0000, 1198.4395, 175.2000], [0.0000, 0.0000, 1.0000]])
                p2d = np.zeros_like(p2d_temp)
                p2d[:, 0] = p2d_temp[:, 1]
                p2d[:, 1] = p2d_temp[:, 0]

                joints_uv = np.zeros_like(joints_uv_temp)
                # joints_uv[:,0] = 1920 - joints_uv_temp[:,1]
                joints_uv[:, 1] = joints_uv_temp[:, 0]
                joints_uv[:, 0] = joints_uv_temp[:, 1]

                # K = self.K[idx]
                # self.vis_kp(img, joints_uv)

                mano_param_temp = frame_anno['mano']
                this_rot = frame_anno['cam_rot']
                this_transl = frame_anno['cam_transl']
                camera_pose = np.eye(4)
                camera_pose[:3, 3] = this_transl
                camera_pose[:3, :3] = this_rot
                all_addition_g = frame_anno['grab2world_R'] @ np.linalg.inv(camera_pose)[:3, :3].T
                all_addition_t = (frame_anno['grab2world_T'] @ np.linalg.inv(camera_pose)[:3, :3].T
                                  + np.linalg.inv(camera_pose)[:3, 3])
                temp_tl = mano_param_temp['transl']
                all_addition_t_no_transl = temp_tl @ all_addition_g + all_addition_t
                variance_list = np.random.normal(0., 0.1, size=(1, 45))
                mano_param = \
                np.concatenate((mano_param_temp['global_orient'], mano_param_temp['hand_pose'], mano_param_temp['betas']),
                               1)[0]

                # self.mano_params[idx]

                # object information
                gray = Image.open(os.path.join(BASE_DATA,'mask/', seqName, id + '.png'))
                gray = np.asarray(gray)
                gray = gray.copy()
                # if not (np.any(gray==100) and np.any(gray==200)):
                #     return {}
                gray[gray > 199] = 255
                gray[gray < 199] = 0
                gray = Image.fromarray(np.uint8(gray))
                if self.mode == 'train':
                    img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj, rot_aug = self.data_aug(img, mano_param,
                                                                                                               joints_uv, K,
                                                                                                               gray, p2d)
                else:
                    img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj, rot_aug = self.data_aug_val(img,
                                                                                                               mano_param,
                                                                                                               joints_uv, K,
                                                                                                               gray, p2d)
                # sample["img"] = functional.to_tensor(img) /255
                sample["bbox_hand"] = bbox_hand
                sample["bbox_obj"] = bbox_obj
                sample["mano_param"] = mano_param
                sample["cam_intr"] = K
                sample["joints2d"] = joints_uv
                # sample["obj_p2d"] = p2d
                # sample["obj_mask"] = obj_mask
                sample["all_addition_g"] = all_addition_g
                sample["all_addition_t_no_transl"] = all_addition_t_no_transl
                sample['rot_aug'] = rot_aug
                inputs = {'img': functional.to_tensor(img)}
                targets = sample
                meta_info = {'root_joint_cam': rot_aug}
            else:
                joints_uv_temp = self.base_info[self.set_list[idx]]['joints_uv']
                p2d_temp = self.base_info[self.set_list[idx]]['p2d']
                K = np.array([[1198.4395, 0.0000, 960.0000], [0.0000, 1198.4395, 175.2000], [0.0000, 0.0000, 1.0000]])
                p2d = np.zeros_like(p2d_temp)
                p2d[:, 0] = p2d_temp[:, 1]
                p2d[:, 1] = p2d_temp[:, 0]

                joints_uv = np.zeros_like(joints_uv_temp)
                # joints_uv[:,0] = 1920 - joints_uv_temp[:,1]
                joints_uv[:, 1] = joints_uv_temp[:, 0]
                joints_uv[:, 0] = joints_uv_temp[:, 1]

                # K = self.K[idx]
                # self.vis_kp(img, joints_uv)
                joints_uv_orignal = joints_uv.copy()

                mano_param_temp = frame_anno['mano']
                this_rot = frame_anno['cam_rot']
                this_transl = frame_anno['cam_transl']
                camera_pose = np.eye(4)
                camera_pose[:3, 3] = this_transl
                camera_pose[:3, :3] = this_rot
                all_addition_g = frame_anno['grab2world_R'] @ np.linalg.inv(camera_pose)[:3, :3].T
                all_addition_t = (frame_anno['grab2world_T'] @ np.linalg.inv(camera_pose)[:3, :3].T
                                  + np.linalg.inv(camera_pose)[:3, 3])
                temp_tl = mano_param_temp['transl']
                all_addition_t_no_transl = temp_tl @ all_addition_g + all_addition_t
                variance_list = np.random.normal(0., 0.1, size=(1, 45))
                mano_param = \
                    np.concatenate(
                        (mano_param_temp['global_orient'], mano_param_temp['hand_pose'], mano_param_temp['betas']),
                        1)[0]

                # self.mano_params[idx]

                # object information
                gray = Image.open(os.path.join(BASE_DATA,'mask', seqName, id + '.png'))
                gray = np.asarray(gray)
                gray = gray.copy()
                # if not (np.any(gray==100) and np.any(gray==200)):
                #     return {}
                gray[gray > 199] = 255
                gray[gray < 199] = 0
                gray = Image.fromarray(np.uint8(gray))
                img_original = img.copy()
                if self.mode == 'train':
                    img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj, rot_aug = self.data_aug(img,
                                                                                                               mano_param,
                                                                                                               joints_uv,
                                                                                                               K,
                                                                                                               gray,
                                                                                                               p2d)
                else:
                    img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj, rot_aug = self.data_aug_val(img,
                                                                                                                   mano_param,
                                                                                                                   joints_uv,
                                                                                                                   K,
                                                                                                                   gray,
                                                                                                                   p2d)

                # self.draw_box(img_original, dataset_util.get_bbox_joints(joints_uv_orignal, bbox_factor=1.5))
                scale_temp = dataset_util.get_bbox_joints(joints_uv_orignal, bbox_factor=1.5)
                scale_img = scale_temp[2] - scale_temp[0]
                # sample["img"] = functional.to_tensor(img) /255
                sample["bbox_hand"] = bbox_hand
                sample["scale_img"] = scale_img
                sample["bbox_obj"] = bbox_obj
                sample["mano_param"] = mano_param
                sample["cam_intr"] = K
                sample["joints2d"] = joints_uv
                # sample["obj_p2d"] = p2d
                # sample["obj_mask"] = obj_mask
                sample["all_addition_g"] = all_addition_g
                sample["all_addition_t_no_transl"] = all_addition_t_no_transl
                sample['rot_aug'] = rot_aug
                inputs = {'img': functional.to_tensor(img)}
                targets = sample
                meta_info = {'root_joint_cam': rot_aug}
        else:
            this_image_name = self.set_list[idx]

            this_json = pickle.load(open(self.base_info['annoname'][idx], 'rb'))

            kp2d_this = this_json['kp2d']
            bbox1 = np.array(
                [np.min(kp2d_this[:, 0]), np.min(kp2d_this[:, 1]), np.max(kp2d_this[:, 0]), np.max(kp2d_this[:, 1])])
            img = Image.open(this_image_name).convert(
                "RGB")


            seqName = 'd_friem_2'
            id = '00011'
            frame_anno = pickle.load(
                open('/media/rui/mac_data/POV_surgery/annotation/d_diskplacer_2/00011.pkl', 'rb'))

            sample["seqName"] = seqName
            sample["id"] = id
            jointsMapManoToSimple = [0, 13, 14, 15, 16,
                                          1, 2, 3, 17,
                                          4, 5, 6, 18,
                                          10, 11, 12, 19,
                                          7, 8, 9, 20]
            jointsMapSimpleToMano = np.argsort(jointsMapManoToSimple)
            joints_uv_temp  = kp2d_this[jointsMapSimpleToMano,:2]
            p2d_temp = joints_uv_temp
            K = np.array([[1198.4395, 0.0000, 960.0000], [0.0000, 1198.4395, 175.2000], [0.0000, 0.0000, 1.0000]])
            p2d = np.zeros_like(p2d_temp)
            p2d[:, 0] = p2d_temp[:, 0]
            p2d[:, 1] = p2d_temp[:, 1]

            joints_uv = np.zeros_like(joints_uv_temp)
            # joints_uv[:,0] = 1920 - joints_uv_temp[:,1]
            joints_uv[:, 0] = joints_uv_temp[:, 0]
            joints_uv[:, 1] = joints_uv_temp[:, 1]

            # K = self.K[idx]
            # self.vis_kp(img, joints_uv)
            joints_uv_orignal = joints_uv.copy()

            mano_param_temp = frame_anno['mano']
            this_rot = frame_anno['cam_rot']
            this_transl = frame_anno['cam_transl']
            camera_pose = np.eye(4)
            camera_pose[:3, 3] = this_transl
            camera_pose[:3, :3] = this_rot
            all_addition_g = frame_anno['grab2world_R'] @ np.linalg.inv(camera_pose)[:3, :3].T
            all_addition_t = (frame_anno['grab2world_T'] @ np.linalg.inv(camera_pose)[:3, :3].T
                              + np.linalg.inv(camera_pose)[:3, 3])
            temp_tl = mano_param_temp['transl']
            all_addition_t_no_transl = temp_tl @ all_addition_g + all_addition_t
            variance_list = np.random.normal(0., 0.1, size=(1, 45))
            mano_param = \
                np.concatenate(
                    (mano_param_temp['global_orient'], mano_param_temp['hand_pose'], mano_param_temp['betas']),
                    1)[0]

            # self.mano_params[idx]

            # object information
            gray = Image.open(os.path.join('/media/rui/mac_data/POV_surgery/mask/', seqName, id + '.png'))
            gray = np.asarray(gray)
            gray = gray.copy()
            # if not (np.any(gray==100) and np.any(gray==200)):
            #     return {}
            gray[gray > 199] = 255
            gray[gray < 199] = 0
            gray = Image.fromarray(np.uint8(gray))
            img_original = img.copy()
            if self.mode == 'train':
                img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj, rot_aug = self.data_aug(img,
                                                                                                           mano_param,
                                                                                                           joints_uv,
                                                                                                           K,
                                                                                                           gray,
                                                                                                           p2d)
            else:
                img, mano_param, K, obj_mask, p2d, joints_uv, bbox_hand, bbox_obj, rot_aug = self.data_aug_val(img,
                                                                                                               mano_param,
                                                                                                               joints_uv,
                                                                                                               K,
                                                                                                               gray,
                                                                                                               p2d)

            # self.draw_box(img_original, dataset_util.get_bbox_joints(joints_uv_orignal, bbox_factor=1.5))
            scale_temp = dataset_util.get_bbox_joints(joints_uv_orignal, bbox_factor=1.5)
            scale_img = scale_temp[2] - scale_temp[0]
            # sample["img"] = functional.to_tensor(img) /255
            sample["bbox_hand"] = bbox_hand
            sample["scale_img"] = scale_img
            sample["bbox_obj"] = bbox_obj
            sample["mano_param"] = mano_param
            sample["cam_intr"] = K
            sample["joints2d"] = joints_uv
            # sample["obj_p2d"] = p2d
            # sample["obj_mask"] = obj_mask
            sample["all_addition_g"] = all_addition_g
            sample["all_addition_t_no_transl"] = all_addition_t_no_transl
            sample['rot_aug'] = rot_aug
            inputs = {'img': functional.to_tensor(img)}
            targets = sample
            meta_info = {'root_joint_cam': rot_aug}

        return inputs, targets, meta_info

    def draw_box(self, original_img, bbox):
        original_img = np.asarray(original_img).astype(np.uint8)
        bbox_vis = np.array(bbox, int)
        # bbox_vis[2:] += bbox_vis[:2]
        cvimg = cv2.rectangle(original_img.copy(), bbox_vis[:2], bbox_vis[2:], (255, 0, 0), 3)
        cv2.imwrite('/home/rui/projects/sp2_ws/HandOccNet/debg/hand_box.jpg', cvimg[:, :, ::-1])
    def OLD__getitem__(self, idx):
        data = copy.deepcopy(self.datalist[idx])
        img_path, img_shape, bbox = data['img_path'], data['img_shape'], data['bbox']

        # img
        img = load_img(img_path)
        img, img2bb_trans, bb2img_trans, rot, scale = augmentation(img, bbox, self.data_split, do_flip=False)
        img = self.transform(img.astype(np.float32)) / 255.

        if self.data_split == 'train':
            ## 2D joint coordinate
            joints_img = data['joints_coord_img']
            joints_img_xy1 = np.concatenate((joints_img[:, :2], np.ones_like(joints_img[:, :1])), 1)
            joints_img = np.dot(img2bb_trans, joints_img_xy1.transpose(1, 0)).transpose(1, 0)[:, :2]
            # normalize to [0,1]
            joints_img[:, 0] /= cfg.input_img_shape[1]
            joints_img[:, 1] /= cfg.input_img_shape[0]

            ## 3D joint camera coordinate
            joints_coord_cam = data['joints_coord_cam']
            root_joint_cam = copy.deepcopy(joints_coord_cam[self.root_joint_idx])
            joints_coord_cam -= joints_coord_cam[self.root_joint_idx, None, :]  # root-relative
            # 3D data rotation augmentation
            rot_aug_mat = np.array([[np.cos(np.deg2rad(-rot)), -np.sin(np.deg2rad(-rot)), 0],
                                    [np.sin(np.deg2rad(-rot)), np.cos(np.deg2rad(-rot)), 0],
                                    [0, 0, 1]], dtype=np.float32)
            joints_coord_cam = np.dot(rot_aug_mat, joints_coord_cam.transpose(1, 0)).transpose(1, 0)

            ## mano parameter
            mano_pose, mano_shape = data['mano_pose'], data['mano_shape']
            # 3D data rotation augmentation
            mano_pose = mano_pose.reshape(-1, 3)
            root_pose = mano_pose[self.root_joint_idx, :]
            root_pose, _ = cv2.Rodrigues(root_pose)
            root_pose, _ = cv2.Rodrigues(np.dot(rot_aug_mat, root_pose))
            mano_pose[self.root_joint_idx] = root_pose.reshape(3)
            mano_pose = mano_pose.reshape(-1)

            inputs = {'img': img}
            targets = {'joints_img': joints_img, 'joints_coord_cam': joints_coord_cam, 'mano_pose': mano_pose,
                       'mano_shape': mano_shape}
            meta_info = {'root_joint_cam': root_joint_cam}

        else:
            root_joint_cam = data['root_joint_cam']
            inputs = {'img': img}
            targets = {}
            meta_info = {'root_joint_cam': root_joint_cam}

        return inputs, targets, meta_info

    def evaluate(self, outs, cur_sample_idx):
        annots = self.datalist
        sample_num = len(outs)
        for n in range(sample_num):
            annot = annots[cur_sample_idx + n]

            out = outs[n]

            verts_out = out['mesh_coord_cam']
            joints_out = out['joints_coord_cam']

            # root align
            gt_root_joint_cam = annot['root_joint_cam']
            verts_out = verts_out - joints_out[self.root_joint_idx] + gt_root_joint_cam
            joints_out = joints_out - joints_out[self.root_joint_idx] + gt_root_joint_cam

            # convert to openGL coordinate system.
            verts_out *= np.array([1, -1, -1])
            joints_out *= np.array([1, -1, -1])

            # convert joint ordering from MANO to HO3D.
            joints_out = transform_joint_to_other_db(joints_out, mano.joints_name, self.joints_name)

            self.eval_result[0].append(joints_out.tolist())
            self.eval_result[1].append(verts_out.tolist())

    def print_eval_result(self, test_epoch):
        output_json_file = osp.join(cfg.result_dir, 'pred{}.json'.format(test_epoch))
        output_zip_file = osp.join(cfg.result_dir, 'pred{}.zip'.format(test_epoch))

        with open(output_json_file, 'w') as f:
            json.dump(self.eval_result, f)
        print('Dumped %d joints and %d verts predictions to %s' % (
        len(self.eval_result[0]), len(self.eval_result[1]), output_json_file))

        cmd = 'zip -j ' + output_zip_file + ' ' + output_json_file
        print(cmd)
        os.system(cmd)

