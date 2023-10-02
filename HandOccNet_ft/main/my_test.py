import os

import torch
import argparse
from tqdm import tqdm
import numpy as np
import torch.backends.cudnn as cudnn
from config import cfg
from base import Tester, MY_VAL


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gpu', type=str, dest='gpu_ids')
    parser.add_argument('--test_epoch', type=str, dest='test_epoch')
    args = parser.parse_args()

    if not args.gpu_ids:
        assert 0, "Please set propoer gpu ids"

    if '-' in args.gpu_ids:
        gpus = args.gpu_ids.split('-')
        gpus[0] = int(gpus[0])
        gpus[1] = int(gpus[1]) + 1
        args.gpu_ids = ','.join(map(lambda x: str(x), list(range(*gpus))))

    # assert args.test_epoch, 'Test epoch is required.'
    return args


def main(path1):
    args = parse_args()
    cfg.set_args(args.gpu_ids)
    cudnn.benchmark = True
### MY_VAL Tester
    tester = MY_VAL()
    tester._make_batch_generator()
    tester._make_model(path1)

    eval_result = {}
    cur_sample_idx = 0

    vert_box = []
    j2t_box = []
    j3d_box = []
    j3d_pa = []
    v3d_pa = []

    for itr, (inputs, targets, meta_info) in enumerate(tqdm(tester.batch_generator)):

        # forward
        with torch.no_grad():
            out = tester.model(inputs, targets, meta_info, 'my_val',this_name=str(itr).zfill(6))

        # save output
        out = {k: v.cpu().numpy() for k, v in out.items()}
        vert_box.append(out['mano_verts'].item())
        j3d_box.append(out['mano_joints'].item())
        j2t_box.append(out['joints_img'].item())
        j3d_pa.append(out['j3d_pa'].item())
        v3d_pa.append(out['v3d_pa'].item())
    print('J2D : ', np.mean(j2t_box))
    print('J3D : ', np.mean(j3d_box))
    print('VERT : ', np.mean(vert_box))
    print('J3D_PA: ', np.mean(j3d_pa))
    print('VERT_PA: ', np.mean(v3d_pa))
    import pickle
    with open(os.path.join('/media/rui/data/demo_statistics/HANDOCCNET.pkl'), 'wb') as f:
        pickle.dump({'J2D': j2t_box}, f)

    OUT_PATH_SCORE = '/home/rui/projects/sp2_ws/HandOccNet/output/eval/ft_ver.txt'
    os.makedirs(os.path.dirname(OUT_PATH_SCORE), exist_ok=True)
    text_file = open(OUT_PATH_SCORE, "a+")
    n = text_file.write('======================================================================================================================================')
    n = text_file.write(path1)
    n = text_file.write('J2D: ' + str(np.mean(j2t_box)))
    n = text_file.write('\n')
    n = text_file.write('J3D: ' + str(np.mean(j3d_box)))
    n = text_file.write('\n')
    n = text_file.write('VERT: ' + str(np.mean(vert_box)))
    n = text_file.write('\n')
    n = text_file.write('J3d_PA: ' + str(np.mean(j3d_pa)))
    n = text_file.write('\n')
    n = text_file.write('V3d_PA: ' + str(np.mean(v3d_pa)))
    n = text_file.write('\n')

    text_file.close()
        # for k, v in out.items(): batch_size = out[k].shape[0]
        # out = [{k: v[bid] for k, v in out.items()} for bid in range(batch_size)]

        # evaluate
        # tester._evaluate(out, cur_sample_idx)
        # cur_sample_idx += len(out)

    # tester._print_eval_result(args.test_epoch)


if __name__ == "__main__":
    # all_ckpt = os.listdir('/home/rui/projects/sp2_ws/HandOccNet/output/model_dump')
    # for i in range(1):
    #     this_check = 48
    #     if os.path.exists(os.path.join('/home/rui/projects/sp2_ws/HandOccNet/output/model_dump/', 'snapshot_' + str(this_check) + '.pth.tar')):
    main(os.path.join('/home/rui/projects/sp2_ws/HandOccNet/demo/snapshot_demo.pth.tar'))
    # for i in range(1):
    #     this_check = 26
    #     if os.path.exists(os.path.join('/home/rui/projects/sp2_ws/HandOccNet/output/model_dump/', 'snapshot_' + str(this_check) + '.pth.tar')):
    #         main('/home/rui/projects/sp2_ws/HandOccNet/demo/snapshot_demo.pth.tar')