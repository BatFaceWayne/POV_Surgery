import open3d as o3d
import numpy as np
import os
import shutil
from tqdm import tqdm

ROOT_DIR = '/home/ray/Downloads/zju-ls-feng/output/smplx'
OUT_dir = os.path.join(ROOT_DIR, 'texture_rotate')
TMP_dir = os.path.join(ROOT_DIR, 'tmp')
os.makedirs(TMP_dir,exist_ok=True)
os.makedirs(OUT_dir,exist_ok=True)
mesh = o3d.io.read_triangle_mesh('/home/ray/code_release/hand_texture/transfer_surgical_Source/test1am.obj',enable_post_processing=True)
mesh_save = mesh
BASE_mesh = os.path.join(ROOT_DIR, 'rotated_body_ply')


all_file_list = []
for i in (range(0, 10000)):
    this_mesh = os.path.join(BASE_mesh,str(i).zfill(5)+'.ply')
    if not os.path.exists(this_mesh):
        continue
    all_file_list.append(this_mesh)


for i in tqdm(range(0, len(all_file_list)+1)):
    this_mesh = os.path.join(BASE_mesh,str(i).zfill(5)+'.ply')
    if not os.path.exists(this_mesh):
        continue
    temp = o3d.io.read_triangle_mesh(this_mesh)
    # mesh.triangle_uvs
    mesh = mesh_save
    mesh.vertices = temp.vertices
    mesh.triangles = temp.triangles
    # os.rename(os.path.join(OUT_dir, 'default_smplx_male.obj'), os.path.join(OUT_dir, str(i).zfill(5) + '.obj'))
    # o3d.io.write_triangle_mesh(os.path.join(OUT_dir,'default_smplx_male.obj'),mesh)
    o3d.io.write_triangle_mesh(os.path.join(TMP_dir, 'default_smplx_male.obj'), mesh)
    shutil.move(os.path.join(TMP_dir, 'default_smplx_male.obj'),os.path.join(OUT_dir, str(i).zfill(5) + '.obj'))
    if i == 0 :
        shutil.move(os.path.join(TMP_dir, 'default_smplx_male.mtl'), os.path.join(OUT_dir, 'default_smplx_male.mtl'))
        shutil.move(os.path.join(TMP_dir, 'default_smplx_male_0.png'), os.path.join(OUT_dir, 'default_smplx_male_0.png'))

