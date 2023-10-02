import sys
sys.path.append('.')
sys.path.append('..')
import torch
import numpy as np
from psbody.mesh.colors import name_to_rgb
import os
import time
from grabnet.tools.meshviewer import Mesh
import trimesh


meshes = torch.load('./OUT/meshes.pt')
save_name = 'test_meshes'
out_dir = os.path.join('./OUT', save_name)
os.makedirs(out_dir, exist_ok=True)
N = len(meshes)

for i in range(N):
    obj = meshes[i][0]
    hand = meshes[i][1]

    hand.set_vertex_colors(vc=[245, 191, 177])

    def_color = np.array([[102, 102, 102, 255]])
    if hasattr(obj.visual, 'vertex_colors') and (obj.visual.vertex_colors == np.repeat(def_color, obj.vertices.shape[0], axis=0)).all():
        obj.set_vertex_colors(vc=name_to_rgb['yellow'])
    combined = trimesh.util.concatenate( [hand, obj] )
    temp = combined.export(os.path.join(out_dir,str(i).zfill(6)+'_Combined.ply'))
    temp = hand.export(os.path.join(out_dir,str(i).zfill(6)+'_Hand.ply'))
    temp = obj.export(os.path.join(out_dir,str(i).zfill(6)+'_Object.ply'))
