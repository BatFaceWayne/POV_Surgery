pip install tensorboardX numpy>=1.16.2 torchgeometry>=0.1.2 pillow tqdm Ninja trimesh pyrender PyYAML loguru omegaconf
cd MANO
pip install -e .
cd ..
pip install git+https://github.com/MPI-IS/mesh.git
pip install git+https://github.com/otaheri/chamfer_distance
pip install git+https://github.com/otaheri/bps_torch
pip install pybullet
pip install sk-video
pip install rtree
git clone https://github.com/vchoutas/torch-trust-ncg.git
cd torch-trust-ncg
python setup.py install
pip install einops

