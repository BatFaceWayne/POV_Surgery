## Blender Rendering
Please download the blender accordingly from [here](https://www.blender.org/). Then you should install [bpycv](https://github.com/DIYer22/bpycv). Please refer to their repo for instructions. 
## Usage
Firstly render the RGB image with cycle render engine. Then render the depth image with bpycv, as bpycv might mess the textures and materials.
### To use blender RGB rendering
- Launch blender and open the 'render_demo_rgb.blend'. 
- Import the room scene in the data/sim_room folder. 
- Adjust the room location and orientation to a desired position.
- Adjust the light variation in the room.
- Change the body model path and camera path in the python script. 
- Run render with 'CPU' or 'GPU' option. We recommend using GPU and 'optix' option for faster rendering.
### To use blender depth rendering
- Launch blender and open the 'render_demo_depth_mask.blend'. 
- Import the room scene in the blender. 
- Adjust the room location and orientation to *the same location* as in RGB position.
- Change the body model path in the python script. 
- Run render script.
