import json
import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio
pred_file = open('/home/rui/Downloads/v2v_loss_transfer_previous.json').read()
pred_list = json.loads(pred_file)
interatee_loss = []
camera_wearer_loss = []
id_x = []
for i, key in enumerate(pred_list):
    temp = pred_list[key]
    id_x.append(key)
    interatee_loss.append(temp[0])
    camera_wearer_loss.append(temp[1])
interatee_loss = np.array(interatee_loss)
camera_wearer_loss = np.array(camera_wearer_loss)
id_x = np.array(id_x)

issue_interactee = id_x[interatee_loss>0.005]
issue_camera_wearer = id_x[camera_wearer_loss>0.005]

issue_interactee_seqence = []
issue_camera_wearer_sequence = []
for i in issue_interactee:
    if i.split('/')[0] not in (issue_interactee_seqence):
        issue_interactee_seqence.append(i.split('/')[0])

for i in issue_camera_wearer:
    if i.split('/')[0] not in (issue_camera_wearer_sequence):
        issue_camera_wearer_sequence.append(i.split('/')[0])

plt.plot(interatee_loss)
plt.title('interactee')
plt.show()
plt.plot(camera_wearer_loss)
plt.title('camera_wearer')
plt.show()

save_dict = {}
save_dict['issue_interactee'] = issue_interactee
save_dict['issue_camera_wearer'] = issue_camera_wearer

sio.savemat('/home/rui/Downloads/issue_smplx2smpl.mat', save_dict)
# for i in range(len(interatee_loss)):
#     if interatee_loss[i]>0.0057 and interatee_loss[i]<0.006:
#         print(id_x[i])

1