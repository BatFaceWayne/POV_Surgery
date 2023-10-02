% file:///home/rui/Downloads/test_calib/src/debug_hand.mat

% clc
% hand_correct = pcread('/Users/wangrui/Downloads/00000_Hand.ply');
% hand_wrong = pcread('/Users/wangrui/Downloads/000000_NewHand.ply');
% 
% correct_v = hand_correct.Location;
% wrong_v = hand_wrong.Location;
% forward_transfer = knnsearch(correct_v,wrong_v);
% backward_transfer = knnsearch(wrong_v,correct_v);
% save MANO_version.mat forward_transfer backward_transfer\c
clc
close all
clear

base_transl_dict = '/media/rui/File_exchan/Technopark_Recordings/capture_23_01_23/isabel_diskplacer/output/smplx/texture_rotate/transl_dict_raw.mat';
base_rot = '/media/rui/File_exchan/Technopark_Recordings/capture_23_01_23/isabel_diskplacer/output/smplx/texture_rotate/rot_dict_raw.mat';

load(base_transl_dict)
load(base_rot)
out_base_transl = strcat(base_transl_dict(1:end-8), '.mat');
out_base_rot = strcat(base_rot(1:end-8), '.mat');
% rot_dict = wrapTo360(rot_dict);
rot_dict_copy = rot_dict;
% rot_dict_copy(rot_dict_copy<0,3) = rot_dict_copy(rot_dict_copy<0) + 360;

% rot_dict_copy(rot_dict_copy(:,1)>90,3) = rot_dict_copy(rot_dict_copy(:,1)>90,3) - 360;
figure(1),

subplot(2,1,1)
plot(rot_dict_copy,'LineWidth',1)
legend('x','y','z')
title('Camera Rotation Before Stabalization')
ylabel('Degree [^\circ]')
xlabel('Frames')
set(gca,'fontname','serif') 

figure(2),
% figure(1),
subplot(2,1,1)
plot(transl_dict,'LineWidth',1)
legend('x','y','z')
title('Camera Translation Before Stabalization')
ylabel('Location [m]')
xlabel('Frames')
set(gca,'fontname','serif') 
rot_dict = filloutliers(rot_dict_copy,'linear');
rot_dict = smoothdata(rot_dict,'movmean',16);


figure(1),
subplot(2,1,2),plot(rot_dict,'LineWidth',1)
ylim([-200,200])
title('Camera Rotation After Stabalization')
ylabel('Degree [^\circ]')
xlabel('Frames')
legend('x','y','z')
set(gca,'fontname','serif') 

transl_dict = filloutliers(transl_dict,'linear');
transl_dict = smoothdata(transl_dict,'movmean',16);
set(gcf,'WindowStyle','Normal');
set(gcf,'Position',[1 1 600 400]);
% eval(['print /Users/wangrui/Downloads/filter_camera_' num2str(1) '.eps -depsc2 -r600']);
figure(2),
subplot(2,1,2),plot(transl_dict,'LineWidth',1)
legend('x','y','z')
title('Camera Translation After Stabalization')
ylabel('Location [m]')
xlabel('Frames')
set(gca,'fontname','serif') 

set(gcf,'WindowStyle','Normal');
set(gcf,'Position',[1 1 600 400]);

% eval(['print /Users/wangrui/Downloads/filter_camera_' num2str(2) '.eps -depsc2 -r600']);

% fig2plotly(gcf);
save(out_base_transl, 'transl_dict')
save(out_base_rot, 'rot_dict')

function norm_rot()
end