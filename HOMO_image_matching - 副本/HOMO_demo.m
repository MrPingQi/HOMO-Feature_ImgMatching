%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing

% HOMO-feature cross-modal image matching algorithm
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clc;
clear;
addpath('functions','func_Math','func_Geo','func_HOMO')

%% Parameters
rot_flag = 1;  % Is there any obvious rotation difference, yes:1, no:0
scl_flag = 0;  % Is there any obvious scale difference
par_flag = 0;  % Do you want parallel computing in multi-scale strategy
               % (This may cause problem depending on computer system,
               %  for example, the RAM should better be at least 64GB)
trans_form = 'affine';  % What spatial transformation model do you need at the end:
                        % 'similarity', 'affine', 'projective'
out_form = 'Union';  % What output form of the image pair do you need at the end: 
                     % 'Reference', 'Union', 'Inter'
chg_scale = 1;% Do you want the resolution of sensed image be changed to the reference
Is_flag = 1;  % Do you want the visualization of registration results
I3_flag = 1;  % Do you want the Overlap form of registration results
I4_flag = 1;  % Do you want the Mosaic form of registration results
save_path = '.\save_image\';

nOctaves = 3;    % Gaussian pyramid octave number, default: 3
nLayers  = 4;    % Gaussian pyramid  layer number, default: 4
G_resize = 1.2;  % Gaussian pyramid downsampling ratio, default: 2
G_sigma  = 1.6;  % Gaussian blurring standard deviation, default: 1.6
key_type = 'PC-ShiTomasi'; % What kind of feature point do you want as the keypoint:
                           % 'Harris', 'ShiTomasi', 'PC-Harris', 'PC-ShiTomasi', 'Detector-free'
                           % 'Detector-free' means 
radius   = 1;    % Local non-maximum suppression radius, default: 1 or 2
thresh   = 0;    % Keypoints response threshold, default: 0
Npoint   = 5000; % Keypoints number threshold, default: 5000
patchsize= 72;   % GPolar patchsize, default: 72 or 96
NBA      = 12;   % GPolar location division, default: 12
NBO      = 12;   % GPolar orientation division, default: 12
PixLoss  = 5;    % Outlier removal pixel loss threshold, default: 3 or 5

%% Image input and preprocessing
if ~exist('file1','var')
    file1 = [];
end
if ~exist('file2','var')
    file2 = [];
end
[image_1,file1,DataInfo1] = Readimage(file1);
[image_2,file2,DataInfo2] = Readimage(file2);
[I1_s,I1] = Preproscessing(image_1,1,[]); figure,imshow(I1_s);
[I2_s,I2] = Preproscessing(image_2,1,[]); figure,imshow(I2_s);

%%
if par_flag && isempty(gcp('nocreate'))
    parpool(maxNumCompThreads);  % Start parallel computing, time needed
end
warning off
    fprintf('\n** Image matching starts, have fun\n\n');



%% Build HOMO feature pyramids ---------- MOM and DoM
tic,[MOM_pyr1,DoMOM_pyr1] = Build_Homo_Pyramid(I1,...
    nOctaves,nLayers,G_resize,G_sigma,patchsize,NBA,1);
    t(1)=toc; fprintf(['Done: HOMO-feature extraction of reference image, time cost: ',num2str(t(1)),'s\n']);
%     Display_Pyramid(MOM_pyr1,'MOM Pyramid of Reference Image',0);
%     Display_Pyramid(DoMOM_pyr1,'DoMOM Pyramid of Reference Image',0);

tic,[MOM_pyr2,DoMOM_pyr2] = Build_Homo_Pyramid(I2,...
    nOctaves,nLayers,G_resize,G_sigma,patchsize,NBA,1);
    t(2)=toc; fprintf(['Done: HOMO-feature extraction of sensed image, time cost: ',num2str(t(2)),'s\n']);
%     Display_Pyramid(MOM_pyr2,'MOM Pyramid of Sensed Image',0);
%     Display_Pyramid(DoMOM_pyr2,'DoMOM Pyramid of Sensed Image',0);

%% Keypoints detection ---------- DoM enhancing
ratio = sqrt((size(I1,1)*size(I1,2))/(size(I2,1)*size(I2,2)));
if ratio>=1
    r2 = radius; r1 = round(radius*ratio);
else
    r1 = radius; r2 = round(radius/ratio);
end

tic,keypoints_1 = Detect_Homo_Keypoint(I1,DoMOM_pyr1,6,thresh,r1,Npoint,G_resize,key_type);
    t(3)=toc; fprintf(['Done: Keypoints detection of sensed image, time cost: ',num2str(t(3)),'s\n']);
    figure,imshow(I1_s); hold on; plot(keypoints_1(:,1),keypoints_1(:,2),'r.'); 
    title(['Reference image —— ',num2str(size(keypoints_1,1)),' keypoints']); drawnow
    clear DoMOM_pyr1

tic,keypoints_2 = Detect_Homo_Keypoint(I2,DoMOM_pyr2,6,thresh,r2,Npoint,G_resize,key_type);
    t(4)=toc; fprintf(['Done: Keypoints detection of sensed image, time cost: ',num2str(t(4)),'s\n']);
    figure,imshow(I2_s); hold on; plot(keypoints_2(:,1),keypoints_2(:,2),'r.');
    title(['Sensed image —— ',num2str(size(keypoints_2,1)),' keypoints']); drawnow
    clear DoMOM_pyr2
    
%% Keypoints description and matching (Multiscale strategy) ---------- GPolar and MsS
tic,[cor1,cor2] = Multiscale_Strategy(keypoints_1,keypoints_2,MOM_pyr1,MOM_pyr2,...
    patchsize,NBA,NBO,G_resize,PixLoss,1,rot_flag,scl_flag,par_flag);
    t(5)=toc; fprintf(['Done: Keypoints description and matching, time cost: ',num2str(t(5)),'s\n']);
    clear MOM_pyr1 MOM_pyr2
    matchment = Show_Matches(I1_s,I2_s,cor1,cor2,0);



%% Image transformation (Geography enable)
tic,[I1_r,I2_r,I1_rs,I2_rs,I3,I4,t_form,pos] = Transformation(image_1,image_2,...
    cor1,cor2,trans_form,out_form,chg_scale,Is_flag,I3_flag,I4_flag);
    if ~isempty(DataInfo1) && ~isempty(DataInfo1.SpatialRef)
        pos(3:4) = pos(3:4)+1;
        GeoInfo2 = Create_GeoInfo(I2_r,pos,DataInfo1);
        if strcmpi(out_form,'Geo') || strcmpi(out_form,'Reference')
            GeoInfo1 = DataInfo1.SpatialRef;
        else
            [rows,cols,~] = size(I1_r);
            GeoInfo1 = GeoInfo2; GeoInfo1.RasterSize = [rows,cols];
        end
    else
        GeoInfo1 = []; GeoInfo2 = [];
    end
    t(6)=toc; fprintf(['Done: Image tranformation, time cost: ',num2str(t(6)),'s\n\n']);
    figure,imshow(I3); title('Overlap Form'); drawnow
    figure,imshow(I4); title('Mosaic Form'); drawnow

%%
T=num2str(sum(t)); fprintf([' Done image registration, total time: ',T,'s\n\n']);
    
%% Save results
Date = datestr(now,'yyyy-mm-dd_HH-MM-SS__'); tic
if exist('matchment','var') && ~isempty(matchment) && isvalid(matchment)
    saveas(matchment, [save_path,Date,'0 Matching Result','.jpg']);
end
if strcmpi(out_form,'reference')
    Imwrite(image_1, [save_path,Date,'1 Reference Image','.tif'],GeoInfo1,DataInfo1);
    if Is_flag
        [I1_s,~] = Preproscessing(image_1,1,[]); 
        Imwrite(I1_s, [save_path,Date,'3 Reference Image Show','.png']);
    end
else
    Imwrite(I1_r , [save_path,Date,'1 Reference Image','.tif'],GeoInfo1,DataInfo1);
    Imwrite(I1_rs, [save_path,Date,'3 Reference Image Show','.png']);
end
Imwrite(I2_r , [save_path,Date,'2 Registered Image','.tif'],GeoInfo2,DataInfo1);
Imwrite(I2_rs, [save_path,Date,'4 Registered Image Show','.png']);
Imwrite(I3   , [save_path,Date,'5 Overlap of results','.png']);
Imwrite(I4   , [save_path,Date,'6 Mosaic of results','.png']);
T=num2str(toc); fprintf([' Registration results are saved in the save_image folder, time: ',T,'s\n']);