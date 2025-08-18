%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gao Chenzhong
%   1. National Key Laboratory of Science and Technology on Space-Born Intelligent Information Processing
%   2. School of Information and Electronics, Beijing Institute of Technology (BIT), Beijing 100081, China
%   3. Beijing Institute of Technology, Zhuhai (ZHBIT), Guangdong 519088, China
% Contact: gao-pingqi@qq.com

% Pure HOMO-Feature cross-modal image matching demo.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
close all; clc; clear;
addpath('functions','func_Math','func_Geo','func_HOMO')
save_path = '.\save_image\';

%% Parameters
int_flag = 1;  % Is there any obvious intensity difference (multi-modal), yes:1, no:0
rot_flag = 1;  % Is there 0any obvious rotation difference
scl_flag = 0;  % Is there any obvious scale difference
par_flag = 0;  % Do you want parallel computing in multi-scale strategy
trans_form = 'affine'; % What spatial transform model do you need
out_form = 'Union';    % What image pair output form do you need
chg_scale = 1; % Do you want the resolution of sensed image be changed to the reference
Is_flag = 1;   % Do you want the visualization of registration results
I3_flag = 1;   % Overlap form output
I4_flag = 1;   % Mosaic form output

nOctaves = 3;    % Gaussian pyramid octave number, default: 3
nLayers  = 2;    % Gaussian pyramid  layer number, default: 4 or 2
G_resize = 1.2;  % Gaussian pyramid downsampling ratio, default: 2 or 1.2
G_sigma  = 1.6;  % Gaussian blurring standard deviation, default: 1.6
key_type = 'PC-ShiTomasi'; % What kind of feature point do you want as the keypoint
                 % keypoint choice: Harris, ShiTomasi, PC-Harris, PC-ShiTomasi, detector-free
thresh   = 0;    % Keypoints response threshold, default: 0
radius   = 1;    % Local non-maximum suppression radius, default: 2
Npoint   = 5000; % Keypoints number threshold, default: 5000
patchsize= 72;   % GPolar patchsize, default: 72 or 96
NBA = 12;        % GPolar location division, default: 12
NBO = 12;        % GPolar orientation division, default: 12
Error = 5;       % Outlier removal pixel loss, default: 5 or 3
K = 1;           % Outlier removal repetition times

%% Image input and preprocessing
if ~exist('file1','var')
    file1 = [];
end
if ~exist('file2','var')
    file2 = [];
end
[image_1,file1,DataInfo1] = Readimage(file1);
[image_2,file2,DataInfo2] = Readimage(file2);
[~,resample1] = Deal_Extreme(image_1,64,512,0);
[~,resample2] = Deal_Extreme(image_2,64,512,0);
[I1_s,I1] = Preproscessing(image_1,resample1,[]); figure,imshow(I1_s);
[I2_s,I2] = Preproscessing(image_2,resample2,[]); figure,imshow(I2_s);

%% Start
if par_flag && isempty(gcp('nocreate'))
    parpool(maxNumCompThreads);  % Start parallel computing, time needed
end
warning off
    fprintf('\n** Image matching starts, have fun\n');

%% Build HOMO feature pyramids
tic,[MOM_pyr1,DoM_pyr1] = Build_Homo_Pyramid(I1,...
    nOctaves,nLayers,G_resize,G_sigma,patchsize,NBA,int_flag);
    t(1)=toc; fprintf([' Done: HOMO-feature extraction of reference image, time cost: ',num2str(t(1)),'s\n']);
%     Display_Pyramid(MOM_pyr1,'MOM Pyramid of Reference Image',0);
%     Display_Pyramid(DoM_pyr1,'DoM Pyramid of Reference Image',0);

tic,[MOM_pyr2,DoM_pyr2] = Build_Homo_Pyramid(I2,...
    nOctaves,nLayers,G_resize,G_sigma,patchsize,NBA,int_flag);
    t(2)=toc; fprintf([' Done: HOMO-feature extraction of sensed image, time cost: ',num2str(t(2)),'s\n']);
%     Display_Pyramid(MOM_pyr2,'MOM Pyramid of Sensed Image',0);
%     Display_Pyramid(DoM_pyr2,'DoM Pyramid of Sensed Image',0);

%% Keypoints detection
ratio = sqrt((size(I1,1)*size(I1,2))/(size(I2,1)*size(I2,2)));
if ratio>=1
    r2 = radius; r1 = round(radius*ratio);
else
    r1 = radius; r2 = round(radius/ratio);
end

tic,keypoints_1 = Detect_Homo_Keypoint(I1,DoM_pyr1,6,thresh,r1,Npoint,G_resize,key_type);
    t(3)=toc; fprintf([' Done: Keypoints detection of reference image, time cost: ',num2str(t(3)),'s\n']);
%     figure,imshow(I1_s); hold on; plot(keypoints_1(:,1),keypoints_1(:,2),'r.'); 
%     title(['Reference image —— ',num2str(size(keypoints_1,1)),' keypoints']); drawnow
    clear DoMOM_pyr1

tic,keypoints_2 = Detect_Homo_Keypoint(I2,DoM_pyr2,6,thresh,r2,Npoint,G_resize,key_type);
    t(4)=toc; fprintf([' Done: Keypoints detection of sensed image, time cost: ',num2str(t(4)),'s\n']);
%     figure,imshow(I2_s); hold on; plot(keypoints_2(:,1),keypoints_2(:,2),'r.');
%     title(['Sensed image —— ',num2str(size(keypoints_2,1)),' keypoints']); drawnow
    clear DoMOM_pyr2
    
%% Keypoints description and matching (Multiscale strategy)
tic,[cor1,cor2] = Multiscale_Strategy(keypoints_1,keypoints_2,MOM_pyr1,MOM_pyr2,...
    patchsize,NBA,NBO,G_resize,Error,K,rot_flag,scl_flag,par_flag);
    t(5)=toc; fprintf([' Done: Keypoints description and matching, time cost: ',num2str(t(5)),'s\n']);
    clear MOM_pyr1 MOM_pyr2
    matchment = Show_Matches(I1_s,I2_s,cor1,cor2,0);
    cor1 = cor1/resample1; cor2 = cor2/resample2;
    
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
    t(6)=toc; fprintf([' Done: Image tranformation, time cost: ',num2str(t(6)),'s\n']);
    figure,imshow(I3); title('Overlap Form'); drawnow
    figure,imshow(I4); title('Mosaic Form'); drawnow

%% Done
T=num2str(sum(t)); fprintf(['* Done image matching, total time: ',T,'s\n']);

%% Save results
Date = datestr(now,'yyyy-mm-dd_HH-MM-SS__'); tic
Imwrite({cor1; cor2; t_form}, [save_path,Date,'0 correspond','.mat']);
if exist('matchment','var') && ~isempty(matchment) && isvalid(matchment)
    saveas(matchment, [save_path,Date,'0 Matching Result','.png']);
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
t(7)=toc; disp([' Matching results are saved at ', save_path,', time cost: ',num2str(t(7)),'s']);