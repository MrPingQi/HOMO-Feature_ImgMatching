%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing

% Core Process of the Multisource/Multimodal Image Matching
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [cor1,cor2,keypoints_1,keypoints_2] = main_image_matching(I1, I2, parameters, ...
    int_flag, rot_flag, scl_flag, par_flag, show_flag)
%% Parameters
nOctaves  = parameters.nOctaves;   % Gaussian pyramid octave number, default: 3
nLayers   = parameters.nLayers;    % Gaussian pyramid layer number, default: 4
G_resize  = parameters.G_resize;   % Gaussian pyramid downsampling ratio, default: 2
G_sigma   = parameters.G_sigma;    % Gaussian blurring standard deviation, default: 1.6
key_type  = parameters.key_type;
radius    = parameters.radius;     % Local non-maximum suppression radius, default: 2
thresh    = parameters.thresh;     % Keypoints response threshold, default: 0
Npoint    = parameters.Npoint;     % Keypoints number threshold, default: 5000
patchsize = parameters.patch_size; % GPolar patchsize, default: 72 or 96
NBA       = parameters.NBA;        % GPolar localtion division, default: 12
NBO       = parameters.NBO;        % GPolar orientation division, default: 12
Error     = parameters.Error;      % Outlier removal pixel loss, default: 5
K         = parameters.K;          % Outlier removal repetition times

%% Build HOMO feature pyramids
tic,[MOM_pyr1,DoMOM_pyr1] = Build_Homo_Pyramid(I1,...
    nOctaves,nLayers,G_resize,G_sigma,patchsize,NBA,int_flag);
    fprintf(['Done: HOMO-feature extraction of reference image, time cost: ',num2str(toc),'s\n']);
%     Display_Pyramid(MOM_pyr1,'MOM Pyramid of Reference Image',0);
%     Display_Pyramid(DoMOM_pyr1,'DoMOM Pyramid of Reference Image',0);

tic,[MOM_pyr2,DoMOM_pyr2] = Build_Homo_Pyramid(I2,...
    nOctaves,nLayers,G_resize,G_sigma,patchsize,NBA,int_flag);
    fprintf(['Done: HOMO-feature extraction of sensed image, time cost: ',num2str(toc),'s\n']);
%     Display_Pyramid(MOM_pyr2,'MOM Pyramid of Sensed Image',0);
%     Display_Pyramid(DoMOM_pyr2,'DoMOM Pyramid of Sensed Image',0);

DoMOM_pyr1 = []; DoMOM_pyr2 = [];

%% Keypoints detection
ratio = sqrt((size(I1,1)*size(I1,2))/(size(I2,1)*size(I2,2)));
if ratio>=1
    r2 = radius; r1 = round(radius*ratio);
else
    r1 = radius; r2 = round(radius/ratio);
end

tic,keypoints_1 = Detect_Homo_Keypoint(I1,DoMOM_pyr1,6,thresh,r1,Npoint,G_resize,key_type);
    fprintf(['Done: Keypoints detection of sensed image, time cost: ',num2str(toc),'s\n']);
    if show_flag
        figure,imshow(I1); hold on; plot(keypoints_1(:,1),keypoints_1(:,2),'r.'); 
        title(['Reference image —— ',num2str(size(keypoints_1,1)),' keypoints']); drawnow
    end

tic,keypoints_2 = Detect_Homo_Keypoint(I2,DoMOM_pyr2,6,thresh,r2,Npoint,G_resize,key_type);
    fprintf(['Done: Keypoints detection of sensed image, time cost: ',num2str(toc),'s\n']);
    if show_flag
        figure,imshow(I2); hold on; plot(keypoints_2(:,1),keypoints_2(:,2),'r.');
        title(['Sensed image —— ',num2str(size(keypoints_2,1)),' keypoints']); drawnow
    end

%% Keypoints matching
tic,[cor1,cor2] = Multiscale_Strategy(keypoints_1,keypoints_2,MOM_pyr1,MOM_pyr2,...
    patchsize,NBA,NBO,G_resize,Error,K,rot_flag,scl_flag,par_flag);
    fprintf(['Done: Keypoints description and matching, time cost: ',num2str(toc),'s\n']);
    if show_flag
        Show_Matches(I1,I2,cor1,cor2,0);
    end