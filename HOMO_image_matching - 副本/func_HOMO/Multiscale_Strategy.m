%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [cor1,cor2] = Multiscale_Strategy(kps_1,kps_2,MOM_pyr1,MOM_pyr2,...
    patch_size,NBA,NBO,G_resize,Error,K,rot_flag,scl_flag,par_flag)
%% Multiscale procedure initialization
[nOctaves,nLayers] = size(MOM_pyr1);
matches = cell(nOctaves,nLayers,nOctaves,nLayers);
confidence = zeros(nOctaves,nLayers,nOctaves,nLayers);

keypoints_1 = cell(nOctaves,2); keypoints_1(1,:) = {kps_1(:,1:2),kps_1(:,1:2)};
keypoints_2 = cell(nOctaves,2); keypoints_2(1,:) = {kps_2(:,1:2),kps_2(:,1:2)};
img_size1 = zeros(nOctaves,2); img_size1(1,:) = size(MOM_pyr1{1,1});
img_size2 = zeros(nOctaves,2); img_size2(1,:) = size(MOM_pyr2{1,1});

for octave1=2:nOctaves
    kps_1t = round(kps_1(:,1:2)./G_resize^(octave1-1));
    [~,index1,~] = unique(kps_1t,'rows');
    keypoints_1(octave1,:) = {kps_1(index1,1:2), kps_1t(index1,1:2)};
    img_size1(octave1,:) = size(MOM_pyr1{octave1,1});
end
for octave2=2:nOctaves
    kps_2t = round(kps_2(:,1:2)./G_resize^(octave2-1));
    [~,index2,~] = unique(kps_2t,'rows');
    keypoints_2(octave2,:) = {kps_2(index2,1:2), kps_2t(index2,1:2)};
    img_size2(octave2,:) = size(MOM_pyr2{octave2,1});
end


%% Multiscale descriptor and matching procedure --- version 1 (parallel computing allowed)
if par_flag
% 描述符并行提取，索引统一化
nScales = nOctaves*nLayers;
descriptor1 = cell(nScales,1);
descriptor2 = cell(nScales,1);
parfor k=1:nScales
% for k=1:nScales
    octave = ceil(k/nLayers);
    layer = k-(octave-1)*nLayers;
    
    kps_1  = keypoints_1{octave,1}; kps_1 = [kps_1(:,1:2),(1:size(kps_1,1))'];
    kps_1t = keypoints_1{octave,2};
    kps_2  = keypoints_2{octave,1}; kps_2 = [kps_2(:,1:2),(1:size(kps_2,1))'];
    kps_2t = keypoints_2{octave,2};
    
    descriptor1{k} = GPolar_Descriptor(...
            ones(img_size1(octave,:)),...
            MOM_pyr1{octave,layer},...
            kps_1, kps_1t,...
            patch_size, NBA, NBO, rot_flag);

    descriptor2{k} = GPolar_Descriptor(...
            ones(img_size2(octave,:)),...
            MOM_pyr2{octave,layer},...
            kps_2, kps_2t,...
            patch_size, NBA, NBO, rot_flag);
end

% 描述符索引重分配
descriptors_1 = cell(nOctaves,nLayers);
descriptors_2 = cell(nOctaves,nLayers);
for k=1:nScales
    octave = ceil(k/nLayers);
    layer = k-(octave-1)*nLayers;
    
    NUM = size(descriptor1{k},1);
    descriptors_1{octave,layer} = [descriptor1{k}(:,1:5),...
        ones(NUM,1)*octave, ones(NUM,1)*layer, ...
        descriptor1{k}(:,6), (1:NUM)', descriptor1{k}(:,7:end)];
    
    NUM = size(descriptor2{k},1);
    descriptors_2{octave,layer} = [descriptor2{k}(:,1:5),...
        ones(NUM,1)*octave, ones(NUM,1)*layer, ...
        descriptor2{k}(:,6), (1:NUM)', descriptor2{k}(:,7:end)];
end
clear descriptor1 descriptor2

% 并行匹配索引初始化
if scl_flag
    nScales = nOctaves*nLayers*nOctaves*nLayers;
    k = 1:nScales;
    Octave2 = ceil(k/(nOctaves*nLayers*nLayers));
    Octave1 = mod(ceil(k/(nLayers*nLayers))-1,nOctaves)+1;
else
    nScales = nOctaves*nLayers*nLayers;
    k = 1:nScales;
    Octave2 = ceil(k/(nLayers*nLayers));
    Octave1 = Octave2;
end
Layer2 = mod(ceil(k/nLayers)-1,nLayers)+1;
Layer1 = mod(k-1,nLayers)+1;

% 并行匹配，索引统一化
tmatches = cell(nScales,1);
tconfidence = zeros(nScales,1);
parfor k=1:nScales
% for k=1:nScales
    [tmatches{k},tconfidence(k)] = Match_Keypoint(...
        descriptors_1{Octave1(k),Layer1(k)}, ...
        descriptors_2{Octave2(k),Layer2(k)}, Error, K);
end
clear descriptors_1 descriptors_2

% 匹配索引重分配
for k=1:nScales
    matches{Octave1(k),Layer1(k),Octave2(k),Layer2(k)} = tmatches{k};
    confidence(Octave1(k),Layer1(k),Octave2(k),Layer2(k)) = tconfidence(k);
end
clear tmatches tconfidence
end


%% Multiscale descriptor and matching procedure --- version 2 (new iterative optimization)
if ~par_flag
for octave2=1:nOctaves
    kps_2  = keypoints_2{octave2,1};
    kps_2t = keypoints_2{octave2,2};

    for octave1=1:nOctaves
        if ~scl_flag
            octave1 = octave2;
        end
        kps_1  = keypoints_1{octave1,1};
        kps_1t = keypoints_1{octave1,2};

        for layer2=1:nLayers
            kps_2 = [kps_2(:,1:2),(1:size(kps_2,1))'];
            idx = [];

            descriptor2 = GPolar_Descriptor(...
                MOM_pyr2{octave2,layer2}*0+1, MOM_pyr2{octave2,layer2},...
                kps_2, kps_2t,...
                patch_size, NBA, NBO, rot_flag);

            NUM = size(descriptor2,1);
            descriptor2 = [descriptor2(:,1:5),...
                ones(NUM,1)*octave2,...
                ones(NUM,1)*layer2,...
                descriptor2(:,6),...
                (1:NUM)',...
                descriptor2(:,7:end)];

            descriptors_2{octave2,layer2} = descriptor2;

            for layer1=1:nLayers
                kps_1 = [kps_1(:,1:2),(1:size(kps_1,1))'];

                descriptor1 = GPolar_Descriptor(...
                    MOM_pyr1{octave1,layer1}*0+1, MOM_pyr1{octave1,layer1},...
                    kps_1, kps_1t,...
                    patch_size, NBA, NBO, rot_flag);

                NUM = size(descriptor1,1);
                descriptor1 = [descriptor1(:,1:5),...
                    ones(NUM,1)*octave1,...
                    ones(NUM,1)*layer1,...
                    descriptor1(:,6),...
                    (1:NUM)',...
                    descriptor1(:,7:end)];

                descriptors_1{octave1,layer1} = descriptor1;

                [match, confidence(octave1,layer1,octave2,layer2)] = ...
                    Match_Keypoint(descriptor1,descriptor2,Error,K);
                matches{octave1,layer1,octave2,layer2} = match;

%                 if size(match,1)>50
%                     kps_1(match(:,8),:) = [];
%                     kps_1t(match(:,8),:) = [];
%                     descriptor2(match(:,18),:) = [];
%                     descriptor2(:,9) = 1:size(descriptor2,1);
%                     idx = [idx; match(:,17)];
%                 end
            end
%             kps_2(idx,:) = [];
%             kps_2t(idx,:) = [];
        end
        if ~scl_flag
            break;
        end
    end
end
end


%% Optimizing
Confidence = zeros(nOctaves,nOctaves);
Matches = cell(nOctaves,nOctaves);
for octave1=1:nOctaves
    for octave2=1:nOctaves
        matches_t = [];
        for layer1=1:nLayers
            for layer2=1:nLayers
                matches_t = [matches_t; matches{octave1,layer1,octave2,layer2}];
%                 if size(matches{octave1,layer1,octave2,layer2},1)>3
%                     Show_Matches(MOM_pyr1{octave1,layer1},...
%                                  MOM_pyr2{octave2,layer2},...
%                                  matches{octave1,layer1,octave2,layer2}(:,3:4),...
%                                  matches{octave1,layer1,octave2,layer2}(:,12:13),0);
%                     title(['octave1 = ',num2str(octave1),...
%                         '  layer1 = ',num2str(layer1),...
%                         '  octave2 = ',num2str(octave2),...
%                         '  layer2 = ',num2str(layer2)]); drawnow
%                 end
            end
        end
        if size(matches_t,1)>20
            [~,index1,~] = unique(matches_t(:,1:2),'rows');
            matches_t = matches_t(index1,:);
            [~,index2,~] = unique(matches_t(:,10:11),'rows');
            matches_t = matches_t(index2,:);
        end
        if size(matches_t,1)>20
            Matches{octave1,octave2} = matches_t;
            Confidence(octave1,octave2) = size(matches_t,1);
        end
    end
end
[max_O1,max_O2] = find(Confidence==max(max(Confidence)));

MMatches = [];
for i = 1-min(max_O1,max_O2):min(nOctaves-max_O1,nOctaves-max_O2)
    matches_t = Matches{max_O1+i,max_O2+i};
    if size(matches_t,1)>3
        MMatches = [MMatches; matches_t];
    end
end
[~,index1,~] = unique(MMatches(:,1:2),'rows');
MMatches = MMatches(index1,:);
[~,index2,~] = unique(MMatches(:,10:11),'rows');
MMatches = MMatches(index2,:);


%% One last outlier removal
NCMs = zeros(K,1); indexPairs = cell(K,1);
for k = 1:K
    [~,~,indexPairs{k}] = Outlier_Removal(MMatches(:,1:9),MMatches(:,10:end),Error);
    NCMs(k) = sum(indexPairs{k});
end
[~,maxIdx] = max(NCMs);
indexPairs = indexPairs{maxIdx};
MMatches = MMatches(indexPairs,:);
cor1 = MMatches(:,1:9); cor2 = MMatches(:,10:end);