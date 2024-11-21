%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function descriptor = GPolar_Descriptor(MOM_m, MOM_o, kps_o, kps, ...
    patch_size, NBA, NBO, rot_flag)

W = floor(patch_size/2);  % 窗半径
X = -W : W;  % 邻域x坐标
Y = -W : W;  % 邻域y坐标
[XX,YY] = meshgrid(X,Y);
Wcircle = ((XX.^2 + YY.^2) < (W+1)^2)*1.0;  % 圆形窗

% Rho area divide
rr1 = W^2/(2*NBA+1); rr2 = rr1*(NBA+1);
Rho = XX.^2+YY.^2;
Rho(Rho<=rr1) = 1;
Rho(Rho>rr1 & Rho<=rr2) = 2;
Rho(Rho>rr2) = 3;
Rho = Rho .* Wcircle - 1;  % 三个区域：0、1、2

% Theta area divide
if ~rot_flag
    Theta = atan2(YY,XX) + pi;  % 取值范围：[0,2pi]
    Theta = mod(floor(Theta*NBA/pi/2),NBA)+1;  % 取值范围：[1,NBA]
end

% Deeper feature parameters
w_ratio1 = sqrt(rr1/rr2);
w_ratio2 = sqrt(rr1)/W;
c_idx = [2:NBA,1];

%% 基准方向 Base direction
% kps_o记录特征点原始坐标，kps记录特征点在当前尺度下坐标，需根据代码索引正确更改顺序
if rot_flag
    kps_new = Base_Direction([kps,kps_o],MOM_m,MOM_o,W,NBO);
    kps_o = kps_new(:,3:4); kps = kps_new(:,[1,2,6,5]);
else
    kps(:,3) = 0; kps(:,4) = kps_o(:,3); kps_o = kps_o(:,1:2);
end

%% 描述符 GPolar
descriptor = zeros(size(kps,1), (2+3*NBA)*NBO);  % descriptor (size:(1+3×s+1)×o)
feat_all = zeros(NBO,1);
for k = 1:size(kps,1)
    x = kps(k,1); x1 = max(1,x-W); x2 = min(x+W,size(MOM_o,2));
    y = kps(k,2); y1 = max(1,y-W); y2 = min(y+W,size(MOM_o,1));
    angle_bin = zeros(patch_size+1,patch_size+1);  % 0 表示不统计
    
    % Rotation invariance
    if rot_flag
        orient = kps(k,3);
        angle_bin(W+y1-y+1:W+y2-y+1, W+x1-x+1:W+x2-x+1) = ...
            floor(mod(MOM_o(y1:y2, x1:x2)-orient,pi) * NBO/pi) + 1;  % 取值范围：[1,NBO] & 0
        sin_p = sin(orient);
        cos_p = cos(orient);
        Xr =  cos_p * XX + sin_p * YY;
        Yr = -sin_p * XX + cos_p * YY;
        Theta = atan2(Yr,Xr) + pi;  % 取值范围：[0,2pi]
        Theta = mod(floor(Theta*NBA/pi/2),NBA)+1;  % 取值范围：[1,NBA]
    else
        angle_bin(W+y1-y+1:W+y2-y+1, W+x1-x+1:W+x2-x+1) = ...
            floor(MOM_o(y1:y2, x1:x2) * NBO/pi) + 1;  % 取值范围：[1,NBO] & 0
    end
    
    % Feature histogram
    feat_center = zeros(NBO,1);
    feat_outer = zeros(NBO,NBA,3);
    weight = zeros(patch_size+1,patch_size+1);
    weight(W+y1-y+1:W+y2-y+1, W+x1-x+1:W+x2-x+1) = MOM_m(y1:y2, x1:x2);
    for xx = 1:patch_size+1
        for yy = 1:patch_size+1
            Rho_t = Rho(yy,xx);
            Theta_t = Theta(yy,xx);
            angle_t = angle_bin(yy,xx);
            if angle_t<1 || Rho_t<0 || Rho_t>2
                continue
            elseif Rho_t==0
                feat_center(angle_t) = feat_center(angle_t) + weight(yy,xx);
            else
                feat_outer(angle_t,Theta_t,Rho_t) = ...
                    feat_outer(angle_t,Theta_t,Rho_t) + weight(yy,xx);
            end
        end
    end
    
    % Blurring
%     feat_outer(:,:,1:2) = imfilter(feat_outer(:,:,1:2), H, 'circular');
    
%     Inversion dealing
    if rot_flag
        des_H1 = feat_outer(:,1:NBA/2,1:2);
        des_H2 = feat_outer(:,NBA/2+1:NBA,1:2);
        V = var(des_H1,0,'all') - var(des_H2,0,'all');
        if V>0
            feat_outer(:,:,1:2) = cat(2,des_H2,des_H1);
        end
    end
    
    % Deeper feature
    feat_outer(:,:,3) = (squeeze(...
        (feat_outer(:,:,1) + feat_outer(:,c_idx,1)) * w_ratio1 + ...
        (feat_outer(:,:,2) + feat_outer(:,c_idx,2)) * w_ratio2 ) / 2 + ...
        repmat(feat_center, 1, NBA)) /3 /2;
    feat_all = squeeze(sum(feat_outer(:,:,3),2)/NBA);
    
    % Descriptor vectors
    descriptor(k,:) = [feat_center; feat_outer(:); feat_all];
end
descriptor = [kps_o, kps, descriptor];