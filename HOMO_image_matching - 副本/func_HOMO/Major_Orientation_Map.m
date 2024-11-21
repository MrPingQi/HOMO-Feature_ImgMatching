%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [magnitude,orientation] = Major_Orientation_Map(I,R1,R2,s,int_flag)
% MOM: Major Orientation Map
sigma = 0.5;
w = 2*round(3*sigma)+1;
w = fspecial('gaussian',[w,w],sigma);
I = imfilter(I,w,'replicate');

%% LogGabor feature maps
Ns = 4; No = 6;
EO = LogGabor(I,Ns,No,3,1.6,0.75);  % minWaveLength = 3, mult = 1.6, sigmaOnf = 0.75
[M,N] = size(I); clear I
Gx = zeros(M,N); Gy = zeros(M,N);
angle = pi*(0:No-1)/No;
angle_cos = cos(angle);
angle_sin = sin(angle);
for j=1:No
    for i=1:Ns
        direct = sign(sign(imag(EO{i,j}))+0.5);  % [-1,0,1] ——> [-0.5,0.5,1.5] ——> [-1,1]
        Gx = Gx - (imag(EO{i,j}) * 1i...
                +  real(EO{i,j}) .* direct) * angle_cos(j)*(Ns-i+1);  % 注意图像与matlab矩阵坐标关系，y轴是反的
        Gy = Gy + (imag(EO{i,j}) * 1i...
                +  real(EO{i,j}) .* direct) * angle_sin(j)*(Ns-i+1);
    end
end

%% Averaging Squared LogGabor (ASLG) feature maps
W = floor(R2);  % 窗半径
dx = -W : W;  % 邻域x坐标
dy = -W : W;  % 邻域y坐标
[dx,dy] = meshgrid(dx,dy);
Wcircle = ((dx.^2 + dy.^2) < (W+1)^2)*1.0;  % 圆形窗
Patchsize = 2*W+1;

if s==1
    h = fspecial('gaussian',[Patchsize,Patchsize], R1/6);
else
    step = (R2-R1)/(s-1);
    h = zeros(Patchsize,Patchsize);
    for i=0:s-1
        sigma = (R1+step*i)/6;
        h = h + fspecial('gaussian',[Patchsize,Patchsize], sigma);
    end
end
h = h.*Wcircle;

Gx1 = imag(Gx); Gx2 = real(Gx); clear Gx
Gy1 = imag(Gy); Gy2 = real(Gy); clear Gy

Gxx1 = imfilter(Gx1.*Gx1, h, 'replicate');
Gyy1 = imfilter(Gy1.*Gy1, h, 'replicate');
Gxy1 = imfilter(Gx1.*Gy1, h, 'replicate'); clear Gx1 Gy1
Gsx1 = Gxx1-Gyy1;                          clear Gxx1 Gyy1
Gsy1 = 2*Gxy1;                             clear Gxy1

Gxx2 = imfilter(Gx2.*Gx2, h, 'replicate');
Gyy2 = imfilter(Gy2.*Gy2, h, 'replicate');
Gxy2 = imfilter(Gx2.*Gy2, h, 'replicate'); clear Gx2 Gy2
Gsx2 = Gxx2-Gyy2;                          clear Gxx2 Gyy2
Gsy2 = 2*Gxy2;                             clear Gxy2

%% Odd/Even-ASLG feature maps
orientation1 = atan2(Gsy1,Gsx1)/2 + pi/2;  % 取值范围：[-pi,pi] ——> [0,pi]
magnitude1 = sqrt(sqrt(Gsx1.^2+Gsy1.^2)); clear Gsx1 Gsy1
orientation2 = atan2(Gsy2,Gsx2)/2 + pi/2;  % 取值范围：[-pi,pi] ——> [0,pi]
magnitude2 = sqrt(sqrt(Gsx2.^2+Gsy2.^2)); clear Gsx2 Gsy2

% orientation = orientation1;
% magnitude = magnitude1;
% orientation = orientation2;
% magnitude = magnitude2;

%% Odd/Even coupling to MOM feature
idx = ceil((sign(magnitude1-magnitude2)+0.1)/2);
orientation = idx.*orientation1 + (1-idx).*orientation2;  %%% 耦合！
orientation = mod(orientation,pi);  % 取值范围：[0,pi] ——> [0,pi)
if int_flag
    magnitude = ones(M,N);
else
    magnitude = idx.*magnitude1 + (1-idx).*magnitude2;
end