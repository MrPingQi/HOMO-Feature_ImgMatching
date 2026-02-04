%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gao Chenzhong
% Contact: gao-pingqi@qq.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [MOM_pyr,DoM_pyr] = Build_Homo_Pyramid(I,...
    nOctaves,nLayers,G_resize,G_sigma,patch_size,NBA,int_flag,key_type)

DoM_flag = ~contains(lower(key_type), 'free');
sig = Get_Gaussian_Scale(G_sigma,nLayers);
W = floor(patch_size/2);
r = sqrt(W^2/(2*NBA+1));

%% Build pyramids
MOM_pyr = cell(nOctaves,nLayers);
DoM_pyr = cell(nOctaves,nLayers);
for octave = 1:nOctaves
    I_t = imresize(I,1/G_resize^(octave-1),'bicubic');
    for layer = 1:nLayers
        I_t = Gaussian_Scaling(I_t, layer, sig(layer));
        [~,MOM_pyr{octave,layer}] = Major_Orientation_Map(I_t,1,r,4,int_flag);
    end
    if DoM_flag
        if layer>1
            temp = abs(MOM_pyr{octave,layer-1} - MOM_pyr{octave,layer});  % 只关心差值，不判断正负
            temp = pi/2 - abs(temp-pi/2);  % 根据[0,pi)限制，方向角度最大差值为pi/2，超过的部分按反向处理
            DoM_pyr{octave,layer-1} = temp;
        end
    end
end



function sig = Get_Gaussian_Scale(sigma,numLayers)
sig = zeros(1,numLayers);
sig(1) = sigma;  % 认为第一个图像尺度就是σ
if numLayers<2, return; end
k = 2^(1.0/(numLayers-1));
for i = 2:1:numLayers
    sig_prev = k^(i-2)*sigma;
    sig_curr = k*sig_prev;
    sig(i) = sqrt(sig_curr^2-sig_prev^2);
end



function I_t = Gaussian_Scaling(I_t,layer,sig)
if(layer>1)
    window_size = round(3*sig);
    window_size = 2*window_size+1;
    w = fspecial('gaussian',[window_size,window_size],sig);
    I_t = imfilter(I_t,w,'replicate');
end