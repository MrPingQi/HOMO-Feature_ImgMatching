%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gao Chenzhong
% Contact: gao-pingqi@qq.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function kps = Detect_Homo_Keypoint(I,DoM_pyr,scale,thresh,radius,N,G_resize,type)
%% Detector-free
if contains(lower(type), 'free')
    Im = size(I,1); In = size(I,2); step = max(sqrt(Im*In/N),radius);
    Nn = round(In/step); Ind_x = round(In/Nn*((1:Nn)-0.5));
    Nm = round(Im/step); Ind_y = round(Im/Nm*((1:Nm)-0.5));
    [XX,YY] = meshgrid(Ind_x,Ind_y);
    kps = [XX(:),YY(:)];
    value = I(sub2ind(size(I), kps(:, 2), kps(:, 1)));
    kps(value<=0,:) = [];
    kps = [kps,zeros(size(kps,1),1)]; return
end

%% HOMO feature participate
[nOctaves,nLayers] = size(DoM_pyr);
[rows,cols] = size(I);
mask = ceil(I/max(I(:)));
homoness = ones(rows,cols);
if nLayers>0
    for octave = 1:nOctaves
        for layer = 1:nLayers
            if ~isempty(DoM_pyr{octave,layer})
                homoness = homoness .* imresize(DoM_pyr{octave,layer},[rows,cols],'bilinear');
            end
        end
    end
    homoness = 1 ./ homoness.^(6/(nOctaves*max(nLayers-1,1)));  % 归一化
end

%% Phase-Congruency participate
if contains(type, 'PC')
    Ns = 4; No = 6;
    [pc_M,pc_m,~,~,~,~,~] = phasecong3(I,Ns,No,3,'mult',1.6,'sigmaOnf',0.75,'g', 3, 'k',1);
    I = pc_M + pc_m;
    I = I.*mask;
end

%% Image pat and image mask
I = I/max(I(:))*255;
imagepat = 5;
I_p = Image_Pat(I,imagepat);  % Add image pat at boundary
msk = Mask(I_p,-10);
homoness = Image_Pat(homoness,imagepat);

%% Keypoints detection
if contains(lower(type), 'harris')
    value = Harris(I_p,scale);
elseif contains(lower(type), 'shitomasi')
    value = ShiTomasi(I_p,scale);
end
border = imagepat+max(scale,radius)*2+1;
value([1:1+border,end-border:end],:) = 0;
value(:,[1:1+border,end-border:end]) = 0;
value = value .* homoness;  % HOMO feature participate

%% Nonmaximal suppression and threshold
if nargin > 2
    sze = 2*radius+1;                      % Size of mask
	mx = ordfilt2(value,sze^2,ones(sze));  % Grey-scale dilate
	value_t = (value==mx)&(value>thresh);  % Find maxima
	[rows,cols] = find(value_t);           % Find row,col coords.
    value = value(sub2ind(size(value),rows,cols));
    kps = [cols, rows, value];
else
    kps = [];
end

%% Post-processing
kps = Remove_Boundary_Points(kps,msk,max(10,G_resize^(nOctaves-2)));
if size(kps,1)<10
    kps = [];
    return
end
kps = sortrows(kps,-3);
kps = kps(1:min(N,size(kps,1)),:);
kps = kps(:,1:2)-imagepat;


function I_p = Image_Pat(I,s)
[m,n] = size(I);
I_p = zeros([m+2*s,n+2*s]);
I_p(s+1:end-s,s+1:end-s) = I;


function msk = Mask(I,th)
I = I./max(I(:))*255;
msk = double(I>th);
h = D2gauss(7,4,7,4,0);
msk = (conv2(msk,h,'same')>0.0);  % 0.8


function p = Remove_Boundary_Points(loc,msk,s)
se = strel('disk',s);
msk = ~(imdilate(~msk,se));
p = [];
for i = 1:size(loc,1)
    if msk(loc(i,2),loc(i,1)) == 1
        p = [p;loc(i,:)];
    end
end