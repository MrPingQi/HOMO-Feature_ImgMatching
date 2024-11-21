%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [I_s,I] = Preproscessing(img,resample,bands)
if ~exist('resample','var')
    resample = [];
end
if ~exist('bands','var')
    bands = [];
end

if isempty(resample)
    img = double(img);
elseif size(resample,2)==1
    img = double(imresize(img,resample));
elseif size(resample,2)==2
    img = double(imresize(img,[round(size(img,1)*resample(1)),...
                               round(size(img,2)*resample(2))]));
else
    error('Parameters error');
end

%% Data fitting and normalization
img(isnan(img)) = 0; img(isinf(img)) = 0;
if ~isempty(bands) && isnumeric(bands) && numel(bands) == 1 && bands == 0
    I = sum(img,3);
%     I = Visual(I);
    I_s = []; return
end

if size(img,3)==1
    I = img;
    I_s = I;
elseif size(img,3)==3
    if isempty(bands) || size(bands,2)~=1
        I = 0.2989*img(:,:,1) + 0.5870*img(:,:,2) + 0.1140*img(:,:,3);
%         I = ((img(:,:,1).^2.2+(1.5*img(:,:,2)).^2.2+(0.6*img(:,:,3)).^2.2)/(1+1.5^2.2+1.6^2.2)).^(1/2.2);
        I_s = img;
    else
        I = img(:,:,bands);
        I_s = I;
    end
else
    if isempty(bands) || (size(bands,2)~=3 && size(bands,2)~=1)
        I = sum(img,3);
        I_s = I;
    else
        I = sum(img,3);
        I_s = img(:,:,bands);
    end
end
% I = Visual(I);
I_s = Visual(I_s);