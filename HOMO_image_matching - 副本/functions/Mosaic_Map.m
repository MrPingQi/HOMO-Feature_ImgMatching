%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [image1,image2,image3] = Mosaic_Map(I1,I2,d)

[M1,N1,~] = size(I1);
M11 = ceil(M1/d);
N11 = ceil(N1/d);
for i = 1:2:M11
    for j = 2:2:N11
        I1((i-1)*d+1:i*d,(j-1)*d+1:j*d,:)=0;
    end
end
for i = 2:2:M11
    for j = 1:2:N11
        I1((i-1)*d+1:i*d,(j-1)*d+1:j*d,:)=0;
    end
end
image1 = I1(1:M1,1:N1,:);

%%
[m2,n2,~] = size(I2);
m22 = ceil(m2/d);
n22 = ceil(n2/d);
for i = 1:2:m22
    for j = 1:2:n22
       I2((i-1)*d+1:i*d,(j-1)*d+1:j*d,:) = 0;
    end
end
for i = 2:2:m22
    for j = 2:2:n22
       I2((i-1)*d+1:i*d,(j-1)*d+1:j*d,:) = 0;
    end
end
image2 = I2(1:m2,1:n2,:);

%%
image3 = image1+image2;