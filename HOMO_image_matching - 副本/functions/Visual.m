%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function I = Visual(I)
% I = abs(I);

I = I - min(I(:)); I = I / max(I(:));

% I = I - min(I(:)); I = I / mean(I(:)) / 2.5;

% I = LinearPercent(I,[0,1]);

% percent = 2;
% valid = rmoutliers(I(:),'percentiles',[percent 100-percent]);
% A = max(valid(:)); B = min(valid(:));
% I = (I-B)/(A-B);

% percent = 0.25;
% A = min(I(:)); B = max(I(:));
% thr = percent*(B-A);
% A = A+thr; B = B-thr;
% A = min(I(I>A)); B = max(I(I<B));
% I(I<A) = A; I(I>B) = B;
% I = I / mean(I(:)) / 2.5;

% dark = min(I(I>0));
% I(I>0) = I(I>0)-dark;
% valid = I(I>0);
% I = I/mean(valid)/2.5;
end



function I = LinearPercent(I,percent)
percent(percent<1) = percent(percent<1)*100;
percent = [percent(1),100-percent(2)];

[rows,cols,bands] = size(I);
I_sum = sum(I,3); I_sum(I_sum==0) = eps;
valid = rmoutliers(I_sum(:),'percentiles',percent);
A = min(valid(:)); B = max(valid(:));
maskA =  ones(rows,cols); maskA(I_sum<A) = 0;
maskB = zeros(rows,cols); maskB(I_sum>B) = 1;
valid = maskA.*(1-maskB);
ratioA = 1-A./I_sum;
ratioB = B./I_sum;
for i = 1:bands
    aaa = I(:,:,i) .* maskB .* ratioB;
    I(:,:,i) = I(:,:,i) .* ratioA .* valid + aaa;
end
I = I*bands/(B-A);
end