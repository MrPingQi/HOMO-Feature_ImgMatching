%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [I,resample] = Deal_Extreme(I,A,B,trans_flag)
[M,N,~] = size(I);

% if min(M,N)>=B
%     resample = 1/min(ceil(max(M,N)/A),floor(min(M,N)/B));
% elseif max(M,N)<=A
%     resample = min(floor(A/max(M,N)),ceil(B/min(M,N)));
% else
%     resample = 1;
% end

A = A*A; B = B*B; picN = M*N;
if picN>=B
    resample = 1/min(ceil(picN/A),floor(picN/B));
elseif picN<=A
    resample = min(floor(A/picN),ceil(B/picN));
else
    resample = 1;
end
resample = sqrt(resample);

if trans_flag
    I = imresize(I,resample);
end