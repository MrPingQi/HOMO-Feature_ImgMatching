%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [cor1,cor2,inliersIndex] = Outlier_Removal(cor1,cor2,Error)
if size(cor1,1)<20
    cor1 = []; cor2 = []; inliersIndex = []; return
end

%% Fast RANSAC
[H,~,cor1_new,~] = FSC(cor1(:,1:2),cor2(:,1:2),'affine',Error);
if isempty(H) || size(cor1_new,1)<20
    cor1 = []; cor2 = []; inliersIndex = []; return
end
Y_ = H*[cor1(:,1:2)'; ones(1,size(cor1(:,1:2),1))];
Y_(1,:) = Y_(1,:)./Y_(3,:);
Y_(2,:) = Y_(2,:)./Y_(3,:);
E = sqrt(sum((Y_(1:2,:)-cor2(:,1:2)').^2));
inliersIndex = E<Error;
cor1 = cor1(inliersIndex, :);
cor2 = cor2(inliersIndex, :);