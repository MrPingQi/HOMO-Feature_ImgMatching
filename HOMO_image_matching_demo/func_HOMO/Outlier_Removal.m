%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Author: Gao Chenzhong
% Contact: gao-pingqi@qq.com
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [cor1,cor2,inliersIndex] = Outlier_Removal(cor1,cor2,Error,iter,trans_form,ref)
if size(cor1,1)<20, cor1 = []; cor2 = []; inliersIndex = []; return; end
if ~exist('ref','var') || isempty(ref), ref = {[],[]}; end
% if ~exist('iter','var'), iter = 10000; end
if ~exist('iter','var'), iter = 800; end

%% RANSAC
[H,~,cor1_new,~] = FSC(cor1(:,1:2),cor2(:,1:2),trans_form,Error,iter,ref);
if isempty(H) || size(cor1_new,1)<20, cor1 = []; cor2 = []; inliersIndex = []; return; end
if ~isempty(ref), cor1 = [cor1; ref{1}]; cor2 = [cor2; ref{2}]; end
Y_ = H*[cor1(:,1:2)'; ones(1,size(cor1(:,1:2),1))];
Y_(1,:) = Y_(1,:)./Y_(3,:);
Y_(2,:) = Y_(2,:)./Y_(3,:);
E = sqrt(sum((Y_(1:2,:)-cor2(:,1:2)').^2));
inliersIndex = E<Error;
cor1 = cor1(inliersIndex,:);
cor2 = cor2(inliersIndex,:);