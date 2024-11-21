%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [I1_r,I2_r,trans_1,trans_2] = ...
    Transform_temp(I1_o,I2_o,cor1,cor2,trans_form)

t_form = fitgeotrans(cor2(:,1:2),cor1(:,1:2),trans_form);
trans = t_form.T;

[M1,N1,~]=size(I1_o);
[M2,N2,~]=size(I2_o);

% C1 = [1 1 1]*solution; C2 = [N2 1 1]*solution;
% C3 = [1 M2 1]*solution; C4 = [N2 M2 1]*solution;
Cs = [[1,1];[N2,1];[1,M2];[N2,M2]];
[Cs,~] = XY_Transform(Cs,{trans,trans_form},1);
Cs = round(Cs);
C1 = Cs(1,:); C2 = Cs(2,:); C3 = Cs(3,:); C4 = Cs(4,:);

X_left  = min([C1(1),C2(1),C3(1),C4(1)]); dX = (X_left>1)*(1-X_left); X_left = max(X_left,1);
Y_up    = min([C1(2),C2(2),C3(2),C4(2)]); dY = (Y_up>1)*(1-Y_up); Y_up = max(Y_up,1);
X_right = max([C1(1),C2(1),C3(1),C4(1)]); X_right = min(N1,X_right);
Y_down  = max([C1(2),C2(2),C3(2),C4(2)]); Y_down  = min(M1,Y_down);
sX = ceil(X_right-X_left); sY = ceil(Y_down-Y_up);

trans_1 = [1,0,0; 0,1,0; dX,dY,1];
t_form = projective2d(trans_1);
I1_r = imwarp(I1_o,t_form,'OutputView',imref2d([sY,sX]));

cor1 = [cor1(:,1)+dX,cor1(:,2)+dY];
t_form = fitgeotrans(cor2(:,1:2),cor1(:,1:2),trans_form);
I2_r = imwarp(I2_o,t_form,'OutputView',imref2d([sY,sX]));
trans_2 = t_form.T;

% I1_r = I1_r.*(sum(I2_r,3)~=0);
% I2_r = I2_r.*(sum(I1_r,3)~=0);

trans_1 = {trans_1,'affine'};
trans_2 = {trans_2,trans_form};