%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function matchment = Show_Matches(I1,I2,cor1,cor2,option)
if isempty(cor1) || isempty(cor2)
    matchment = []; return
end
if option==0  % 自动确定横/竖版
    option = 1+((size(I1,1)+size(I2,1))*2.5 < (size(I1,2)+size(I2,2)));
end

[I3,cor1,cor2] = Append_Images(I1,I2,cor1,cor2,option,'middle');  % 'top','middle','bottom'

% Figure 尺寸贴合绘图
[M,N,~] = size(I3);
matchment = figure; 
pos = get(matchment,'Position');
if M<N
    pos(3) = ceil(pos(4)*N/M);
else
    pos(4) = ceil(pos(3)*M/N);
end
pos(4) = pos(4)+45;

% Figure 屏幕居中
screenSize = get(0, 'ScreenSize');
newFigPos = [(screenSize(3)-pos(3))/2,(screenSize(4)-pos(4))/2,pos(3),pos(4)];
set(matchment, 'Position', newFigPos);

% Points 颜色
% pColor1 = determinePointColor(I1, 'g.', 'r.');
% pColor2 = determinePointColor(I2, 'g+', 'r+');

imshow(I3,'Border','tight','Initialmagnification','fit'); hold on
for i=1:size(cor1,1)
    line([cor1(i,1),cor2(i,1)],[cor1(i,2),cor2(i,2)], 'Color', 'y');
%     plot(cor1(i,1),cor1(i,2),pColor1)
%     plot(cor2(i,1),cor2(i,2),pColor2)
end
% for i=1:size(cor1,1)
%     text(cor1(i,1),cor1(i,2),num2str(i),'color','y');
%     text(cor2(i,1),cor2(i,2),num2str(i),'color','y');
% end
hold off

if option==1
    title(['Left is reference image --- ',num2str(size(cor1,1)),' matching pairs --- Right is sensed image']);
elseif option==2
    title(['Top is reference image --- ',num2str(size(cor1,1)),' matching pairs --- Bottom is sensed image']);
end
drawnow


function [img,cor1,cor2] = Append_Images(I1,I2,cor1,cor2,option,pos)
B1 = size(I1,3); B2 = size(I2,3);
if B1~=1 && B1~=3
    I1 = sum(I1,3);
end
if B2~=1 && B2~=3
    I2 = sum(I2,3);
end
I1 = Visual(I1); I2 = Visual(I2);

[M1,N1,B1] = size(I1); [M2,N2,B2] = size(I2);
if B1==1 && B2==3
    I1 = repmat(I1, [1,1,3]);
elseif B1==3 && B2==1
    I2 = repmat(I2, [1,1,3]);
end

switch pos
    case 'top'
        dM = 0; dN = 0;
    case 'middle'
        dM = floor(abs(M1-M2)/2); dN = floor(abs(N1-N2)/2);
    case 'bottom'
        dM = abs(M1-M2); dN = abs(N1-N2);
end

if option==1
    if (M1 < M2)
        I1(M2,1,:) = 0;
        I1(dM+1:dM+M1,:,:) = I1(1:M1,:,:);
        I1(1:dM,:,:) = 0;
        cor1(:,2) = cor1(:,2)+dM;
    else
        I2(M1,1,:) = 0;
        I2(dM+1:dM+M2,:,:) = I2(1:M2,:,:);
        I2(1:dM,:,:) = 0;
        cor2(:,2) = cor2(:,2)+dM;
    end
    img = [I1,I2];
    cor2(:,1) = cor2(:,1)+size(I1,2);
elseif option==2
    if (N1 < N2)
        I1(1,N2,:) = 0;
        I1(:,dN+1:dN+N1,:) = I1(:,1:N1,:);
        I1(:,1:dN,:) = 0;
        cor1(:,1) = cor1(:,1)+dN;
    else
        I2(1,N1,:) = 0;
        I2(:,dN+1:dN+N2,:) = I2(:,1:N2,:);
        I2(:,1:dN,:) = 0;
        cor2(:,1) = cor2(:,1)+dN;
    end
    img = [I1;I2];
    cor2(:,2) = cor2(:,2)+size(I1,1);
end


function pColor = determinePointColor(image, defaultColor, alternateColor)
% 判断图像颜色是否偏绿
if size(image, 3) == 3
    meanColor = mean(mean(image));
    meanG = meanColor(2) * 1.5;
    meanRB = meanColor(1) + meanColor(3);
    if meanG > meanRB
        pColor = alternateColor;
    else
        pColor = defaultColor;
    end
else
    pColor = defaultColor;
end