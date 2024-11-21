%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function image_pyramid = Display_Pyramid(pyramid,str,save_or_not)
[nOctaves,nLayers] = size(pyramid);
size_image = zeros(nOctaves,2);
for i = 1:nOctaves
    for j = 1:nLayers
        if isempty(pyramid{i,j})
            continue
        end
        size_image(i,1:2) = size(pyramid{i,j});
        break
    end
end

%% display Image Pyramid
% ROW_size = sum(size_image(:,1));
% COL_size = size_image(1,1)*nLayers;
image_pyramid = zeros(1,1); 
accumulate_ROW = 0;
for i = 1:nOctaves
    accumulate_ROW = accumulate_ROW+size_image(i,1);
    accumulate_COL = 0;
    for j = 1:nLayers
        if isempty(pyramid{i,j})
            continue
        end
        accumulate_COL = accumulate_COL+size_image(i,2);
        image_pyramid(accumulate_ROW-size_image(i,1)+1:accumulate_ROW,...
           accumulate_COL-size_image(i,2)+1:accumulate_COL) = mat2gray(pyramid{i,j});
    end
end
figure, imshow(image_pyramid,[]), title(str); drawnow
if save_or_not==1
    str = ['.\save_image\',str,'.jpg'];
    imwrite(image_pyramid,str,'jpg');
end