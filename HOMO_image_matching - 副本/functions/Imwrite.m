%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function Imwrite(data, path, GeoInfo, TiffInfo)
if ~exist('data','var') || isempty(data)
    return
end

%% Create file folder if does not exist
if ispc  % case Windows(PC) platform, win32、win64
    FS = '\';
else     % case Linux、Mac
    FS = '/';
end
str = strsplit(path, FS);
Dir = path(1:end-size(str{end},2));
if (exist(Dir,'dir')==0)  % If file folder does not exist
    mkdir(Dir);
end

%% Save image to file as 'mat', 'JPG', 'TIFF', or 'GeoTIFF'
str = strsplit(path, '.');
switch str{end}
    case 'mat'
        saveMatFormat(data, path);
    case {'jpg','bmp'}
        if size(data,3)==1 || size(data,3)==3
            imwrite(data, path);
        elseif size(data,2)==3
            imwrite(data(:,:,1), path);
        else
            imwrite(data(:,:,1:3), path);
        end
    case 'png'
        alpha_channel = double((sum(data,3)~=0));
        alpha_channel(1,1) = 1;
        if size(data,3)==1 || size(data,3)==3
            imwrite(data, path, 'Alpha', alpha_channel);
        elseif size(data,3)==2
            imwrite(data(:,:,1), path, 'Alpha', alpha_channel);
        else
            imwrite(data(:,:,1:3), path, 'Alpha', alpha_channel);
        end
    case {'tif','tiff'}
        if exist('GeoInfo','var')  && ~isempty(GeoInfo) && ...
           exist('TiffInfo','var') && ~isempty(TiffInfo)  % tiff file with geoinfo
            geotiffwrite(path, data, GeoInfo,...
                'GeoKeyDirectoryTag',TiffInfo.GeoTIFFTags.GeoKeyDirectoryTag);
        else
            Img2tiff(data, path);  % tiff file without geoinfo
        end
end
end


%% .mat file saving
function saveMatFormat(data, filepath)
% 计算变量占用的内存大小，单位为bytes
info = whos('data');
variableSize = info.bytes;

% 设置阈值为1GB (1 * 1024^3 bytes)
threshold = 1 * 1024^3;

% 根据大小选择保存格式
if variableSize > threshold
    save(filepath, 'data', '-v7.3');
else
    save(filepath, 'data', '-v7');
end
end

% 关于 -v7.3 选项，这个是用来指定保存文件时使用的MATLAB文件格式。
% MATLAB提供了几种不同的文件格式，每种格式都有其特定的用途和限制：
%    -v7.3：使用HDF5格式保存数据，这种格式支持保存大于2GB的文件，
%           并且兼容一些新的数据类型，如类对象和大型数组。
%           如果你的数据非常大或者涉及复杂的数据类型，使用这个选项是必要的。
%    -v7（默认）：适用于MATLAB 7.0及以上版本的文件格式，支持压缩和Unicode字符编码。
%
% -v7.3 通常在处理非常大的数据集时表现更好，因为它支持部分加载和保存，这可以显著减少内存的使用；
%    但是，对于小型或中等大小的数据，-v7.3 格式的读写速度可能稍慢，因为它涉及更复杂的数据结构处理。
% -v7 格式在读写中等大小或小型文件时通常更快，因为它的数据结构相对简单。