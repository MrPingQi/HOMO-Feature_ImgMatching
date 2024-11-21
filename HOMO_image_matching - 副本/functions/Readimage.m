%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% CVPR 2025 Submission Paper ID #16689
% This code is only for the purpose of reviewing
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function [image, file, info] = Readimage(file, roi)
if ispc  % case Windows(PC) platform, win32、win64
    FS = '\';
else     % case Linux、Mac
    FS = '/';
end
exe_path = [pwd,FS];
UI_pathtemp = [exe_path,'UIpathtemp.txt'];
% UI_pathtemp = 'UIpathtemp.txt';

%% Check input nargin 
file_flag = 0;
if nargin>0 && exist('file','var') && ~isempty(file)
    path_flag = isfolder(file);
    if path_flag
        Path = file;
    else
        if exist(file,'file')
            file_flag = 1;
        else
            path_flag = 1; Path = [];
        end
    end
else
    path_flag = 1; file_flag = 0;
    if (exist(UI_pathtemp,'file')==0) % If file folder does not exist
        fid = fopen(UI_pathtemp,'a+'); fclose(fid); Path = [];
    else
        fid = fopen(UI_pathtemp); Path = fgetl(fid); fclose(fid);
        if Path == -1
            Path = [];
        end
    end
end
if ~exist('roi','var') || size(roi,2)~=4 || roi(3)<roi(1) || roi(4)<roi(2)
    roi = [];
end

%% Fit the file path and name for reading
if path_flag  % Load image with UI
    if ~isempty(Path) && Path(end) ~= FS
        Path = [Path,FS];
    end
    [File, Path] = uigetfile([Path,...
        '*.jpg; *.png; *.bmp; *.gif; *.tif; *.tiff; *.ppm; *.pgm; *.pbm; *.raw; *.mat'], 'Pick an image');
    if File == 0
        error('Image read cancel');
    end
    file = [Path, File];
    
elseif file_flag  % Load image with file path provided
    ff = fliplr(file);
    for i=1:size(ff,2)
        if ff(i) == FS
            break;
        end
    end
    Path = file(1:end-i+1);
    File = file(end-i+2:end);
    
else
    error('Image read failed')
end
fid = fopen(UI_pathtemp,'w+'); fprintf(fid,'%s',Path); fclose(fid);


%% Read the image (with ROI if provided)
str = lower(strsplit(File,'.'));
if strcmp(str{end},'tif') || strcmp(str{end},'tiff')
    if ~isempty(roi)
        roi = {[roi(2),roi(4)],[roi(1),roi(3)]};
    end
    try
        info = imfinfo(file);  % All TIFF
        if isfield(info,'GeoKeyDirectoryTag')
            info = geotiffinfo(file);  % Only GeoTIFF
            image = readgeoraster(file);  % Windows (new)
%             image = geotiffread(file);  % Windows & Linux (old)
        else
            image = imread(file,'PixelRegion',roi); info = [];
        end
    catch
        image = imread(file,'PixelRegion',roi); info = [];
    end
else
    if strcmp(str{end},'mat')
        image = cell2mat(struct2cell(load(file)));
    elseif strcmp(str{end},'raw')
        info = Getrawinfo;
        image = Readraw(file,info);
    elseif strcmp(str{end},'png')
        image = imread(file);
        if size(image,3)>3
            image = image(:,:,1:3);
        end
    else
        image = imread(file);  % jpg, bmp, gif, ...
    end
    info = [];
    if ~isempty(roi)
        [M,N,~] = size(image);
        roi(1) = max(1,roi(1)); roi(2) = max(1,roi(2));
        roi(3) = min(N,roi(3)); roi(4) = min(M,roi(4));
        image = image(roi(2):roi(4),roi(1):roi(3),:);
    end
end

if isa(image,'int64') || isa(image,'uint64')
    image = double(image);
end
end


%% Get information of raw
function info = Getrawinfo
title = '请输入图像参数 Input image ';
prompt = {'宽度/列数 Width ：','高度/行数 Height ：','波段/通道数 Channels ：','位宽 Bits ：'};
definput = {'640','512','1','uint16'};
dims = [1,45];
info = inputdlg(prompt,title,dims,definput);
end


%% Load raw
function img = Readraw(filename,info)
cols = str2double(info{1}); rows = str2double(info{2});
bands = str2double(info{3}); bit = info{4};
fid = fopen(filename,'r');
img = fread(fid, cols * rows * bands, bit);
img = reshape(img, [cols, rows, bands]);
fclose(fid);
end