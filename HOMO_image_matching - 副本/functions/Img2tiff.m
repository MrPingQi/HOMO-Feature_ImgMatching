function data = Img2tiff(data,savepath)
[rows,cols,bands] = size(data);

%% Confirm data type
[data, BitsPerSample, SampleFormat] = data_type(data);

%% Duplicate tags from header
tagnamelist = Tiff.getTagNames();
tag_delete = {...
    'StripByteCounts';...
    'StripOffsets';
    'TileByteCounts';...
    'TileOffsets';...
    'MaxSampleValue';...
    'MinSampleValue';...
    'ResolutionUnit'};
for i = 1:length(tag_delete) % remove read only tag names
    tagnamelist(strcmpi(tag_delete{i},tagnamelist)) = [];
end

%% Update tags determined from imgdata and datatype
tagstruct.ImageLength     = rows;
tagstruct.ImageWidth      = cols;
tagstruct.SamplesPerPixel = bands;
tagstruct.SampleFormat    = SampleFormat;
tagstruct.BitsPerSample   = BitsPerSample;

%% Update some default tags (these tags can be overriden by user input below)
tagstruct.Compression = Tiff.Compression.None;
tagstruct.Photometric = Tiff.Photometric.MinIsBlack;
tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
% tagstruct.Software = 'MATLAB';

%% Creat a Tiff object, set tags, and write to tif file
t = Tiff(savepath,'w');
t.setTag(tagstruct);
t.write(data);
t.close();
end


%%
function [data, BitsPerSample, SampleFormat] = data_type(data)
dtype = class(data);
switch lower(dtype)
    case 'logical'
        BitsPerSample = 1;
        SampleFormat  = 1;
        data = logical(data);
    case 'uint8'
        BitsPerSample = 8;   
        SampleFormat  = 1;         
        data = uint8(data);    
    case 'int8'
        BitsPerSample = 8;
        SampleFormat  = 2;       
        data = int8(data);          
    case 'uint16'
        BitsPerSample = 16;       
        SampleFormat  = 1;         
        data = uint16(data);           
    case 'int16'
        BitsPerSample = 16;       
        SampleFormat  = 2;         
        data = int16(data);        
    case 'uint32'
        BitsPerSample = 32;       
        SampleFormat  = 1;                
        data = uint32(data);            
    case 'int32'
        BitsPerSample = 32;       
        SampleFormat  = 2;     
        data = int32(data);         
     case 'single'
        BitsPerSample = 32;        
        SampleFormat  = 3;                
        data = single(data);       
    case {'uint64','int64','double'}
        BitsPerSample = 64;        
        SampleFormat  = 3;                
        data = double(data);           
    otherwise
        error('Invalid output data type.');
end
end