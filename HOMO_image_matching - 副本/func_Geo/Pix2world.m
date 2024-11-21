function points = Pix2world(points, DataInfo)
if isempty(DataInfo) || isempty(points)
    fprintf('No available GeoInfo to use\n');
    return
end

row = points(:,2); col = points(:,1);
switch class(DataInfo.SpatialRef)
    case 'map.rasterref.MapCellsReference'
        [world_x,world_y] = pix2map(DataInfo.RefMatrix,row,col);
        [lat,lon] = projinv(DataInfo,world_x,world_y);
    case 'map.rasterref.GeographicCellsReference'
        [lat,lon] = pix2latlon(DataInfo.RefMatrix,row,col);
    otherwise
        fprintf('Unrecognized format for GeoInfo\n');
        points = []; return
end
points = [lon,lat];