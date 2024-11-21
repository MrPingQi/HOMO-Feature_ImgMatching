function points = World2pix(points, DataInfo)
if isempty(DataInfo) || isempty(points)
    fprintf('No available GeoInfo to use\n');
    return
end

lat = points(:,2); lon = points(:,1);
switch class(DataInfo.SpatialRef)
    case 'map.rasterref.MapCellsReference'
        [world_x,world_y] = projfwd(DataInfo,lat,lon);
        [row,col] = map2pix(DataInfo.RefMatrix,world_x,world_y);
    case 'map.rasterref.GeographicCellsReference'
        [row,col] = latlon2pix(DataInfo.RefMatrix,lat,lon);
    otherwise
        fprintf('Unrecognized format for GeoInfo\n');
        points = []; return
end
points = [col,row];