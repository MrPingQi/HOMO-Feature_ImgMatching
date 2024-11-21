function GeoInfo = Create_GeoInfo(img,pos,TiffInfo)
if isempty(img) || isempty(pos) || isempty(TiffInfo) || isempty(TiffInfo.SpatialRef)
    GeoInfo = []; return
else
    GeoInfo = TiffInfo.SpatialRef;
    ColBase = GeoInfo.ColumnsStartFrom; RowBase = GeoInfo.RowsStartFrom;
    [rows,cols,~] = size(img);
    switch class(GeoInfo)
        case 'map.rasterref.MapCellsReference'
            [X1,Y1] = pix2map(TiffInfo.RefMatrix,pos(2),pos(1));
            [X2,Y2] = pix2map(TiffInfo.RefMatrix,pos(4),pos(3));
            GeoInfo = maprefcells([X1,X2],... % xlimits
                                  [Y2,Y1],... % ylimits
                                  [rows,cols],... % RasterSize
                                  'ColumnsStartFrom',ColBase,'RowsStartFrom',RowBase);
        case 'map.rasterref.GeographicCellsReference'
            [lat1,lon1] = pix2latlon(TiffInfo.RefMatrix,pos(2),pos(1));
            [lat2,lon2] = pix2latlon(TiffInfo.RefMatrix,pos(4),pos(3));
            GeoInfo = georefcells([lat2,lat1],... % latlimits
                                  [lon1,lon2],... % lonlimits
                                  [rows,cols],... % RasterSize
                                  'ColumnsStartFrom',ColBase,'RowsStartFrom',RowBase);
        otherwise
            error('Unrecognized format for GeoInfo');
    end
end