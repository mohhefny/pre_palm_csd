'''
fill todo
'''

def create_coordinates(x_min, y_min, x_max, y_max, x_res, y_res, crs, fo, dir, lat_utm) -> None:
    '''
    xxxxx
    
    example usage:
        create_coordinates(389442, 5820226, 300, 250, 3, 3, '33N', 'test_x.nc', 'x', 'utm')
      
    Parameters
    ----------
    x_min : floot
        xxxx.
    lat_utm : str
        xxxxx.
    num_cols : int
        xxxx.

    Returns
    -------
    None
    Saves a netCDF file with the name fo.

    '''
    
    from osgeo import gdal, osr
    import numpy as np

    # Choose projection
    valid_crs = "28-37N"
    if crs == "28N":
        EPSG = 25828
    elif crs == "29N":
        EPSG = 25829
    elif crs == "30N":
        EPSG = 25830
    elif crs == "31N":
        EPSG = 25831
    elif crs == "32N":
        EPSG = 25832
    elif crs == "33N":
        EPSG = 25833
    elif crs == "34N":
        EPSG = 25834
    elif crs == "35N":
        EPSG = 25835
    elif crs == "36N":
        EPSG = 25836
    elif crs == "37N":
        EPSG = 25837
    else:
        raise ValueError(f"Invalid input_arg: {crs}. Valid values are: {valid_crs}")

    # Check direction and raster projection
    valid_vars = ['x', 'y']
    if dir not in valid_vars:
        raise ValueError(f"Invalid dir: {dir}. Valid values are: {', '.join(valid_vars)}")
    valid_vars = ['utm', 'latlon']
    if lat_utm not in valid_vars:
        raise ValueError(f"Invalid dir: {lat_utm}. Valid values are: {', '.join(valid_vars)}")

    # Calculate number of columns and rows
    num_rows = int((y_max - y_min) / y_res)
    num_cols = int((x_max - x_min) / x_res)
    x_max = x_min + num_cols * x_res
    y_max = y_min + num_rows * y_res

    # Create new netCDF file
    driver = gdal.GetDriverByName('netCDF')
    out_ds = driver.Create(fo, num_cols, num_rows, 1, gdal.GDT_Float32)

    # Set the geotransform
    geotransform = (x_min, x_res, 0, y_max, 0, -y_res)
    out_ds.SetGeoTransform(geotransform)

    # Set the projection
    srs_utm = osr.SpatialReference()
    srs_utm.ImportFromEPSG(EPSG)
    out_ds.SetProjection(srs_utm.ExportToWkt())

    # Conversion to latlon projection, when necessary
    srs_latlon = osr.SpatialReference()
    srs_latlon.ImportFromEPSG(4326)  # Use WGS84

    # create a transform object to convert between coordinate systems
    transform = osr.CoordinateTransformation(srs_utm,srs_latlon)

    # Creat the x- and y-coordinates
    xmin = x_min + x_res / 2.0
    xmax = xmin + num_cols * x_res
    ymin = y_max - num_rows * y_res + y_res / 2
    ymax = y_max
    x_cor = np.arange(xmin, xmax, x_res)
    y_cor = np.arange(ymin, ymax, y_res)

    # Create and write the data to the raster band
    raster_data = np.zeros((num_rows, num_cols), dtype=np.float32)
    out_band = out_ds.GetRasterBand(1)
    if dir == "x":
        if lat_utm == "utm":
            for i in range(num_cols):
                raster_data[:,i] = x_cor[i]
            out_band.WriteArray(raster_data)
        else:
            for i in range(num_cols):
                for j in range(num_rows):
                    latlong = transform.TransformPoint(x_cor[i],y_cor[j])
                    raster_data[j,i] = latlong[1]
            out_band.WriteArray(raster_data)
    else:
        if lat_utm == "utm":
            for j in range(num_rows):
                raster_data[j,:] = y_cor[j]
            out_band.WriteArray(raster_data)
        else:
            for i in range(num_cols):
                for j in range(num_rows):
                    latlong = transform.TransformPoint(x_cor[i],y_cor[j])
                    raster_data[j,i] = latlong[0]
            out_band.WriteArray(raster_data)


    # Set metadata for the raster band
    out_band.SetMetadata({'units': 'meters'})

    # Close the file
    out_ds = None


def create_feature_from_geotif(fi, fo, x_min, y_min, x_max, y_max, x_res, y_res, write_no_value=False, no_value=0) -> None:
    """
    Generate a netCDF file containing feature data from an input GeoTIFF file, 
    by cutting it to a specified domain size and resolution.

    create_feature_from_geotif canbe used to generate the following palm_csd files:
        1- file_zt (Terrain height)
        2- file_vegetation_height (Vegetation height)


    Parameters:
        fi (str): Input GeoTIFF file.
        fo (str): Output netCDF file.
        x_min (float): Minimum X coordinate of the output domain.
        y_min (float): Minimum Y coordinate of the output domain.
        x_max (float): Maximum X coordinate of the output domain.
        y_max (float): Maximum Y coordinate of the output domain.
        x_res (float): Output resolution in the X direction.
        y_res (float): Output resolution in the Y direction.

    Returns:
        None.
    """
    from osgeo import gdal
    import os

    # Open the input GeoTIFF file
    ds = gdal.Open(fi)

    # Report the input raster size
    gtrf = ds.GetGeoTransform()
    print('Origin before cutting: '+str(gtrf[0])+'/'+str(gtrf[3])+'. Resolution: '+str(gtrf[1])+' m.')
    print('Raster Size: X: '+str(ds.RasterXSize)+' Y: '+str(ds.RasterYSize))

    # Cut the input raster to the required domain size and resolution
    tmpfile = os.path.join(os.path.dirname(fo), 'tmp.tif')
    tmpds = gdal.Translate(tmpfile, ds, projWin=[x_min, y_max, x_max, y_min], xRes=x_res, yRes=y_res)
    # Report the input raster size
    gtrf = tmpds.GetGeoTransform()
    print('Origin after cutting: '+str(gtrf[0])+'/'+str(gtrf[3])+'. Resolution: '+str(gtrf[1])+' m.')
    print('Raster Size: X: '+str(ds.RasterXSize)+' Y: '+str(ds.RasterYSize))

    array = tmpds.ReadAsArray()
    num_rows, num_cols = array.shape

    # Further processing of the array, when necessary
    if write_no_value:
        array[array == no_value] = -9999
    
    # Explicitly close the dataset Remove the temporary file
    tmpds = None 
    os.remove(tmpfile)


    # Create new netCDF file
    out_ds = gdal.GetDriverByName('netCDF').Create(fo, num_cols, num_rows, 1, gdal.GDT_Float32)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)

    # Set no_data value, when requested
    if write_no_value:
        out_band.SetNoDataValue(-9999)

    # Set metadata for the raster band
    out_band.SetMetadata({'units': 'meters'})

    # Set the geotransform
    geotransform = (x_min, x_res, 0, y_max, 0, -y_res)
    out_ds.SetGeoTransform(geotransform)

def create_buildings_2d(fi, fo_prefix, x_min, y_min, x_max, y_max, x_res, y_res, att_bh, att_bt, att_id) -> None:
    '''
    Create 2D netCDF raster files (height, type, id) of buildings from a vector layer with attributes.

    Parameters:
    fi (str): file name with Path to input vector layer.
    fo_prefix (str): Path with the prefix to output netCDF files.
    x_min (float): Minimum x-coordinate of the output raster extent.
    y_min (float): Minimum y-coordinate of the output raster extent.
    x_max (float): Maximum x-coordinate of the output raster extent.
    y_max (float): Maximum y-coordinate of the output raster extent.
    x_res (float): Desired output raster resolution in x-direction.
    y_res (float): Desired output raster resolution in y-direction.
    att_bh (str): Name of the buildngs height attribute to be rasterized.
    att_bt (str): Name of the buildngs type attribute to be rasterized.
    att_id (str): Name of the buildngs id attribute to be rasterized.

    ToDo:
    - add an option to set ALL_TOUCHED to TRUE
    - enable processing of multiple attributes at once

    Returns:
        None.
    '''

    from osgeo import ogr, gdal
    import os

    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(fi)
    srclayer = ds.GetLayer()

    xres = int((x_max - x_min) / x_res)
    yres = int((y_max - y_min) / y_res)

    atts = [att_bh, att_bt, att_id]
    units = ['m', 'type', 'type']
    fo_ext = ['building_height', 'type', 'id']
    no_value = [-9999, -9999, -9999]

    # Create temporary file
    tmp_file = os.path.join(os.path.dirname(fo_prefix), 'tmp.tif')
    trgds = gdal.GetDriverByName('GTiff').Create(tmp_file, xres, yres, 1, gdal.GDT_Float32)
    trgds.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))
    band = trgds.GetRasterBand(1)
    band.SetNoDataValue(-9999)

    for i, att in enumerate(atts):
        # Rasterize layer
        gdal.RasterizeLayer(trgds, [1], srclayer, options=['ATTRIBUTE=' + att])

        # Read raster as array
        array = trgds.ReadAsArray()

        # Create output file
        out_file = fo_prefix + '_' + fo_ext[i] + '.nc'
        driver = gdal.GetDriverByName('netCDF')
        out_ds = driver.Create(out_file, xres, yres, 1, gdal.GDT_Float32)
        out_band = out_ds.GetRasterBand(1)
        out_band.WriteArray(array)
        out_band.SetNoDataValue(no_value[i])

        # Set metadata and geotransform for the raster band and close the file
        out_band.SetMetadata({'units': units[i]})
        geotransform = (x_min, x_res, 0, y_max, 0, -y_res)
        out_ds.SetGeoTransform(geotransform)
        out_ds = None

    # Remove temporary file
    os.remove(tmp_file)

def cut_geotif(fi, fo, x_min, x_max, y_min, y_max) -> None:
    '''
    Cuts a GeoTIFF input file into an output file with bounds xmin/max ymin/max and the same resolution.
    The output file type should be designated with format in mind: ".asc" for ASCII.

    Example usage:
        cut_geotif('leiptig_terrain.tif', 'leiptig_terrain_crop.tif', 
                   x_min=730000, x_max=742000, y_min=190000, y_max=202000)

    Parameters
    ----------
    fi : str
        Filename of the input file. Provide a GeoTIFF raster file.
    fo : str
        Output filename. Set output type by choosing an appropriate file ending (.asc for ASCII).
    x_min : int
        Lower left corner x coordinate.
    x_max : int
        Upper right corner x coordinate.
    y_min : int
        Lower left corner y coordinate.
    y_max : int
        Upper right corner y coordinate.

    ToDo:
    - Check if the domain is located in the original geotiff file

    Returns
    -------
    None
    '''
    from osgeo import gdal
    
    # Open the input file
    ds = gdal.Open(fi)

    # Get the geotransform information and print it
    gtrf = ds.GetGeoTransform()
    print(f'Origin before cutting: {gtrf[0]}/{gtrf[3]}. Resolution: {gtrf[1]} m.')

    # Perform the cropping and save to the output file
    ds = gdal.Translate(fo, ds, projWin=[x_min, y_max, x_max, y_min], xRes=gtrf[1], yRes=gtrf[5])

    # Get the geotransform information again and print it
    gtrf = ds.GetGeoTransform()
    print(f'Origin after cutting: {gtrf[0]}/{gtrf[3]}. Resolution: {gtrf[1]} m.')
    print(f'Raster size: X: {ds.RasterXSize} Y: {ds.RasterYSize}')

def create_feature_type_from_geotif(fi, fo, x_min, y_min, x_max, y_max, x_res, y_res, aloc_array) -> None:
    '''
    create_feature_type_from_geotif generates a netCDF file from a GeoTIFF file that contains information
    about different features such as pavement type, water type, soil type, vegetation type, and street type.
    So it can be used to generate the following files required by palm_csd:
        file_pavement_type, file_street_type, file_vegetation_type, file_water_type

    Example usage:
        create_feature_type_from_geotif('leiptig_landuse.tif', 'leiptig_pavement_type.nc', 
                                        x_min=315000, x_max=315500, y_min=5690600, y_max=5691500,
                                        x_res=2, y_res=2, aloc_array=[[1,2,3],[1,2]])

    Parameters
    ----------
    fi : str
        Filename of the input file. Provide a GeoTIFF raster file.
    fo : str
        Output filename. Set output type by choosing an appropriate file ending for netCDF (.nc).
    x_min : int
        Lower left corner x coordinate.
    x_max : int
        Upper right corner x coordinate.
    y_min : int
        Lower left corner y coordinate.
    y_max : int
        Upper right corner y coordinate.
    x_res : float
        Desired output raster resolution in x-direction.
    y_res : float
        Desired output raster resolution in y-direction.
    att_array: int
        2D attribute array for the raster pavement classes into palm pavement classes

    ToDo:
    - Check if the domain is located in the original geotiff file
    - Check if the domain is located in the original geotiff file

    Returns
    -------
    None
    '''

    from osgeo import gdal
    import os
    import numpy as np

    # Open the input GeoTIFF file
    ds = gdal.Open(fi)

    # Cut the input raster to the required domain size and resolution
    tmpfile = os.path.join(os.path.dirname(fo), 'tmp.tif')
    tmpds = gdal.Translate(tmpfile, ds, projWin=[x_min, y_max, x_max, y_min], xRes=x_res, yRes=y_res)
    try:
        if tmpds is None:
            raise RuntimeError("Failed to create temporary file {}".format(tmpfile))

        array = tmpds.ReadAsArray()
        array = array.astype(float)
        num_rows, num_cols = array.shape

        # Explicitly close the dataset and remove the temporary file
    finally:
        tmpds = None
        os.remove(tmpfile)

    # Loop over the array elements and edit them as needed
    NAvalue = -9999
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i][j] in aloc_array[0]:
                # set the array element to the corresponding value in the second row of allocation_array
                array[i][j] = aloc_array[1][np.where(aloc_array[0] == array[i][j])[0][0]]
            else:
                # set the value to no value
                array[i][j] = NAvalue


    # Create new netCDF file
    ds = gdal.GetDriverByName('netCDF').Create(fo, num_cols, num_rows, 1, gdal.GDT_Float32)
    out_band = ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.SetNoDataValue(NAvalue)

    # Set metadata for the raster band
    out_band.SetMetadata({'units': 'meters'})

    # Set the geotransform and close the dataset
    geotransform = (x_min, x_res, 0, y_max, 0, -y_res)
    ds.SetGeoTransform(geotransform)
    ds = None

def create_empty_netcdf(fo, x_min, x_max, y_min, y_max, x_res, y_res, no_data) -> None:
    '''
    create empty netcdf_for specific domain size and resolution

    Example usage:
        create_empty_netcdf('leiptig_landuse.tif', x_min=315000, y_min=5690600, num_cols=500,
                            num_rows=500, x_res=2, y_res=2, no_data=-9999)

    Parameters
    ----------
    fo : str
        Output filename. Set output type by choosing an appropriate file ending for netCDF (.nc).
    x_min : int
        Lower left corner x coordinate.
    y_min : int
        Lower left corner y coordinate.
    num_cols : int
        number of columns.
    num_rows : int
        number of rows.
    x_res : float
        Desired output raster resolution in x-direction.
    y_res : float
        Desired output raster resolution in y-direction.
    no_data : float
        value of the nodata.

    ToDo:
 
    Returns
    -------
    None
    '''
    import netCDF4 as nc
    import numpy as np

    # Calculate number of columns and rows
    num_rows = int((y_max - y_min) / y_res)
    num_cols = int((x_max - x_min) / x_res)
    x_max = x_min + num_cols * x_res
    y_max = y_min + num_rows * y_res


    # Create the netCDF file and dimensions
    with nc.Dataset(fo, mode='w', format='NETCDF4_CLASSIC') as ncfile:
        ncfile.createDimension('y', num_rows)
        ncfile.createDimension('x', num_cols)

        # Create the variables
        y_var = ncfile.createVariable('y', np.float32, ('y',))
        x_var = ncfile.createVariable('x', np.float32, ('x',))
        data_var = ncfile.createVariable('data', np.float32, ('y', 'x',))

        # Set nodata value for the data variable
        data_var.setncattr('missing_value', no_data)

        # Write the data to the netCDF file
        y_var[:] = np.linspace(y_min, y_min + (num_rows-1)*y_res, num_rows)
        x_var[:] = np.linspace(x_min, x_min + (num_cols-1)*x_res, num_cols)
        data_var[:] = no_data

def project_geotiff(fi, fo, input_projection, output_projection) -> None:
    '''
    !!!!! ATTENTION: NOT WORKING !!!!!!!

    Example usage:
        project_geotiff('input_file.tif', 'output_file.tif', 
                        input_projection="+proj=utm +zone=32 +ellps=WGS84 +datum=WGS84 +units=m +no_defs",
                        output_projection="+proj=utm +zone=33 +ellps=WGS84 +datum=WGS84 +units=m +no_defs")

    Parameters
    ----------
    fo : str
        Output filename. Set output type by choosing an appropriate file ending for netCDF (.nc).

    ToDo:
 
    Returns
    -------
    None
    '''
    import osgeo.gdal as gdal

    # Open the input file and get the input geotransform
    input_dataset = gdal.Open(fi)
    input_geotransform = input_dataset.GetGeoTransform()

    # Create the output dataset with the desired projection and geotransform
    output_driver = gdal.GetDriverByName("GTiff")
    output_dataset = output_driver.Create(fo, input_dataset.RasterXSize, input_dataset.RasterYSize,
                                        input_dataset.RasterCount, gdal.GDT_Float32)
    output_dataset.SetProjection(output_projection)
    output_dataset.SetGeoTransform(input_geotransform)

    # Reproject the input data into the output dataset
    gdal.ReprojectImage(input_dataset, output_dataset, input_projection, output_projection, gdal.GRA_Bilinear)

    # Close the input and output datasets
    input_dataset = None
    output_dataset = None

def create_feature_from_shape(fi, fo, x_min, y_min, x_max, y_max, x_res, y_res, att, units='m', no_value=-9999) -> None:
    '''
    create_feature_from_shape generates a netCDF file from a shape file that contains information
    about a feature such as tree height, tree crown diameter, and tree trunk diameter.
    So it can be used to generate the following files required by palm_csd:
        file_tree_height, file_tree_crown_diameter, file_tree_trunk_diameter

    Example usage:
        create_feature_from_shape('leiptig_trees.shp', 'leiptig_tree_height.nc', 
                                        x_min=315000, x_max=315500, y_min=5690600, y_max=5691500,
                                        x_res=2, y_res=2, att='baumhohe')

    Parameters
    ----------
    fi : str
        Filename of the input file. Provide a GeoTIFF raster file.
    fo : str
        Output filename. Set output type by choosing an appropriate file ending for netCDF (.nc).
    x_min : int
        Lower left corner x coordinate.
    x_max : int
        Upper right corner x coordinate.
    y_min : int
        Lower left corner y coordinate.
    y_max : int
        Upper right corner y coordinate.
    x_res : float
        Desired output raster resolution in x-direction.
    y_res : float
        Desired output raster resolution in y-direction.
    att: str
        attribute name

    ToDo:
    - Check if the domain is located in the original geotiff file

    Returns
    -------
    None
    '''

    from osgeo import ogr, gdal
    import os

    driver = ogr.GetDriverByName('ESRI Shapefile')
    ds = driver.Open(fi)
    srclayer = ds.GetLayer()

    # Calculate and adjust domian size
    num_col = int((x_max - x_min) / x_res)
    num_row = int((y_max - y_min) / y_res)
    x_max = x_min + num_col * x_res
    y_max = y_min + num_row * y_res

    # Create temporary file
    tmp_file = os.path.join(os.path.dirname(fo), 'tmp.tif')
    trgds = gdal.GetDriverByName('GTiff').Create(tmp_file, num_col, num_row, 1, gdal.GDT_Float32)
    trgds.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))
    band = trgds.GetRasterBand(1)
    band.SetNoDataValue(-9999)

    # Rasterize layer
    gdal.RasterizeLayer(trgds, [1], srclayer, options=['ATTRIBUTE=' + att])

    # Read raster as array
    array = trgds.ReadAsArray()

    # Create output file
    driver = gdal.GetDriverByName('netCDF')
    out_ds = driver.Create(fo, num_col, num_row, 1, gdal.GDT_Float32)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.SetNoDataValue(no_value)

    # Set metadata and geotransform for the raster band and close the file
    out_band.SetMetadata({'units': units})
    geotransform = (x_min, x_res, 0, y_max, 0, -y_res)
    out_ds.SetGeoTransform(geotransform)
    out_ds = None

    # Remove temporary file
    os.remove(tmp_file)

def create_tree_type_from_shape(fi_shp, fi_palm_tree, fo, x_min, y_min, x_max, y_max, x_res, y_res, att, no_value=-9999, unit='m', delim=';', palm_tree_name='PALM_name', palm_tree_num='PALM_number') -> None:
    '''
    create_feature_from_shape generates a netCDF file from a shape file that contains information
    about a feature such as tree height, tree crown diameter, and tree trunk diameter.
    So it can be used to generate the following files required by palm_csd:
        file_tree_height, file_tree_crown_diameter, file_tree_trunk_diameter

    Example usage:
        create_feature_from_shape('leiptig_trees.shp', 'leiptig_tree_height.nc', 
                                        x_min=315000, x_max=315500, y_min=5690600, y_max=5691500,
                                        x_res=2, y_res=2, att='baumhohe')

    Parameters
    ----------
    fi : str
        Filename of the input file. Provide a GeoTIFF raster file.
    fo : str
        Output filename. Set output type by choosing an appropriate file ending for netCDF (.nc).
    x_min : int
        Lower left corner x coordinate.
    x_max : int
        Upper right corner x coordinate.
    y_min : int
        Lower left corner y coordinate.
    y_max : int
        Upper right corner y coordinate.
    x_res : float
        Desired output raster resolution in x-direction.
    y_res : float
        Desired output raster resolution in y-direction.
    att: str
        attribute name of tree species

    ToDo:
    - Check if the domain is located in the original geotiff file

    Returns
    -------
    None
    '''

    from osgeo import ogr, gdal
    import os
    import csv

    # Open the shapefile and get the layer
    driver = ogr.GetDriverByName('ESRI Shapefile')
    temp_file = os.path.join(os.path.dirname(fo), 'temp.shp')
    ds = driver.CopyDataSource(ogr.Open(fi_shp), temp_file)
    #ds = driver.Open(fi_shp, 1) # 1 for writing mode
    layer = ds.GetLayer()

    # Load the tree species PALM data (species and type) from the CSV file
    tree_species_palm = {}
    with open(fi_palm_tree, 'r') as f:
        reader = csv.DictReader(f,delimiter=delim)
        for row in reader:
            tree_species_palm[row[palm_tree_name]] = row[palm_tree_num]

    # Create a new field in the shapefile layer for the palm type attribute
    palm_type_field = ogr.FieldDefn('palm_type', ogr.OFTString)
    layer.CreateField(palm_type_field)

    # Loop through the features in the layer and set the palm type attribute based on the allocation data
    for feature in layer:
        tree_species = feature.GetField(att)
        if tree_species:
            tree_species = tree_species.lower()
            for key in tree_species_palm.keys():
                key_str = str(key)
                key_str = key_str.lower()
                if key_str in tree_species:
                    palm_type = tree_species_palm[key]
                    feature.SetField("palm_type", palm_type)
                    layer.SetFeature(feature)
                    break

    # Calculate and adjust domain size
    num_col = int((x_max - x_min) / x_res)
    num_row = int((y_max - y_min) / y_res)
    x_max = x_min + num_col * x_res
    y_max = y_min + num_row * y_res

    # Create temporary file
    tmp_file = os.path.join(os.path.dirname(fo), 'tmp.tif')
    trgds = gdal.GetDriverByName('GTiff').Create(tmp_file, num_col, num_row, 1, gdal.GDT_Float32)
    trgds.SetGeoTransform((x_min, x_res, 0, y_max, 0, -y_res))
    band = trgds.GetRasterBand(1)
    band.SetNoDataValue(no_value)

    # Rasterize layer
    gdal.RasterizeLayer(trgds, [1], layer, options=['ATTRIBUTE=palm_type'])

    # Read raster as array
    array = trgds.ReadAsArray()

    # Create output file
    driver = gdal.GetDriverByName('netCDF')
    out_ds = driver.Create(fo, num_col, num_row, 1, gdal.GDT_Float32)
    out_band = out_ds.GetRasterBand(1)
    out_band.WriteArray(array)
    out_band.SetNoDataValue(no_value)

    # Set metadata and geotransform for the raster band and close the file
    out_band.SetMetadata({'units': unit})
    geotransform = (x_min, x_res, 0, y_max, 0, -y_res)
    out_ds.SetGeoTransform(geotransform)
    out_ds = None

    # Remove temporary file
    os.remove(tmp_file)


# # Test
# fi = '/home/mohamed/Desktop/TMP/geodata/example_buildings.shp'
# fi = '/home/mohamed/Desktop/TMP/geodata/pavement_johannapark.tif'
# fi = '/home/mohamed/Desktop/TMP/geodata/landuse_alkis_johannapark.tif'
# fi = '/home/mohamed/Desktop/TMP/geodata/vegetationheight_johannapark.tif'
# #fi = '/home/mohamed/Desktop/TMP/geodata/dtm_johannapark.tif'

# fo_prefix = '/home/mohamed/Desktop/TMP/geodata/example_buildings'
# fo = '/home/mohamed/Desktop/TMP/leipzig_csd/input/Leipzig_CoordinatesLatLon_y_2m.nc'
# #x_min, y_min, x_max, y_max, x_res, y_res = 682925, 246520, 683025, 246620, 2, 2
# x_min, y_min, x_max, y_max, x_res, y_res = 315000, 5690600, 316000, 5691500, 2, 2  # Leipzig Johannapark
# #x_min, y_min, x_max, y_max, x_res, y_res = 386749, 5818426, 390349, 5820676, 15, 15 # Berlin
# att_bh, att_bt, att_id = 'HEIGHT_TOP', 'BLDGTYP', 'ID'

# num_cols, num_rows = 241, 151
# dir = 'y'
# lat_utm = 'latlon'
# crs = '33N'
# no_data = -9999

# #create_empty_netcdf(fo, x_min, y_min, num_cols, num_rows, x_res, y_res, no_data)

# create_coordinates(x_min, y_max, num_cols, num_rows, x_res, y_res, crs, fo, dir, lat_utm)

# #create_feature_from_geotif(fi, fo, x_min, y_min, x_max, y_max, x_res, y_res, write_no_value=True, no_value=0.5)

# #create_feature_type_from_geotif(fi, fo, x_min, y_min, x_max, y_max, x_res, y_res, [[20,21,7],[1,2]])
# #
# #cutalti(fi, fo, x_min, x_max, y_min, y_max)

# #create_buildings_2d(fi, fo_prefix, x_min, y_min, x_max, y_max, x_res, y_res, att_bh, att_bt, att_id)