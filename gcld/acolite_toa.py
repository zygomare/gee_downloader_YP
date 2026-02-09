import shutil

import rasterio
import glob,os
from tqdm import tqdm
from os.path import join as pjoin
from os.path import split as psplit
import numpy as np
import inspect
from .utils import sensor_info, IlegalAcoliteL1RTifs, wavelength_matching

import logging

_logger = logging.getLogger('Combine ACOLITE TOA')

def combine_toa(l1r_dir,output_dir, sensor='PLE_MSI',aoi_name='none', remove_temp=False, cog=True):
    '''
    combine tif files as one in acolite output dir,
    '''
    ### output for old version of acolite
    # tifs = glob.glob(pjoin(l1r_dir, f'*rhot_*.tif'))
    # rgbs = glob.glob(pjoin(l1r_dir, f'*rhot_RGB.tif'))
    # if len(rgbs)>0:
    #     tifs.remove(rgbs[0])

    tifs = glob.glob(pjoin(l1r_dir, f'*rhot_*.tif'))
    rgbf = None
    for tif in tifs.copy():
        if psplit(tif)[-1].find('rgb') > -1 or  psplit(tif)[-1].find('RGB')> -1:
            tifs.remove(tif)
            rgbf = tif
            shutil.move(rgbf, output_dir)
            rgbf = pjoin(output_dir, psplit(rgbf)[-1])

    # tifs = glob.glob(pjoin(l1r_dir, f'*_L1R_rhot_*.tif'))
    ## new version of acolite, the rgb tif is named as *_L1R_rgb_rhot.tif
    rgbf = glob.glob(pjoin(l1r_dir, f'*_L1R_rgb_rhot.tif'))[0] if len(glob.glob(pjoin(l1r_dir, f'*_L1R_rgb_rhot.tif'))) > 0 else rgbf
    if rgbf is not None:
        shutil.move(rgbf, output_dir)
        rgbf = pjoin(output_dir, psplit(rgbf)[-1])

    bands_dic = sensor_info[sensor]['bands']

    # if len(tifs)!= len(bands_dic.keys()):
    #     raise IlegalAcoliteL1RTifs(sensor, len(tifs), len(bands_dic.keys()))

    ### check if all the bands in sensorinfo are included in the tif files
    wavelength_info = list(bands_dic.values())
    wave_tif_dic = {}
    for tif in tifs.copy():
        wavelength_tif = int(tif.split('_')[-1].replace('.tif',''))
        print(f'wavelength in tif name : {wavelength_tif}')
        wavelength_ref = wavelength_matching(wavelength_tif, np.asarray(wavelength_info))

        if wavelength_ref is not None:
            wavelength_info.remove(wavelength_ref)
            wave_tif_dic[wavelength_ref] = tif
        else:
            tifs.remove(tif)


    if len(wavelength_info) > 0:
        raise  IlegalAcoliteL1RTifs(sensor, len(tifs), len(bands_dic.keys()))

    fname = psplit(tifs[0])[-1]
    basename = fname.split('_rhot')[0]

    pre = fname.split('_')[0]
    datestr = ''.join(fname.split('_')[2:5]) ## + 'T' + ''.join(fname.split('_')[5:8]) keep consistent with gee_downloader

    with rasterio.open(tifs[0]) as src:
        meta = src.meta
        tags = src.tags()
    resolution = int(meta['transform'][0])
    # print(meta)

    out_f = pjoin(output_dir, f'{pre}_L1TOA_{datestr}_{aoi_name}_{resolution}m.tif')

    meta['count'] = len(tifs)
    wavelengths = sorted(bands_dic.values())

    bands_dic_reverse = {}
    for key in bands_dic:
        bands_dic_reverse[bands_dic[key]] = key

    if os.path.exists(pjoin(l1r_dir, f'{basename}_sza.tif')):
        meta['count'] = len(tifs) + 3

    if not cog:
        meta.update({'compress':'lzw','tiled':True, 'blockxsize':512, 'blockysize':512, 'BIGTIFF':'YES'})
    else:
        meta.update(
            {'driver':"COG",
            'compress': "DEFLATE",
            'predictor': 2,
            'overview_resampling': "average"
             }
        )


    with rasterio.open(out_f, 'w', **meta) as dst:
        bandnames = []
        for i, w in tqdm(enumerate(wavelengths, start=1), total=len(wavelengths),desc=f'write rhot'):
            _logger.info(f'{bands_dic_reverse[w]}, {w}')
            with rasterio.open(wave_tif_dic[w]) as src:
                d = src.read(1)
            dst.write(d, i)
            dst.set_band_description(i, f'rhot_{w}')
            bandnames.append(bands_dic_reverse[w])

        if meta['count'] == len(tifs) + 3:
            for i, g in tqdm(enumerate(['sza','vza','raa'], start=len(wavelengths)+1), desc=f'write geometry'):
                _logger.info(g)
                with rasterio.open(pjoin(l1r_dir, f'{basename}_{g}.tif')) as src:
                    d = src.read(1)
                dst.write(d, i)
                dst.set_band_description(i, g)
                bandnames.append(g)

        dst.update_tags(ns='band_names', bandnames=','.join(bandnames), wavelengths = ','.join([str(w) for w in wavelengths]))
        dst.update_tags(ns='geometry', solz = tags['NC_GLOBAL#sza'], senz = tags['NC_GLOBAL#vza'], phi = tags['NC_GLOBAL#raa'])
        dst.update_tags(descriptions = 'combined acolite generated L1TOA reflectance')
        dst.update_tags(ns = 'sensor', sensor_name = sensor, sensing_time = tags['NC_GLOBAL#isodate'])

    if remove_temp:
        # for temp_f in glob.glob(os.path.join(l1r_dir, f"{basename}*")):
        #     if temp_f == out_f or rgbf == temp_f:
        #         continue
        #     os.remove(temp_f)
        shutil.rmtree(l1r_dir)

    return out_f, rgbf

def gen_toa(input_files,output_dir,datestr,**kwargs):
    import acolite as ac
    from os.path import split as psplit
    '''
    call acolite to generate TOA refectance
    limit: south, west,north,east
    '''
    limit = None if 'limit' not in kwargs else kwargs['limit']
    geojson_f = None if 'geojson_f' not in kwargs else kwargs['geojson_f']
    s2_target_res = 10 if 's2_target_res' not in kwargs else int(kwargs['s2_target_res'])
    merge_tiles = False if 'merge_tiles' not in kwargs else bool(kwargs['merge_tiles'])
    reproject_inputfile = False if 'reproject_inputfile' not in kwargs else bool(kwargs['reproject_inputfile'])

    # print(input_files)
    acolite_output_dir = os.path.join(output_dir, f'acolite_{datestr}')

    settings = {
        'inputfile': input_files,
        'output': acolite_output_dir,
        'atmospheric_correction': False,
        'ancillary_data': True,
        'map_raster': False,
        's2_target_res': s2_target_res,
        'output_geometry': True,
        'l1r_export_geotiff_rgb': True,
        'l1r_export_geotiff':True,
        'use_gdal_merge_import': False,
        'merge_tiles' : merge_tiles,
        'reproject_inputfile':reproject_inputfile
    }
    if limit is not None:
        settings.update({'limit': limit})
    if type(input_files)==list and len(input_files) > 0:
        settings.update({'merge_tiles': True})
    if geojson_f is not None:
        settings.update(
            {'polygon': geojson_f,
             'polygon_limit': True}
        )

    ######## This was added to reset the setu_user from previous runs
    ac.settings = {}
    ## read default processing settings
    ac.settings['defaults'] = ac.acolite.settings.parse(None, settings=ac.acolite.settings.load(None), merge=False)
    ## copy defaults to run, run will be updated with user settings and sensor defaults
    ac.settings['run'] = {k:ac.settings['defaults'][k] for k in ac.settings['defaults']}
    ## empty user settings
    ac.settings['user'] = {}
    ##############

    #setu_user = ac.acolite.settings.merge(settings)
    setu_user = ac.acolite.settings.merge(settings)
    ac.settings['run'] = setu_user
    ac.settings['user'] = settings

    ## set user settings and update run settings
    identification = ac.acolite.identify_bundle(input_files[0], output=setu_user['output'])
    input_type = identification[0]
    print(f'input_type for the bundle is: {input_type}')
    sensor = None
    if input_type.startswith('Sentinel-2'):
        print(psplit(input_files[0])[1])
        if psplit(input_files[0])[1].find('S2A') > -1:
            sensor = 'S2A_MSI'
        else:
            sensor = 'S2B_MSI'
    elif input_type == 'Landsat':
        if psplit(input_files[0])[1].find('LC8') > -1 or psplit(input_files[0])[1].find('LC08') > -1:
            sensor = 'L8_OLI'
        else:
            sensor = 'L9_OLI2'
    elif input_type == 'Planet':
        if psplit(input_files[0])[1].find('psscene_analytic_8b') > -1:
            sensor = 'PN_SD8'

    elif input_type == 'Pléiades':
        # if psplit(input_files[0])[1].find('psscene_analytic_8b') > -1:
        sensor = 'PLE_HRMSI'

    # Read bundle
    ret = ac.acolite.acolite_l1r(input_files, setu_user)

    l1r_files, l1r_setu, l1_bundle = ret
    l1r_setu['l1r_export_geotiff'] = True
    l1r_setu['l1r_export_geotiff_rgb'] = True
    for l1r in l1r_files:
        nc_to_geotiff = ac.output.nc_to_geotiff
        # print(nc_to_geotiff)
        # print(l1r)
        sig = inspect.signature(nc_to_geotiff)

        if 'match_file' in sig.parameters.keys():
            ### the old version requires the argument of 'match_file'
            nc_to_geotiff(l1r,
                          match_file=l1r_setu['export_geotiff_match_file'],
                          cloud_optimized_geotiff=l1r_setu[
                              'export_cloud_optimized_geotiff'],
                          skip_geo=l1r_setu[
                                       'export_geotiff_coordinates'] is False)
        else:
            # nc_to_geotiff(l1r, settings=l1r_setu)
            nc_to_geotiff(l1r,settings=l1r_setu)

        ac.output.nc_to_geotiff_rgb(l1r, settings=l1r_setu, remove_temp_files=True)

    #ac.settings = {}
    return l1r_files, l1r_setu, l1_bundle, sensor
