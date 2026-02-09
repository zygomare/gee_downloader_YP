
def gen_toa(proc,input_files,datestr,output_dir,**kwargs):
    '''
    convert the original L1 image which is downloaded from the official release to
    the TOA reflectance (tif) compatiable with GAAC,

    ## currently, the ACOLITE functions are called
    '''

    return globals()[f'__{proc}_toa'](input_files,output_dir,datestr,**kwargs)


def __acolite_toa(input_files,output_dir,datestr,**kwargs):
    from .acolite_toa import gen_toa as gen_toa_acolite
    from .acolite_toa import combine_toa
    import pathlib
    import shutil, os

    l1r_files, l1r_setu, l1_bundle, sensor = gen_toa_acolite(input_files,output_dir,datestr,**kwargs)

    l1r_dir = pathlib.Path(l1r_files[0]).parent

    remove_temp = True if 'remove_temp' not in kwargs else kwargs['remove_temp']

    aoi_name = 'none' if 'aoi_name' not in kwargs else kwargs['aoi_name']
    l1toa_f, rgb_f = combine_toa(str(l1r_dir),output_dir=output_dir, sensor=sensor, aoi_name=aoi_name, remove_temp = remove_temp)
    return l1toa_f, rgb_f
