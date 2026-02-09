import numpy as np

sensor_info = {
    'S2_MSIA':{
        'bands':
            {'B1':443,'B2':492,'B3':560,'B4':665,'B5':704,'B6':740,'B7':783,
             'B8':833,'B8A':865,'B9':940,'B10':1374,'B11':1614,'B12':2202},

        'rgb':('B4', 'B3', 'B2'),

        ### refractive index, data was taken from acolite
        'refri':
            np.asarray([1.3393943220560691,1.336519038713722,1.333643405061483,1.3306334794387231,1.3297612987138283,1.3290486996216024,1.328306494174936,1.3275979713692792,1.3271748246874915,1.3262383421004027,1.3215305636555537,1.318553650283908,1.298343776306548]),

        ### optical thickness due to rayleigh scattering under 1 standard air pressure
        'rot':
            np.asarray([2.356E-01,1.558E-01,9.055E-02,4.498E-02,3.553E-02,2.897E-02,2.316E-02,
                            1.853E-02,1.549E-02,1.083E-02,2.269E-03, 1.269E-03,3.679E-04]),
        'wavelengh_seadas':
            np.asarray([443, 492, 560, 665, 704, 740, 783,835, 865,945,1613,2200])
    },

    'S2_MSIB':{
        'bands':
            {'B1':442,'B2':492,'B3':559,'B4':665,'B5':704,'B6':739,'B7':780,'B8':833,'B8A':864,'B9':943,'B10':1377,'B11':1610,'B12':2186},

        'rgb': ('B4', 'B3', 'B2'),

        'refri': np.array([1.3394266404832016,1.3365351865959296,1.3336755721296898,1.3306258364615668,1.3297672523205486,1.3290739721798959,1.3283570506945426,1.327595898595085,1.3271838686037978,1.3262591782406012,1.3214886977377136,1.318601822441023,1.2994168257542782]),
        'rot':np.asarray([2.356E-01,1.558E-01,9.055E-02,4.498E-02,3.553E-02,2.897E-02,2.316E-02,
                            1.853E-02,1.549E-02,1.083E-02,2.269E-03, 1.269E-03,3.679E-04]),
        'wavelength_seadas':
            np.asarray([442, 492, 559, 665, 704, 739, 780,835, 864,943, 1611,2184])
    },

    'S2_MSIC':{ ###############3 WARNING these values needs to be update
        'bands':
            {'B1':442,'B2':492,'B3':559,'B4':665,'B5':704,'B6':739,'B7':780,'B8':833,'B8A':864,'B9':943,'B10':1377,'B11':1610,'B12':2186},

        'rgb': ('B4', 'B3', 'B2'),

        'refri': np.array([1.3394266404832016,1.3365351865959296,1.3336755721296898,1.3306258364615668,1.3297672523205486,1.3290739721798959,1.3283570506945426,1.327595898595085,1.3271838686037978,1.3262591782406012,1.3214886977377136,1.318601822441023,1.2994168257542782]),
        'rot':np.asarray([2.356E-01,1.558E-01,9.055E-02,4.498E-02,3.553E-02,2.897E-02,2.316E-02,
                          1.853E-02,1.549E-02,1.083E-02,2.269E-03, 1.269E-03,3.679E-04]),
        'wavelength_seadas':
            np.asarray([442, 492, 559, 665, 704, 739, 780,835, 864,943, 1611,2184])
    },

    'L8_OLI':{
            # 443,482,561,655,865,1609,2201
        'bands':
            {'B1':443,'B2':482,'B3':561,'B4':655,'B5':865,'B6':1609,'B7':2201},
        'refri':np.array([1.3393708491065426,1.3370230535972705,1.3336053204608094,1.3308740597471744,1.32717690518053,1.318621928289049,1.2983747175501328]),

        'rot': np.asarray([2.352E-01, 1.685E-01, 9.020E-02, 4.793E-02, 1.551E-02, 1.284E-03, 3.697E-04]),
        'rgb': ('B4', 'B3', 'B2'),
    },

    'L9_OLI2': {
        # 443,482,561,655,865,1609,2201
        'bands':
            {'B1': 443, 'B2': 482, 'B3': 561, 'B4': 655, 'B5': 865, 'B6': 1609, 'B7': 2201},
        'refri': np.array(
            [1.3393708491065426, 1.3370230535972705, 1.3336053204608094, 1.3308740597471744, 1.32717690518053,
             1.318621928289049, 1.2983747175501328]),

        'rot': np.asarray([2.352E-01, 1.685E-01, 9.020E-02, 4.793E-02, 1.551E-02, 1.284E-03, 3.697E-04]),
        'rgb': ('B4', 'B3', 'B2'),
    },

    ### planet, super-dove, multi-spectral instrument, 8 bands
    'PN_SD8':{
        'bands': {'B1': 444, 'B2': 492, 'B3': 533, 'B4': 566, 'B5': 612, 'B6': 666, 'B7': 707, 'B8': 866},
        'rot': np.asarray([0.23363, 0.1583, 0.11096, 0.08731, 0.06364, 0.04462, 0.03513, 0.01557]),
        'rgb': ('B6', 'B4', 'B2'),
        'refri': np.asarray([1.3393275204911967,1.336538829796674,1.3346695032107483,1.3334389058308331,1.3319986315600256,1.3305937265825456,1.3297030671407954,1.32716603183579])
    },

    ### HR stands for High Resolution
    'PLE_HRMSI':{
        'bands': {'B1':439, 'B2':486, 'B3':563, 'B4':655, 'B5':724, 'B6':826}, ### exclude PAN
        'rot': np.asarray([0.25828,0.16728, 0.09014, 0.04891, 0.03232, 0.01941]),
        'rgb': ('B4','B3', 'B2'),
        'refri': np.asarray([1.3399411017280038, 1.3368754982152926, 1.3335529250983595, 1.330876041740837, 1.3293806565437194, 1.3277026329243091])

    }


}

sensor_info['S2A_MSI'] = sensor_info['S2_MSIA']
sensor_info['S2B_MSI'] = sensor_info['S2_MSIB']
sensor_info['S2C_MSI'] = sensor_info['S2_MSIC']


def wavelength_matching(t_wl, ref_wv:np.array, max_diff=10):
    '''
    @t_wl: target wavelength
    @ref_wv: wavelength as reference (ndarray)
    @max_diff: if the mininum difference is greater than max_diff, the matching fails and return None
    '''
    if ref_wv.size > 0:  # This condition was added to deal with some wavelenghts of a sensor not in the referene list
        diff = np.abs(ref_wv - t_wl)
        min_diff = diff.min()
        if max_diff < min_diff:
            return None
        return ref_wv[np.argmin(diff)]
    else:
        print('Target wavelength not in reference wavelength array')
        return None


class IlegalAcoliteL1RTifs(Exception):
    def __init__(self, sensor, count_tifs, count_band):
        self.msg = f'{count_tifs} tif files found, but {sensor} has {count_band} bands'
        super().__init__(self.msg)