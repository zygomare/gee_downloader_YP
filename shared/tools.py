import pandas as pd
import glob, os

def combine_download_info(info_dir,output_csv_path, filter='*_imagecollection_optical_info.csv', snow_ice_threshold=0, cloud_threshold=5):
    '''
    combine the csv files in the info_dir with filter,
    and filter the images with snow_ice_percentage <= snow_ice_threshold and cloud_percentage <= cloud_threshold,
    and save the combined csv file to output_csv_path

    the csv files can be used in the configuration of the downloader to download the images, and the csv file should contain the following columns:
    #date_csv = /mnt/0_ARCTUS_Projects/30_Eeyou_IZMAPPING/data/good_images.csv
    '''
    csv_fs = glob.glob(os.path.join(info_dir, filter))
    if len(csv_fs) == 0:
        raise Exception(f'no csv files found in {info_dir} with filter {filter}')

    dfs = []
    for csv_f in csv_fs:
        df = pd.read_csv(csv_f)
        df['name'] = os.path.basename(csv_f).split('_')[0]
        dfs.append(df)
    combine_df = pd.concat(dfs, ignore_index=True)
    combine_df_filter = combine_df[(combine_df['snow_ice_percentage'] <= snow_ice_threshold) & (combine_df['cloud_percentage'] <= cloud_threshold)]

    combine_df_filter.loc[:,'date'] = combine_df_filter['date'].str.replace('-', '')
    combine_df_filter.to_csv(output_csv_path, index=False)


if __name__ == '__main__':
    info_dir = '/home/yan/WorkSpace/projects/DFO-IZ-MAPPING/data_test_downloader/isea3h_12_aoi_intertidal'
    output_csv_path = '/home/yan/WorkSpace/projects/DFO-IZ-MAPPING/data_test_downloader/good_images.csv'
    combine_download_info(info_dir, output_csv_path)


