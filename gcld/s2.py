import pandas as pd
def search_url_l1toa(product_id: str):
    from google.cloud import bigquery

    client = bigquery.Client()  # requires BigQuery API enabled + billing project

    # product_id = "S2A_MSIL1C_20230131T154711_N0509_R054_T29SNU_20230131T180745"

    sql = f"""
        SELECT base_url,generation_time,granule_id, sensing_time, source_url, product_id
        FROM `bigquery-public-data.cloud_storage_geo_index.sentinel_2_index`
        WHERE CONTAINS_SUBSTR(product_id, "{product_id}")
        ORDER BY generation_time DESC
        LIMIT 10
    """
    df = client.query(sql).to_dataframe()
    if df.empty:
        print(f"No S2 L1TOA URL results found for product_id: {product_id}")
        return None
    df = df.iloc[0]
    url = df['source_url'] if pd.notna(df['source_url']) else df['base_url']
    if url == "":
        print(f"Empty URL found for product_id: {product_id}")
        return None
    return url


def get_l1toa_prodid_essential(product_id: str):
    '''
    Convert Sentinel-2 L2A product ID to L1C essential product ID.

    for example,  "S2A_MSIL2A_20241005T162141_N0511_R040_T17UPT_20241005T195259" to "S2A_MSIL1C_20241005T162141_N0511_R040_T17UPT"
    '''
    if product_id.find("MSIL2A") >= 0:
        product_id = product_id.replace("L2A","L1C")
        if len(product_id.split('_')) == 7:
            product_id = '_'.join(product_id.split('_')[:6])

    return product_id




