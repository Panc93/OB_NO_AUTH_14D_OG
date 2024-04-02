# -*- coding: utf-8 -*-
import click
import logging
from pathlib import Path
import pandas as pd


# from dotenv import find_dotenv, load_dotenv


def fetch_snowflakedatasets():
    from Config.Db_connect import snowflake_conn
    """ Connect to Snowflake """
    query = "SELECT * FROM  Financebi_db.panc.OB_NOIC_ADS_ASOF0224F1 "

    return pd.read_sql_query(query, con=snowflake_conn())

def fetch_dataset_info(download):

    if download == "Yes":
        raw_dataset = fetch_snowflakedatasets()
        print(raw_dataset.shape)
        desired_data_types = {
            'progyny_rx': 'int64',
            'type_of_fully_insured_plan': 'int64',
            'rx_embedded_moop': 'int64',
            'rx_embedded_deductible': 'int64',
            'owner_changes': 'float'
        }
        raw_dataset = raw_dataset.astype(desired_data_types)

        # convert the 'Date' column to datetime format
        raw_dataset['target_date_18'] = pd.to_datetime(raw_dataset['target_date_18'])
        raw_dataset['target_date_12'] = pd.to_datetime(raw_dataset['target_date_12'])
        raw_dataset['target_date_6'] = pd.to_datetime(raw_dataset['target_date_6'])
        raw_dataset['onboard_date'] = pd.to_datetime(raw_dataset['onboard_date'])

        raw_dataset.describe().to_csv(
            r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\raw\Raw_data_describe.csv')
        raw_dataset.to_pickle(
            r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\raw\OB_NO_AUTH24_14D.pkl')
        print("Saving the data types ")
        datatype_info = pd.DataFrame(raw_dataset.dtypes)
        datatype_info.to_csv(
            r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\raw\Raw_data_datatypes.csv')
    else:
        print("Reading pickle dataset")
        raw_dataset = pd.read_pickle(
            r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\raw\OB_NO_AUTH24_14D.pkl')
    return raw_dataset

def main():
    """ Runs data processing scripts to turn raw data from (../raw) into
    raw_data ready to be analyzed (saved in ../processed)
        cleaned data ready to be analyzed (saved in ../processed).
    """
    dataset = fetch_dataset_info(download="Yes")
    print(dataset.shape)

    logger = logging.getLogger(__name__)
    logger.info('making final data set from raw data')




if __name__ == '__main__':
    log_fmt = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    logging.basicConfig(level=logging.INFO, format=log_fmt)

    # not used in this stub but often useful for finding various files
    project_dir = Path(__file__).resolve().parents[2]

    # find .env automagically by walking up directories until it's found, then
    # load up the .env entries as environment variables
    # load_dotenv(find_dotenv())

    main()


