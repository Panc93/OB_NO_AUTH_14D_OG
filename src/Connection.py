
import pandas as pd
import configparser
import snowflake.connector
from snowflake.connector.pandas_tools import write_pandas

config = configparser.ConfigParser()
config.read(r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\Config\Config.ini')
def connect():
    return snowflake.connector.connect(
                        account=config['snowflake']['account'],
                        warehouse = config['snowflake']['warehouse'],
                        database = config['snowflake']['database'],
                        schema = config['snowflake']['schema'],
                        user = config['snowflake']['user'],
                        password = config['snowflake']['password']
                                                   )
conn_sf = connect()
cursor_sf = conn_sf.cursor()
#%%
#! Functions

# ! Snowflake Functions
def execute_snowflake():
    select_query = """
    SELECT * FROM financebi_db.mxp.source_all_calls limit 20
    """

    return(select_query)
#%%
# * run snowflake queries (note: does not execute at this step)
s1 = execute_snowflake()
cursor_sf.execute(s1)
df = cursor_sf.fetch_pandas_all()

df.columns
#%%
# * define snowflake tables to save to

#%%
# * write to snowflake --
cursor_sf.execute(s2)
cursor_sf.execute(s3)
write_pandas(conn_sf,df=df, database=db, schema=schema, table_name=tbl) # expected to append if table exists; use overwrite=True if want to overwrite
#%%
# * select query
cursor_sf.execute(s4)
df2 = cursor_sf.fetch_pandas_all()
# df2.info() # should have 200 rows bc of s3
#%%
# * close connection
conn_sf.close()

# %%
# Package                    Version
# -------------------------- -----------
# absl-py                    1.4.0
# aiobotocore                2.9.0
# aiohttp                    3.9.1
# aioitertools               0.11.0
# aiosignal                  1.3.1
# anyio                      3.7.1
# asn1crypto                 1.5.1
# asttokens                  2.2.0
# async-timeout              4.0.3
# attrs                      22.1.0
# backcall                   0.2.0
# bcrypt                     4.1.2
# blis                       0.7.9
# boto3                      1.33.1
# botocore                   1.33.13
# cachetools                 4.2.4
# catalogue                  2.0.8
# certifi                    2022.12.7
# cffi                       1.15.1
# charset-normalizer         3.0.1
# click                      8.1.3
# cloudpickle                2.0.0
# colorama                   0.4.6
# colour                     0.1.5
# confection                 0.0.4
# contourpy                  1.0.7
# cryptography               40.0.2
# cycler                     0.11.0
# cymem                      2.0.7
# debugpy                    1.6.4
# decorator                  5.1.1
# docopt                     0.6.2
# en-core-web-sm             3.5.0
# entrypoints                0.4
# et-xmlfile                 1.1.0
# exceptiongroup             1.2.0
# executing                  1.2.0
# fastjsonschema             2.16.2
# filelock                   3.12.2
# fonttools                  4.38.0
# frozenlist                 1.4.1
# fsspec                     2023.12.2
# graphviz                   0.20.1
# greenlet                   2.0.1
# idna                       3.4
# importlib-resources        5.13.0
# ipykernel                  6.17.1
# ipython                    8.7.0
# jedi                       0.18.2
# Jinja2                     3.1.2
# jmespath                   1.0.1
# joblib                     1.2.0
# jsonschema                 4.17.3
# jupyter_client             7.4.7
# jupyter_core               5.1.0
# kiwisolver                 1.4.4
# langcodes                  3.3.0
# Logbook                    1.7.0.post0
# MarkupSafe                 2.1.2
# matplotlib                 3.6.3
# matplotlib-inline          0.1.6
# multidict                  6.0.4
# murmurhash                 1.0.9
# mysql-connector            2.2.9
# mysql-connector-python     8.3.0
# mysqlclient                2.1.1
# nbformat                   5.7.3
# nest-asyncio               1.5.6
# networkx                   3.1
# nltk                       3.8.1
# numpy                      1.23.5
# openpyxl                   3.0.10
# oscrypto                   1.3.0
# packaging                  21.3
# pandas                     1.5.2
# paramiko                   3.4.0
# parso                      0.8.3
# pathy                      0.10.1
# pendulum                   2.1.2
# pickleshare                0.7.5
# Pillow                     9.4.0
# pip                        23.3.2
# pipreqs                    0.4.13
# platformdirs               3.11.0
# plotly                     5.13.0
# preshed                    3.0.8
# prompt-toolkit             3.0.33
# psutil                     5.9.4
# pure-eval                  0.2.2
# py4j                       0.10.9.7
# pyarrow                    14.0.2
# pycparser                  2.21
# pycryptodomex              3.18.0
# pydantic                   1.10.4
# Pygments                   2.13.0
# PyJWT                      2.7.0
# pylance                    0.9.6
# PyMySQL                    1.0.2
# PyNaCl                     1.5.0
# pyOpenSSL                  23.2.0
# pyparsing                  3.0.9
# pyrsistent                 0.19.2
# pysftp                     0.2.9
# pyspark                    3.4.1
# python-dateutil            2.8.2
# python-dotenv              1.0.0
# python-postman             0.3.0
# pytimeparse                1.1.8
# pytz                       2022.6
# pytzdata                   2020.1
# pywin32                    305
# PyYAML                     6.0.1
# pyzmq                      24.0.1
# regex                      2022.10.31
# requests                   2.28.2
# retry2                     0.9.5
# retrying                   1.3.4
# s3fs                       2023.12.2
# s3transfer                 0.8.0
# scikit-learn               1.2.1
# scipy                      1.10.0
# seaborn                    0.12.2
# six                        1.16.0
# smart-open                 6.3.0
# sniffio                    1.3.0
# snowflake-connector-python 3.6.0
# snowflake-ml-python        1.1.2
# snowflake-snowpark-python  1.11.1
# sortedcontainers           2.4.0
# spacy                      3.5.0
# spacy-legacy               3.0.12
# spacy-loggers              1.0.4
# SQLAlchemy                 1.4.44
# sqlparse                   0.4.4
# srsly                      2.4.5
# stack-data                 0.6.2
# styleframe                 4.1
# tenacity                   8.2.1
# thinc                      8.1.7
# threadpoolctl              3.1.0
# tomlkit                    0.12.3
# tornado                    6.2
# tqdm                       4.64.1
# traitlets                  5.6.0
# typer                      0.7.0
# typing_extensions          4.4.0
# urllib3                    1.26.14
# wasabi                     1.1.1
# wcwidth                    0.2.5
# wheel                      0.40.0
# wordcloud                  1.8.2.2
# wrapt                      1.16.0
# xgboost                    1.7.6
# XlsxWriter                 3.0.3
# yarg                       0.1.9
# yarl                       1.9.4
