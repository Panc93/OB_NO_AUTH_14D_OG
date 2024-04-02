from snowflake.snowpark import Session
from snowflake.snowpark.version import VERSION
from snowflake.snowpark.types import IntegerType
import snowflake.snowpark.types as T

# Snowpark ML

from snowflake.ml.modeling.xgboost import XGBClassifier
from snowflake.ml.modeling.model_selection.grid_search_cv import GridSearchCV

# warning suppresion
import warnings; warnings.simplefilter('ignore')

# Create Snowflake Session object
connection_parameters = {
    "user": "parnikapancholi",
    "password": "t7@KTCGN3e.A@37MMhz9",
    "account": "mda67638.us-east-1",
    "database": "FINANCEBI_DB",
    "warehouse": "COMPUTE_WH",
    "schema": "panc"
}
session = Session.builder.configs(connection_parameters).create()
session.sql_simplifier_enabled = True

snowflake_environment = session.sql('SELECT current_user(), current_version()').collect()
snowpark_version = VERSION
# Current Environment Details
print('\nConnection Established with the following parameters:')
print('User                        : {}'.format(snowflake_environment[0][0]))
print('Role                        : {}'.format(session.get_current_role()))
print('Database                    : {}'.format(session.get_current_database()))
print('Schema                      : {}'.format(session.get_current_schema()))
print('Warehouse                   : {}'.format(session.get_current_warehouse()))
print('Snowflake version           : {}'.format(snowflake_environment[0][1]))
print('Snowpark for Python version : {}.{}.{}'.format(snowpark_version[0],snowpark_version[1],snowpark_version[2]))
import pandas as pd
# Load the data
src =r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\final'
indsn = 'OB_NO_AUTH24_14D_V3_TR'
#data = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))

# Convert Pandas DataFrame to Snowpark DataFrame
 #session.write_pandas(data,'OB_NO_AUTH24_14D_V3_TR',auto_create_table=True,overwrite=True)
#Code starts
input_tbl = f"{session.get_current_database()}.{session.get_current_schema()}.{'OB_NO_AUTH24_14D_V3_TR'}"
# First, we read in the data from a Snowflake table into a Snowpark DataFrame
data = session.table('FINANCEBI_DB.PANC."OB_NO_AUTH24_14D_V3_TR"')
data.show()


# Define the outer cross-validation loop
outer_cv = GridSearchCV(n_splits=5, shuffle=True, random_state=42)
# Define the inner cross-validation loop
inner_cv = GridSearchCV(n_splits=3, shuffle=True, random_state=42)

# Define the hyperparameter grid