import pandas as pd


def missing_value_treatment(src="", indsn="", wrk="", imputation_methods=" ", outdsn=""):
    """
    Perform missing value treatment on a indsn.
    Args:
        :param indsn: Input input_dfFrame with missing values.
         :param imputation_methods: Dictionary of variables as keys and imputation methods as values.
                                   Available methods: 'mean', 'median', 'mode', 'ffill', 'bfill', 'custom'
                                   :type src: object
        :param src:
    Returns:
        pd.input_dfFrame: input_dfFrame with missing values treated.
    """
    input_df = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))
    print(r"{} successfully read.".format(str(indsn)))
    missing_columns = input_df.columns[input_df.isnull().any()].tolist()

    # Drop columns with 99% or more missing values

    # print(f"Dropping {0} as missing value is =100%".format(columns_to_drop))
    for col in missing_columns:
        print(col)
        if col in imputation_methods:
            method = imputation_methods[col]
            if method == 'Mean':
                input_df[col].fillna(input_df[col].mean(), inplace=True)
            elif method == 'Median':
                input_df[col].fillna(input_df[col].median(), inplace=True)
            elif method == 'Mode':
                input_df[col].fillna(input_df[col].mode()[0], inplace=True)
            elif method == 'Zero':
                input_df[col].fillna(0, inplace=True)
            elif method == 'Exclude':
                print("Dropping x rows null in column {0}".format(col))
                print("Shape of old dataframe {0}".format(input_df.shape))
                input_df.dropna(subset=[col], inplace=True)
                print("Shape of new dataframe {0}".format(input_df.shape))
            elif method == 'Exclude_col':
                print("Dropping column {0}".format(col))
                # missing_threshold = 99
                # columns_to_drop = [col for col in missing_columns if input_df[col].isnull().mean() >= missing_threshold]
                input_df.drop(col, axis=1, inplace=True)
            elif method == 'Value':
                print("Dropping column {0}".format(col))
                # missing_threshold = 99
                # columns_to_drop = [col for col in missing_columns if input_df[col].isnull().mean() >= missing_threshold]
                input_df.drop(col, axis=1, inplace=True)
            else:
                print(f"Invalid imputation method specified for column '{col}'. Skipping imputation.")
        else:
            print(f"No imputation method specified for column '{col}'. Skipping imputation.")
    try:
        print("enter try")
        input_df.to_pickle(r"{}\{}.pkl".format(str(wrk), str(outdsn)))

        print("Following output files successfully saved in Interim folder")
        print("{}".format(str(outdsn)))

        return True
    except:
        print("No output files")
        return False


def Outliers(src="", indsn="", external="", fname="", Train_start_dt="", Train_end_dt="", lower_quantile="",
             upper_quantile=""):
    input_df = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))
    print(r"{} successfully read.".format(str(indsn)))
    outlier_vars = pd.read_excel(r"{}\{}.xlsx".format(str(external), str(fname)))
    print(r"{0} successfully read. There are {1} variables for outlier treatment".format(str(external),
                                                                                         len(outlier_vars)))
    input_df_tr = input_df.loc[
        (input_df['target_date_12'] >= "'" + Train_start_dt + "''") & (
                    input_df['target_date_12'] <= "'" + Train_end_dt + "''")]
    for col in outlier_vars['VAR_NAME']:
        print(col)
        percentile_value_lower = input_df_tr[col].quantile(lower_quantile)
        # Calculate the 95th percentile value
        percentile_value_upper = input_df_tr[col].quantile(upper_quantile)
        print(percentile_value_lower, percentile_value_upper)
        # Cap values lower than the 5th percentile
        input_df[col] = input_df[col].clip(lower=percentile_value_lower)
        # Cap values greater than the 95th percentile
        input_df[col] = input_df[col].clip(upper=percentile_value_upper)

    # Save the input table
    input_df.to_pickle(r"{0}\{1}_V2.pkl".format(str(src), str(indsn[0:16])))
    return True


# imputation_methods_2 = {
#     'cbsa_category': 'Mode',
#     'deductible_equals_oop': 'Mode',
#     'treatments_interested_in': 'Exclude_col',
#     'ethnicity':'Exclude_col',
#     'how_did_you_hear_about_progyny':'Exclude_col',
#     'previous_treatments':'Exclude_col'
# }
# missing_value_treatment(
#             src=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
#             wrk=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
#             indsn='OB_NO_AUTH24_14D_V2',
#             outdsn = 'OB_NO_AUTH24_14D_V3',
#             imputation_methods=imputation_methods_2
#             )
