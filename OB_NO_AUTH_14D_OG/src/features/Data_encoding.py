# -*- coding: utf-8 -*-
"""
Created on Wed Oct 25 14:51:10 2023

@author: ParnikaPancholi
"""


def data_encoding_fit(src="", indsn="", timelwlmt="", timeuplmt="",oup=""):
    """
    :param src: Path to input folder
    :param timelwlmt : training start
    :timeuplmt :training end
    :param indsn: Name of the Input Data frame
    :return: Dataframe with encoded variables in pickle file
    """
    from joblib import dump
    import pandas as pd
    from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
    import sys
    sys.path.insert(1, r'..\.')

    # Check if all the input parameters are properly specified
    if src == "" or indsn == "" or timelwlmt == "" or timeuplmt == "":
        print("INPUT PARAMETERS ARE NOT SPECIFIED PROPERLY IN FUNCTION CALL!")
        return False
    # Read files
    try:
        # read input dataframe
        clm_dtls = pd.read_excel(r"{}\{}_clmn_dtls.xlsx".format(str(src), str(indsn)))
        # read input dataframe

        input_df = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))

        print("Required Dataframes successfully read.")
        print("---------------------------------------------------")
        print("")
    except:
        print("Unable to read required Dataframes!")
        return False

    # Drop eligibility status
    # For categorical and ordinal
    input_df['target_date_12'] = pd.to_datetime(input_df['target_date_12'])

    # Separate out holdout and train data
    input_df_train = input_df.loc[
        (input_df['target_date_12'] >= "'" + timelwlmt + "''") & (input_df['target_date_12'] <= "'" + timeuplmt + "''")]

    # preparing list of variables for which bivariate is required
    # All_char_list = clm_dtls[clm_dtls.BIVAR_TYPE.isin(["CHAR","CHAR - OTHER"])& clm_dtls.DEF_TYPE.isin(["CHAR","NUM"] )]['VAR_NAME']
    Char_list = clm_dtls[clm_dtls.BIVAR_TYPE.isin(["NOMINAL"]) & clm_dtls.DEF_TYPE.isin(['CHAR'])]['VAR_NAME']
    # len(All_char_list
    len(Char_list)
    ohe = OneHotEncoder(sparse_output=False)
    print("run")
    # apply le on categorical feature columns
    ohe_encoded = ohe.fit_transform(input_df_train[Char_list])
    transformed_df = pd.DataFrame(ohe_encoded, columns=ohe.get_feature_names_out())

    # One-hot encode nominal variables

    one_hot_encoded_df = pd.DataFrame(ohe_encoded, index=input_df_train.index,
                                      columns=ohe.get_feature_names_out())

    # Extract only the columns that didnt need to be encoded
    data_other_cols = input_df_train.drop(columns=Char_list)

    # Concatenate the two dataframes :
    data_out = pd.concat([data_other_cols, one_hot_encoded_df], axis=1)
    # Save the label encoder and one-hot encoder to pickle files for holdout encodings in future
    try:
        # Save the one-hot encoded dataframe
        data_out.to_pickle(r"{}\{}_TR.pkl".format(str(oup), str(indsn)))

        print("Input data has {} variables".format(str(input_df.shape)))
        print("Input train data has {} variables".format(str(input_df_train.shape)))
        print("Output data has {} variables".format(str(data_out.shape)))

        # Save the label encoder and one-hot encoder to pickle files for holdout encodings in future
        dump(ohe, r"{}\{}_one_hot_encoder.joblib".format((str(oup)), str(indsn)))
        print("Files saved")
        return True
    except:
        print("Files not saved. Check paths")
        return False


def data_encoding_transform(src="", oup="", wrk="", indsn="", timelwlmt="", timeuplmt=""):
    """
    :param src: Path to input folder
    :param wrk: Path to Work folder
    :param timelwlmt : training start
    :timeuplmt :training end
    :param indsn: Name of the Input Data frame
    :return: Dataframe with encoded variables in pickle file, label and ordinal encoders
    """
    import joblib
    import pandas as pd
    import sys
    sys.path.insert(1, r'..\.')

    # Check if all the input parameters are properly specified
    if src == "" or wrk == "" or oup == "" or indsn == "" or timelwlmt == "" or timeuplmt == "":
        print("INPUT PARAMETERS ARE NOT SPECIFIED PROPERLY IN FUNCTION CALL!")
        return False
    # Read files
    try:
        # read input dataframe
        clm_dtls = pd.read_excel(r"{}\{}_clmn_dtls.xlsx".format(str(src), str(indsn[:16])))
        input_df = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))
        # read input dataframe
        oe_file_path = r"{}\{}_v3_label_encoder.joblib".format((str(src)), str(indsn[:16]))
        ohe_file_path = r"{}\{}_v3_one_hot_encoder.joblib".format((str(src)), str(indsn[:16]))
        print(oe_file_path)
        oe = joblib.load(oe_file_path)
        ohe = joblib.load(ohe_file_path)

        print("Required Dataframes successfully read.")
        print("---------------------------------------------------")
        print("")
    except FileNotFoundError:
        print('File not found ')
        return False

    # Drop eligibility status
    # For categorical and ordinal
    input_df['target_date_12'] = pd.to_datetime(input_df['target_date_12'])

    # Separate out holdout and train data
    input_df_hl = input_df.loc[
        (input_df['target_date_12'] > "'" + timelwlmt + "''") & (input_df['target_date_12'] <= "'" + timeuplmt + "''")]

    char_list = clm_dtls[clm_dtls.BIVAR_TYPE.isin(["ORDINAL"]) & clm_dtls.DEF_TYPE.isin(["NOMINAL"])]['VAR_NAME']
    # len(All_char_list)
    print(len(char_list))
    # apply le on categorical feature columns
    oe_encoded = oe.transform(input_df_hl[char_list])
    encoded_df = pd.DataFrame(oe_encoded, columns=char_list)
    # One-hot encode nominal variables
    one_hot_encoded_df = pd.DataFrame(ohe.transform(encoded_df), index=input_df_hl.index,
                                      columns=ohe.get_feature_names_out())

    # Extract only the columns that didnt need to be encoded
    data_other_cols = input_df_hl.drop(columns=char_list)

    # Concatenate the two dataframes :
    data_out = pd.concat([data_other_cols, one_hot_encoded_df], axis=1)
    try:
        # Save the one-hot encoded dataframe
        data_out.to_pickle(r"{}\{}_hl.pkl".format(str(src), str(indsn)))
        print("Input data has {} variables".format(str(input_df.shape)))
        print("HL data has {} variables".format(str(input_df_hl.shape)))
        print("Final HL data (encoded Output) has {} variables".format(str(data_out.shape)))
        print("Files saved")
        return True
    except:
        print("Files not saved. Check paths")
        return False

