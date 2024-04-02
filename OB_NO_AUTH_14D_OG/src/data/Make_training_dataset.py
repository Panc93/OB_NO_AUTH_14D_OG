from OB_NO_AUTH_14D_OG.src.features import Bivariate
from OB_NO_AUTH_14D_OG.src.features import Data_imputation
from OB_NO_AUTH_14D_OG.src.data.Univariate import univar_exec
from OB_NO_AUTH_14D_OG.src.data import Data_dictionary
from OB_NO_AUTH_14D_OG.src.features import Data_encoding
from OB_NO_AUTH_14D_OG.src.features import build_features
# cbsa_category
# deductible_equals_oop
Data_dictionary.data_dict(src=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\raw',
              wrk=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\interim',
              oup=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\raw',
              tnp='Temp', indsn= 'OB_NO_AUTH24_14D', limit=5)
univar_exec(
            src=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\raw',
            wrk=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\interim',
            oup=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\raw',
            tnp='Temp',
            indsn='OB_NO_AUTH24_14D',
            Train_start_dt='2021-07-01',
            Train_end_dt='2023-06-01',
            minmisslmt=10, l1misslmt=30, l2misslmt=50,
            maxmisslmt=75)
imputation_methods = {
        'bo_email_cnt': 'Zero',
        'bo_ibcalls': 'Zero',
        'bo_obcalls': 'Zero',
        'coinsurance': 'Median',
        'days_email_activity12d': 'Zero',
        'days_email_activity18d': 'Zero',
        'days_since_agreed_terms': 'Exclude_col',
        'days_since_last_call_12d': 'Zero',
        'days_since_last_call_18d': 'Zero',
        'days_since_last_email_12d': 'Zero',
        'days_since_last_email_18d': 'Zero',
        'days_since_last_ibemail_12d': 'Zero',
        'days_since_last_ibemail_18d': 'Zero',
        'deductible': 'Median',
        'ibabandon_queue_time_0': 'Zero',
        'ibabandon_queue_time_1': 'Zero',
        'ibabandon_queue_time_2': 'Zero',
        'ibabandon_queue_time_3': 'Zero',
        'ibabandon_queue_time_4': 'Zero',
        'ibabandon_queue_time_5': 'Zero',
        'ibagent_time_0': 'Median',
        'ibagent_time_1': 'Zero',
        'ibagent_time_2': 'Zero',
        'ibagent_time_3': 'Zero',
        'ibagent_time_4': 'Zero',
        'ibagent_time_5': 'Zero',
        'ibqueue_time_0': 'Median',
        'ibqueue_time_1': 'Zero',
        'ibqueue_time_2': 'Zero',
        'ibqueue_time_3': 'Zero',
        'ibqueue_time_4': 'Zero',
        'ibqueue_time_5': 'Zero',
        'moop': 'Median',
        'ob_emails_0': 'Exclude',
        'ob_emails_1': 'Exclude',
        'ob_emails_2': 'Exclude',
        'ob_emails_3': 'Exclude',
        'ob_emails_4': 'Exclude',
        'ob_emails_5': 'Exclude',
        'obagent_time_0': 'Zero',
        'obagent_time_1': 'Zero',
        'obagent_time_2': 'Zero',
        'obagent_time_3': 'Zero',
        'obagent_time_4': 'Zero',
        'obagent_time_5': 'Zero',
        'smart_cycles_allowed_by_plan': 'Median',
        'tot_ib_emails_18d': 'Zero',
        'tot_ibabandon_queue_time_12d': 'Zero',
        'tot_ibabandon_queue_time_18d': 'Zero',
        'tot_ibabandon_queue_time_6d': 'Zero',
        'tot_ibagent_time_12d': 'Zero',
        'tot_ibagent_time_18d': 'Zero',
        'tot_ibagent_time_6d': 'Zero',
        'tot_ibcalls_12d': 'Zero',
        'tot_ibcalls_18d': 'Zero',
        'tot_ibqueue_time_12d': 'Zero',
        'tot_ibqueue_time_18d': 'Zero',
        'tot_ibqueue_time_6d': 'Zero',
        'tot_ob_emails_12d': 'Zero',
        'tot_ob_emails_18d': 'Zero',
        'tot_ob_emails_6d': 'Zero',
        'tot_obagent_time_12d': 'Zero',
        'tot_obagent_time_18d': 'Zero',
        'tot_obagent_time_6d': 'Zero',
        'w_btwn_coverage_onboard': 'Median',
        'days_since_last_email_6d': 'Median',
        'days_since_last_ibemail_6d' : 'Median',
        'days_since_last_call_6d' : 'Median',
        'tot_ib_emails_12d' : 'Zero',
        'tot_ib_emails_6d' : 'Zero',
        'bo_ibabandoncalls' : 'Zero',
        'ibcalls_0' : 'Median',
        'ibabandon_calls_0' : 'Zero',
        'obcalls_0' : 'Zero',
        'ibcalls_1' : 'Zero',
        'ibabandon_calls_1' : 'Zero',
        'obcalls_1' : 'Zero',
        'ibcalls_2' : 'Zero',
        'ibabandon_calls_2' : 'Zero',
        'obcalls_2' : 'Zero',
        'ibcalls_3' : 'Zero',
        'ibabandon_calls_3' : 'Zero',
        'obcalls_3' : 'Zero',
        'ibcalls_4' : 'Zero',
        'ibabandon_calls_4' : 'Zero',
        'obcalls_4' : 'Zero',
        'ibcalls_5' : 'Zero',
        'ibabandon_calls_5' : 'Zero',
        'obcalls_5' : 'Zero',
        'tot_ibcalls_6d' : 'Median',
        'tot_ibabandon_calls_18d' : 'Median',
        'tot_ibabandon_calls_12d' : 'Median',
        'tot_ibabandon_calls_6d' : 'Median',
        'tot_obcalls_18d' : 'Median',
        'tot_obcalls_12d' : 'Median',
        'tot_obcalls_6d' : 'Median',
        'cbsa_category':'Mode',
        'deductible_equals_oop':'Mode',
        'cbsa_category': 'Mode',
        'deductible_equals_oop': 'Mode',

            }


Data_imputation.missing_value_treatment(
            src=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\raw',
            wrk=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
            indsn='OB_NO_AUTH24_14D',
            outdsn = 'OB_NO_AUTH24_14D_V1',
            imputation_methods=imputation_methods
            )

Data_imputation.Outliers(
        src=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
        external=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\external',
        indsn= 'OB_NO_AUTH24_14D_V1',
        Train_start_dt='2021-07-01',
        Train_end_dt='2023-06-01',
        fname='Outliers_treatment_variables',
        lower_quantile=0.003,
        upper_quantile=0.997)

Data_dictionary.data_dict(src=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
              wrk=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\interim',
              oup=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
              tnp='Temp', indsn= 'OB_NO_AUTH24_14D_V2', limit=3)
univar_exec(
            src=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
            wrk=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\interim',
            oup=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\processed',
            tnp='Temp',
            indsn='OB_NO_AUTH24_14D_V2',
            Train_start_dt='2021-07-01',
            Train_end_dt='2023-06-01',
            minmisslmt=10, l1misslmt=30, l2misslmt=50,
            maxmisslmt=75)

Bivariate.bivar_exec(src=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
           oup=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\processed',
           wrk=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\processed',
           indsn='OB_NO_AUTH24_14D_V2', resp='events', timelwlmt='2021-07-01',
           timeuplmt='2023-06-01')
#Drop the og columns -
# ethnicity
# how_did_you_hear_about_progyny
# previous_treatments
# treatments_interested_in
#cbsa_category
#deductible_equals_oop - null treatment



build_features.feature_selection(src =r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
                wrk=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\processed',
                oup=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
                p_val= 0.05,mi_score = 0 ,indsn='OB_NO_AUTH24_14D_V2',outdsn= 'OB_NO_AUTH24_14D_V3',
                bivar_file='OB_NO_AUTH24_14D_V2')
imputation_methods_2 = {

    'treatments_interested_in': 'Exclude_col',
    'ethnicity':'Exclude_col',
    'how_did_you_hear_about_progyny':'Exclude_col',
    'previous_treatments':'Exclude_col'
}

Data_imputation.missing_value_treatment(
            src=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
            wrk=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
            indsn='OB_NO_AUTH24_14D_V3',
            outdsn = 'OB_NO_AUTH24_14D_V4',
            imputation_methods=imputation_methods_2
            )
Data_dictionary.data_dict(src=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
              wrk=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\interim',
              oup=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
              tnp='Temp', indsn= 'OB_NO_AUTH24_14D_V4', limit=3)

Data_encoding.data_encoding_fit(src=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\interim',
                                oup=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\final',
                                indsn='OB_NO_AUTH24_14D_V4', timelwlmt='2021-07-01',timeuplmt='2023-06-01'
                                )

Data_dictionary.data_dict(src=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\final',
              wrk=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\interim',
              oup=r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\final',
              tnp='Temp', indsn= 'OB_NO_AUTH24_14D_V4_TR', limit=3)