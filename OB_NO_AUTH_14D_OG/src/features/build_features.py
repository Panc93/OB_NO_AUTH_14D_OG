#Removing features with zero mutual information and p value <0.05

def feature_selection(src='', oup='', wrk='',p_val='', mi_score='', indsn='', outdsn='', bivar_file=''):
    # Output of this module is data set with removed column and list of columns dropped
#Read the files
    import pandas as pd

    input_df =pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))
    mi_test= pd.read_pickle(r"{}\{}_mi_scores.pkl".format(str(wrk), str(bivar_file)))
    bi_chisq = pd.read_pickle(r"{}\{}_bi_chisq.pkl".format(str(wrk), str(bivar_file)))
    mi_score_exclusion= mi_test[mi_test['Mutual Information Score']== mi_score]
    chisq_exclusion= bi_chisq[bi_chisq['P-VALUE']>p_val].loc[:,['VAR_NAME','P-VALUE']]
    # =mi_test.append(bi_chisq, axis=1, ignore_index=True)

    mi_score_exclusion.rename(columns={'Variable':'VAR_NAME', 'Mutual Information Score':'Test_statistic'}, inplace=True)
    mi_score_exclusion['Test']= 'MI'

    chisq_exclusion.rename(columns={'P-VALUE':'Test_statistic'}, inplace=True)
    chisq_exclusion['Test']= 'CHI_SQ'
    Exclusion_list= pd.concat([chisq_exclusion, mi_score_exclusion], ignore_index=True)
    #Creating new feat :
    input_df['tot_previous_treatments'] = input_df['previous_treatments_cp'] + input_df['previous_treatments_ivf'] + input_df[
        'previous_treatments_tst'] + input_df['previous_treatments_ds'] + input_df['previous_treatments_iui'] + input_df[
                                          'previous_treatments_fet'] + input_df['previous_treatments_onc'] + input_df[
                                          'previous_treatments_pgt'] + input_df['previous_treatments_ado']
    input_df['tot_treatments_interested'] = input_df['treatments_interested_adoption'] + input_df['treatments_interested_donor_surrogacy'] + \
                                          input_df['treatments_interested_fet'] + input_df['treatments_interested_iui'] + \
                                          input_df['previous_treatments_iui'] + input_df['treatments_interested_ivf'] + input_df['treatments_interested_pgt'] + \
                                          input_df['treatments_interested_testing']
    print("total_excluded columns", len(set(Exclusion_list['VAR_NAME'])))
    print('Shape of input df:', input_df.shape)
    columns_drop = Exclusion_list['VAR_NAME'].to_list()
    new_df= input_df.drop(columns=columns_drop, axis=1).copy()
    print('Shape after exclusion:', new_df.shape)
    print("check the excluded vars")
    print("Saving excluded column list in Exclusion_list.xlsx")
    Exclusion_list.to_excel(r"{}\{}_Exclusion_list.xlsx".format(str(src), str(indsn)))
    print("Saving the final dataset in Final folder")
    new_df.to_pickle(r"{}\{}.pkl".format(str(oup), str(outdsn)))
    return True




