import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost import XGBClassifier
from sklearn.metrics import roc_auc_score, classification_report
import shap
import joblib
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import cohen_kappa_score
import matplotlib.pyplot as plt
from shap import TreeExplainer
# Load the data
src =r'C:\Users\ParnikaPancholi\PycharmProjects\OB_NO_AUTH_14D\OB_NO_AUTH_14D_OG\data\final'
indsn = 'OB_NO_AUTH24_14D_V4_TR'
# hl = 'Ob_Noic_23_v2_hl'
data = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(indsn)))
print(data.shape)
# hold_out = pd.read_pickle(r"{}\{}.pkl".format(str(src), str(hl)))

# hold_out = hold_out.drop(columns=[col for col in hold_out.columns if col.startswith('current_pharmacy')])
drop_columns = ['mem_id_no_client', 'patient_id', 'onboard_date', 'target_date_12', 'events','target_date_6','target_date_18']

# hold_out['tot_previous_treatments'] = hold_out['previous_treatments_cp']+hold_out['previous_treatments_ivf']+hold_out['previous_treatments_tst']+hold_out['previous_treatments_ds'] + hold_out['previous_treatments_iui']+hold_out['previous_treatments_fet']+hold_out['previous_treatments_onc']+hold_out['previous_treatments_pgt']+hold_out['previous_treatments_ado']
patterns_to_match = ['current_pharmacy', '18d','mem_id_no_client', 'patient_id', 'onboard_date', 'target_date_12', 'events','target_date_6','target_date_18','_5','_4']

# Drop columns that start with any of the specified patterns
columns_to_drop = [col for col in data.columns if any(pattern in col for pattern in patterns_to_match)]

# Drop columns that start with any of the specified patterns
X = data.drop(columns_to_drop, axis=1)
y = data['events']
col =pd.DataFrame(X.columns)
# Define the outer cross-validation loop
outer_cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
# Define the inner cross-validation loop
inner_cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)

# Define the hyperparameter grid
param_grid ={
    'n_estimators': [300],
    'learning_rate': [0.05],
    'lambda': [ 0.5, 1,],
    'max_depth': [3, 4],
    'reg_alpha': [ 1.5],
    'subsample':[0.5,0.8],
    'reg_lambda': [ 1, 1.5],
    'gamma': [ 0.1, 0.5],
    'min_child_weight': [3, 5]
    }

#'n_estimators': [ 200,300, 500],
# 'learning_rate': [0.01, 0.05, 0.1],
# 'lambda': [0.1, 0.5, 1]

#'max_depth': [3, 4, 5],
#'reg_alpha': [0.1, 1, 1.5],
#'gamma': [0.1, 0.5],
#'subsample':[0.5,0.8,1]
# Define the Xgboost classifier param_grid = {
#     'n_estimators': [ 200, 500],
#     'max_depth': [3, 4, 5],
#     'learning_rate': [ 0.1],
#     'reg_alpha': [0.1, 1, 1.5],
#     'reg_lambda': [ 1, 1.5],
#     'gamma': [ 0.1, 0.5],
#     'min_child_weight': [1, 3, 5],
#  'n_estimators': [ 300],
#     'learning_rate': [ 0.05],
#     'lambda': [1],
# 'max_depth': [4],
#      'learning_rate':  [0.05],
#      'reg_alpha':  [1],
#    'lambda':  [1],
#     'gamma':  [0.1],
# 'min_child_weight': [ 5]
clf = XGBClassifier()

# Perform nested cross-validation

classification_rep = pd.DataFrame()
inner_cv_results_df = pd.DataFrame()
outer_train_scores = []
outer_test_scores = []
loop_index =[]
# Initialize lists to store ROC curve data for all iterations
all_fpr_train = []
all_tpr_train = []
all_fpr_test = []
all_tpr_test = []
outer_cv_outputs_df =pd.DataFrame()
for i, (train_index, test_index) in enumerate(outer_cv.split(X, y)):
    print(f"Fold {i}")
    print('Set difference of train and test indexes {}'.format(len(set(train_index)&set(test_index))))
    X_train, X_test = X.iloc[train_index, :], X.iloc[test_index, :]
    y_train, y_test = y.iloc[train_index], y.iloc[test_index]
    print("run train_test ")
    # Perform inner cross-validation to find the best hyperparameters
    grid_search = GridSearchCV(clf, param_grid=param_grid, cv=inner_cv, return_train_score=True,scoring='roc_auc')
    grid_search.fit(X_train, y_train)
    print("run fit ")
    # Get the best hyperparameters and score
    best_params_inner = grid_search.best_params_
    best_score_inner = grid_search.best_score_
    best_est = grid_search.best_estimator_
    inner_cv_results = grid_search.cv_results_
    # Convert the cv_results to a DataFrame
    inner_cv_results_df = inner_cv_results_df._append(pd.DataFrame(inner_cv_results), ignore_index=True)
    print("scores and params ")
    print(f"Fold {i} cv BEST SCORE SELECTED{best_score_inner}")
    print(f"Fold {i} cv BEST PARAMETER SELECTED {best_params_inner}")
    # Train the Xgboost classifier with the best hyperparameters
    clf.set_params(**best_params_inner)
    clf.fit(X_train, y_train)
    y_test_pred = clf.predict_proba(X_test)[:, 1]
    print("fit on X_train with best parameter ")
    # Evaluate the Xgboost classifier on the test set
    train_roc_score = roc_auc_score(y_train, clf.predict_proba(X_train)[:, 1])
    test_roc_score = roc_auc_score(y_test, y_test_pred)

    # Calculate the ROC curves for train and test sets
    fpr_train, tpr_train, thresholds_train = roc_curve(y_train, clf.predict_proba(X_train)[:, 1])
    fpr_test, tpr_test, thresholds_test = roc_curve(y_test, y_test_pred)
    roc_auc_train = auc(fpr_train, tpr_train)
    roc_auc_test = auc(fpr_test, tpr_test)
    all_fpr_train.append(fpr_train)
    all_tpr_train.append(tpr_train)
    all_fpr_test.append(fpr_test)
    all_tpr_test.append(tpr_test)
    # Plot the ROC curves
    plt.plot(fpr_train, tpr_train, label="Train ROC (AUC = %0.2f)" % roc_auc_train )
    plt.plot(fpr_test, tpr_test, label='Test ROC  (AUC = %0.2f)' % roc_auc_test)
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend()
    plt.show()
    outer_test_classif_report = classification_report(y_test, clf.predict(X_test),output_dict=True)
    outer_test_classif_df= pd.DataFrame(outer_test_classif_report).transpose()
    outer_test_classif_df.reset_index(inplace=True)
    # Append the best score and best hyperparameters to the lists
    # summarize the estimated performance of the model
    classification_rep =pd.concat([classification_rep,outer_test_classif_df])
    params = pd.DataFrame.from_dict(best_params_inner,orient='index').T
    # Create a DataFrame from the outer_train_scores, outer_test_scores, and best_scores_in lists
    t1 = pd.DataFrame(columns= ['Fold','Outer_train_score','Outer_Test_ROC','Best_validate_scores'])
    t1.loc[0,'Fold'] = i
    t1.loc[0,'Outer_train_score'] = train_roc_score
    t1.loc[0,'Outer_Test_ROC'] = test_roc_score
    t1.loc[0,'Best_validate_scores'] = best_score_inner
    t1=pd.concat([t1,params],axis=1)
    outer_cv_outputs_df= pd.concat([outer_cv_outputs_df, t1])

# Plot the ROC curves for all iterations
plt.figure(figsize=(8, 6))  # Adjust figure size if needed
for i in range(len(all_fpr_train)):
    plt.plot(all_fpr_train[i], all_tpr_train[i], label=f"Train ROC Fold {i}")
    plt.plot(all_fpr_test[i], all_tpr_test[i], label=f'Test ROC Fold {i}')

plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves for all Iterations')
plt.legend()
plt.show()
# _Na_la_lr
# Save the inner loop cv results to an Excel file
with pd.ExcelWriter('nested_cv_all_output2.xlsx', mode='w') as writer:
    inner_cv_results_df.to_excel(writer, sheet_name='Inner_roc_all', index=False)

# Save the outer loop cv results to sheet 2 of an Excel file
with pd.ExcelWriter('nested_cv_all_output2.xlsx', mode='a') as writer:
    outer_cv_outputs_df.to_excel(writer, sheet_name='Outer_roc_all', index=False)

with pd.ExcelWriter('nested_cv_all_output2.xlsx', mode='a') as writer:
    classification_rep.to_excel(writer, sheet_name='test_Classif_all', index=False)


# Calculate the ROC AUC scores for train and test sets
roc_auc_train = auc(fpr_train, tpr_train)
roc_auc_test = auc(fpr_test, tpr_test)

# Plot the ROC curves
plt.plot(fpr_train, tpr_train, label='Train ROC curve (AUC = %0.2f)' % roc_auc_train)
plt.plot(fpr_test, tpr_test, label='Test ROC curve (AUC = %0.2f)' % roc_auc_test)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()
from sklearn.metrics import precision_recall_curve
# calculate the no skill line as the proportion of the positive class
no_skill = len(y[y==1]) / len(y)
# plot the no skill precision-recall curve
pyplot.plot([0, 1], [no_skill, no_skill], linestyle='--', label='No Skill')
# calculate model precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, y_test_pred)
# plot the model precision-recall curve
pyplot.plot(recall, precision, marker='.', label='Logistic')
# axis labels
pyplot.xlabel('Recall')
pyplot.ylabel('Precision')
# show the legend
pyplot.legend()
pyplot.show()


# calculate the precision-recall auc
auc_score = auc(recall, precision)
print(auc_score)
# show the plot

#Building best model from best paramenters]

param_grid_f={
    'n_estimators':  300,
    'learning_rate': 0.05,
    'lambda': 0.5,
    'max_depth': 4,
    'learning_rate':  0.05,
    'reg_alpha':  1.5,
    'lambda':  1,
    'gamma':  0.1,
    'min_child_weight': 3,
    'subsample' : 0.8
}
clf_final = XGBClassifier()
clf_final.set_params(**param_grid_f)
print(clf_final)
clf_final.fit(X,y)
#y_hl_prob = clf_final.predict_proba(X_hl)[:, 1]
x_train_prob= clf_final.predict_proba(X)[:,1]
#y_hl_p = clf_best1.predict(X_hl)
#hl_score = roc_auc_score(y_hl,y_hl_prob)
train_score = roc_auc_score( y,x_train_prob)

print(train_score)
print(hl_score)
cohen_test= cohen_kappa_score(y_hl, y_hl_p)
print(cohen_test)
# Save the model
clf_final.save_model('Xgboost_14d_tr76_ts74_v0.json')

#reading the model
clf_best = XGBClassifier()
clf_best.load_model('Xgboost_roc_tr82_ts80_v3.json')
clf_best =clf_final
# Rename features in the data
X.rename(columns={"tot30_ib_calls": "total_inbound_calls", "ethnifity_f_3.0":"Ethinicity_missing"}, inplace=True)
# Create a SHAP explainer
explainer = shap.TreeExplainer(clf_best)
# Calculate SHAP values
shap_values = explainer.shap_values(X)
shap_values.shape
# Create a SHAP chart
#shap.summary_plot(shap_values, X, plot_type='bar',max_display=30)
# Calculate shap_values
#shap.summary_plot(shap_values[1], X_test, show=False)
shap.summary_plot(shap_values,X,max_display=30)
plt.title('SHAP Chart')
#plt.xlim(0,1)
plt.show()
#Model on features with just important features
print(shap_values)
# Get feature names
feature_names = X.columns

# Calculate average SHAP values
average_shap_values = shap_values.mean(axis=0)

# Create a DataFrame with feature names and average SHAP values
shap_df = pd.DataFrame({'Feature': feature_names, 'SHAP Value': average_shap_values})

# Sort DataFrame by absolute SHAP values for visualization
shap_df['Absolute SHAP Value'] = abs(shap_df['SHAP Value'])
shap_df = shap_df.sort_values(by='Absolute SHAP Value', ascending=False)

# Print or visualize the DataFrame
print(shap_df)
from sklearn.feature_selection import RFE

# Create an RFE estimator
rfe = RFE(estimator=clf_best, n_features_to_select=25)  # Select top 10 features
rfe.fit(X, y)
selected_features = X.columns[rfe.support_]
print("Selected features:", selected_features)
X_selected = X[selected_features]


clf_best_new = XGBClassifier() # Use selected features to train a new XGBoost model
clf_best_new.set_params(**param_grid)
clf_best_new.fit(X_selected, y)

#Model performance
X_hl_selected = X_hl[selected_features]
y_hl_prob = clf_best_new.predict_proba(X_hl_selected)[:, 1]
x_train_prob= clf_best_new.predict_proba(X_selected)[:,1]

hl_score = roc_auc_score(y_hl,y_hl_prob)
train_score = roc_auc_score( y,x_train_prob)
print(train_score)
print(hl_score)



# Calculate the ROC curve
fpr, tpr, thresholds = roc_curve(y_hl, y_hl_prob)

# Calculate the ROC AUC
roc_auc = auc(fpr, tpr)

# Plot the ROC curve
plt.plot(fpr, tpr, label='ROC curve (AUC = %0.2f)' % roc_auc)
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()


# Feat importance using permutation  improtance
for feat, importance in zip(X.columns,clf_best.feature_importances_):
    print('{}, {}'.format(feat,importance))
# Calculate permutation importance
feature_importances = permutation_importance(model, X, y, n_repeats=10)
# Print feature importance
print('Feature Importance:')
for feature, importance in zip(X.columns, feature_importances['mean_importance']):
    print(f'{feature}: {importance}'
