import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#import seaborn as sns

import xgboost as xgb
#from sklearn.ensemble import GradientBoostingClassifier,RandomForestClassifier  #,BaggingClassifier, AdaBoostClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_auc_score

from imblearn.over_sampling import SMOTE

import pickle



###### import dataset ######
df_raw = pd.read_excel('dataset_default_of_credit_card.xls')


###### Preprocess ######
df = df_raw.copy()

# change datatype
df = df.astype({'ID': 'str', 
           'SEX': 'category', 
           'EDUCATION': 'category',
           'PAY_0': 'category',
           'PAY_2': 'category',
           'PAY_3': 'category',
           'PAY_4': 'category',
           'PAY_5': 'category',
           'PAY_6': 'category',
           'default payment next month': 'category'
           })


###### exploratory data analysis ######
df.info()
des = df.describe()
corrmat = df.corr()
cov = df.cov()

pd.crosstab(df['SEX'], df['default payment next month'], margins = True)
pd.crosstab(df['MARRIAGE'], df['default payment next month'], margins = True)

# histogram of default payment next month
plt.hist(df['default payment next month'], bins = 50)
plt.xlabel('default payment next month')
plt.savefig('./data/default payment next month', dpi=200, bbox_inches='tight', pad_inches=1)
plt.show()

#plt.scatter(df['ActualAmt_USD'], df['ActualGross_USD'], c='blue', alpha=0.5)
#plt.xlabel("ActualAmt_USD")
#plt.ylabel("ActualGross_USD")
#plt.show()

X = df.iloc[:,1:-1]
y = df.iloc[:,-1]

# Save features (for production use)
features = X.columns.tolist()
file_features = open('./data/features.pickle', 'wb')

# load features
#pickle.dump(features, file_features)
#file_features.close()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100, stratify=y) # stratify=y 按原數據y中各類比例分層抽樣

# over sampling training set
method = SMOTE(random_state=100)
X_train_resampled, y_train_resampled = method.fit_sample(X_train, y_train)


###### set up XGBoost Pipeline & GridSearch ######
xgb_pipeline = Pipeline(steps=[("st_scaler", StandardScaler()), 
                               ("pca", PCA(n_components = 0.95)), #'mle' (自動找)
                               ("xgb_model",xgb.XGBClassifier(objective='binary:logistic', random_state = 100))])

param_grid = { 'xgb_model__n_estimators': [200, 300],
               'xgb_model__max_depth': [5],
               'xgb_model__subsample': [0.9],
               'xgb_model__colsample_bytree': [0.8, 0.9],
               'xgb_model__learning_rate': [0.05,0.2],
               #'xgb_model__scale_pos_weight': [scale_pos_weight],
               'xgb_model__gamma': [0.5]
               }  


grid_search_roc = GridSearchCV(estimator= xgb_pipeline, 
                                    param_grid= param_grid, 
                                    scoring='roc_auc',
                                    cv=10,
                                    refit=True,
                                    n_jobs = -1)   



grid_search_roc.fit(X_train_resampled, y_train_resampled)
print(grid_search_roc.best_params_)
print(grid_search_roc.best_score_)


# extract best model pipeline
xgbc_best = grid_search_roc.best_estimator_


###### prediction ######

Prob_train = xgbc_best.predict_proba(X_train)
Pred_train = xgbc_best.predict(X_train)

Prob = xgbc_best.predict_proba(X_test)
Pred = xgbc_best.predict(X_test)


###### model validation ######

Accuracy_train = xgbc_best.score(X_train, y_train)
print('Train set accuracy', Accuracy_train)
Accuracy = xgbc_best.score(X_test, y_test)
print('Test set accuracy', Accuracy)


roc_auc_train = roc_auc_score(y_train, Prob_train[:,1])
print('Train set ROC AUC score: {:.4f}'.format(roc_auc_train))
roc_auc = roc_auc_score(y_test, Prob[:,1])
print('Test set ROC AUC score: {:.4f}'.format(roc_auc))


###### feature importance ######
### build-in plot ### 
#for PCA included pipeline
xgb.plot_importance(grid_search_roc.best_estimator_.named_steps["xgb_model"], importance_type='weight')
#plt.savefig('./data/ft_importance_20200701.png', dpi=200, bbox_inches='tight', pad_inches=1)
plt.show() 


##Save model/pipeline###
with open('./data/xgb_pipeline_v1.0', 'wb') as f: 
    pickle.dump(xgbc_best, f)




### manually plot ###
#for PCA excluded pipeline
#ft_importance = grid_search_roc.best_estimator_.named_steps["xgb_model"].feature_importances_
#cols = X_train.columns.tolist()
#df_ftimportance = pd.DataFrame(list(zip(cols, ft_importance)), columns=["feature","ft_importance"])
#df_ftimportance = df_ftimportance.sort_values('ft_importance', ascending=False)
#df_ftimportance.head(30)
#
## Set plotting style
#sns.set_style('whitegrid')
#g = sns.barplot(y='feature', x='ft_importance', data=df_ftimportance, palette="Blues_r")
#
#
## Annotate every single Bar with its value, based on it's width           
#for p in g.patches:
#    width = p.get_width()
#    plt.text(0.02+p.get_width(), p.get_y()+0.55*p.get_height(),
#             '{:1.2f}'.format(width),
#             ha='center', va='center', size = 8)



## make strategy table for different thresholds
#Prob_copy = pd.DataFrame(Prob.copy())
## Set all the acceptance rates to test 
#accept_rates = [1.0, 0.95, 0.9, 0.85, 0.8, 0.75, 0.7, 0.65, 0.6, 0.55, 
#               0.5, 0.45, 0.4, 0.35, 0.3, 0.25, 0.2, 0.15, 0.1, 0.05] 
## Create lists to store thresholds and bad rates 
#thresholds = [] 
#AvgRiskAmts = []
#n_accepted_loans = []
#bad_rates = [] 
#for rate in accept_rates: 
#    
#    # Calculate threshold 
#    threshold = np.quantile(Prob[:,1], rate).round(3) 
#    
#    # Store threshold value in a list 
#    thresholds.append(threshold)
#    
#    # Apply the threshold to reassign loan_status 
#    Prob_copy['pred_riskamt_status'] = Prob_copy[1].apply(lambda x: 1 if x > threshold else 0)
#    Prob_copy['actual_riskamt_status'] = y_test.values
#    Prob_copy['RiskAmt_lastwk'] = X_test['AR_RiskAmt_lastwk'].values
#    
#    # AvgRiskAmt those who are accepted 
#    AvgRiskAmt =  np.mean(Prob_copy[Prob_copy[1] < np.quantile(Prob_copy[1], rate)]['RiskAmt_lastwk'])
#    AvgRiskAmts.append(AvgRiskAmt)
#    # n_accepted_loans
#    n_accepted_loan = len(Prob_copy[Prob_copy[1] < np.quantile(Prob_copy[1], rate)])
#    n_accepted_loans.append(n_accepted_loan)
#    
#    # Create accepted loans set of predicted non-defaults 
#    accepted_loans = Prob_copy[Prob_copy['pred_riskamt_status'] == 0] 
#    
#    # Calculate and store bad rate 
#    bad_rate = (np.sum(accepted_loans['actual_riskamt_status']) / accepted_loans['actual_riskamt_status'].count()).round(3)
#    bad_rates.append(bad_rate)
#    
#df_strategy = pd.DataFrame(zip(accept_rates, thresholds, bad_rates, n_accepted_loans, AvgRiskAmts), 
#                       columns = ['Acceptance_Rate','Threshold','Bad_Rate', 'Num_Accepted', 'Avg_RiskAmt_lastwk']) 
#
##df_strategy['Avg_RiskAmt_lastwk'] = np.mean(X_test['AR_RiskAmt_lastwk'])
#df_strategy['Estimated_Value'] = \
#    df_strategy['Num_Accepted'] * (1- df_strategy['Bad_Rate']) * df_strategy['Avg_RiskAmt_lastwk']\
#    - df_strategy['Num_Accepted'] * df_strategy['Bad_Rate'] * df_strategy['Avg_RiskAmt_lastwk']









