import shap  # used to calculate Shap values

from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import matplotlib.pyplot as plt

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


df.info()

X = df.iloc[:,1:-1]
X = pd.get_dummies(X)

y = df.iloc[:,-1]



X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=100, stratify=y)

# transform data type to DMatrix
d_train = xgb.DMatrix(X_train, y_train)
d_test = xgb.DMatrix(X_test, y_test)


# modeling
watchlist = [(d_train, "train"), (d_test, "test")]
params = {'n_estimators': 500, 'max_depth': 3}
model = xgb.train(params, d_train, num_boost_round=2000, evals=watchlist, 
                      early_stopping_rounds=100, verbose_eval=10)




### Calculate SHAP Values ###

# Create object that can calculate shap values
explainer = shap.TreeExplainer(model)

# Calculate Shap values
shap_values = explainer.shap_values(X_train)

# global interpretability
shap.summary_plot(shap_values, X_train)

# Local interpretability for single data point
#shap.initjs()
shap.force_plot(explainer.expected_value, shap_values[10,:], X_train.iloc[10,:], matplotlib=True)

