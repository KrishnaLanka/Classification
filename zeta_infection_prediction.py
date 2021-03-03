#!/usr/bin/env python
# coding: utf-8

# In[1]:
get_ipython().system('pip install imbalanced-learn')
# In[2]:
get_ipython().system('pip install xgboost')
# In[182]:

# Import libraries

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import RepeatedStratifiedKFold

from sklearn.metrics import confusion_matrix

from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import recall_score

pd.set_option("display.float", "{:.2f}".format)
pd.options.display.max_columns = 999
pd.options.display.max_rows = 999

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)


# In[183]:
# Data

df = pd.read_csv("Data.csv")
pred_data = pd.read_csv("Test.csv")

# In[184]:
df.shape, pred_data.shape

# In[185]:
df = pd.concat([df, pred_data], ignore_index = True)

# In[186]:
df.shape

# In[188]:

#Summary statistics
df.describe()

# In[189]:
# Nulls in the data set
df.isnull().sum(axis=0)

# In[190]:
data = df.copy()

# In[191]:
# Discarding the records from the final prediction data set.
df = df[0:800].reset_index(drop=True)

# In[192]:
# Checking for imbalance in the data with respect to the target variable
sns.countplot(x="zeta_disease", data=df)

# In[193]:
# Create labels for age and weight

data['age_label'] = np.select([data['age'] < 20,
                               data['age'].between(20,30),
                               data['age'].between(30,40),
                               data['age'].between(40,50),
                               data['age'] >= 50], [1,2,3,4,5])

data['weight_label'] = np.select([data['weight'] < 100,
                                  data['weight'].between(100,150),
                                  data['weight'].between(150,200),
                                  data['weight'].between(200,250),
                                  data['weight'] >= 250], [1,2,3,4,5])

# In[194]:
# Create a label for smoking habits as well

data['smoking'] = np.where(data['years_smoking'] == 0,0,1)

data['smoking_label'] = np.select([data['years_smoking'] == 0,
                                   data['years_smoking'].between(1,3),
                                   data['years_smoking'].between(3,10),
                                   data['years_smoking'].between(10,20),
                                   data['years_smoking'] >= 20], [1,2,3,4,5])

# In[195]:
# Count the number of records per each age and weight group
data['age_weight_class_size'] = data.groupby(by=['age_label','weight_label'])['zeta_disease'].transform('size')

# In[196]:
# Discarding the records from the final prediction data set.
df = data[0:800].reset_index(drop=True)

# In[197]:
sns.catplot(x="age_label", y="age_weight_class_size", hue="weight_label", col="zeta_disease", data=df)

# In[198]:
sns.catplot(x="age_label", y="age_weight_class_size", hue="weight_label", col="zeta_disease", row="smoking", data=df)

# In[199]:
sns.catplot(x="zeta_disease", hue="age_label", col="smoking",
            data=df, kind="count",
            height=4, aspect=.7);

# In[200]:
sns.catplot(x="zeta_disease", hue="age_label", col="smoking", row="weight_label",
            data=df, kind="count",
            height=4, aspect=.7);

# In[201]:
target_column = ['zeta_disease']
cat_columns = ['age_label', 'weight_label', 'smoking_label', 'smoking']

quant_columns = []
for column in df.columns:
    if column not in cat_columns and column not in target_column:
        quant_columns.append(column)

# In[202]:
plt.figure(figsize=(15, 10))

for i, column in enumerate(cat_columns[0:3], 1):
    plt.subplot(3, 3, i)
    df[df["zeta_disease"] == 0][column].hist(bins=50, color='green', label='No Zeta disease', alpha=0.75)
    df[df["zeta_disease"] == 1][column].hist(bins=50, color='red', label='Zeta disease', alpha=0.75)
    plt.legend()
    plt.xlabel(column)

# In[203]:
#### 
#   1. People in age group 30 to 50 (age labels 3 & 4) are more prone to zeta disease.
#   2. People in weight group 200 to 250 pounds are at higher risk of Zeta disease (weight label 4).
#   3. People with smoking habits for 3 or more years are more risk than others.
####

# In[204]:
plt.figure(figsize=(15, 10))

for i, column in enumerate(quant_columns, 1):
    plt.subplot(3, 3, i)
    df[df["zeta_disease"] == 0][column].hist(bins=50, color='green', label='No Zeta disease', alpha=0.75)
    df[df["zeta_disease"] == 1][column].hist(bins=50, color='red', label='Zeta disease', alpha=0.75)
    plt.legend()
    plt.xlabel(column)

# In[205]:
#### 
#   1. For Age, Weight and Smoking habits, earlier interpretation stands the same.
#   2. Liver stress test of above 1 (mostly 1-1.5 and above 2) is of concern and the risk of disease is high.
#   3. No similar conclusions can be made with blood pressure. People with blood pressure as high as 125 and greater have not mant zeta disease cases.
####

# In[206]:
plt.figure(figsize=(10, 7))

# Scatter plot with zeta disease cases and no zeta disease cases
plt.scatter(df.age[df.zeta_disease==1], df.weight[df.zeta_disease==1], color ="red")
plt.scatter(df.age[df.zeta_disease==0], df.weight[df.zeta_disease==0], color="green")

# Plot labels
plt.title("Zeta disease with respect to age and weight of the person")
plt.xlabel("Age of the person")
plt.ylabel("Weight of the person")
plt.legend(["Zeta disease", "No Zeta disease"]);

# In[207]:
plt.figure(figsize=(10, 7))

# Scatter plot with zeta disease cases and no zeta disease cases
plt.scatter(df.age[df.zeta_disease==1], df.years_smoking[df.zeta_disease==1], color ="red")
plt.scatter(df.age[df.zeta_disease==0], df.years_smoking[df.zeta_disease==0], color="green")

# Plot labels
plt.title("Zeta disease with respect to age and smoking habits of the person")
plt.xlabel("Age of the person")
plt.ylabel("Years of smoking by the person")
plt.legend(["Zeta disease", "No Zeta disease"]);


# In[208]:
####
#   1. The above plot gives away some of the issues in the data. How can a person of age 19 have smoking habits for 22 and 38 years? Clearly, Incorrect entries.
#   2. Same issue with people between 40 and 45 years of age.
####

df[(df['age'] <= 20)
       & (df['years_smoking'] >= 20)].reset_index(drop=True)

# In[209]:
df[(df['age'].isin([40,41,42,43,44,45]))
       & (df['years_smoking'] >= 35)].reset_index(drop=True)

# In[210]:
data['age_when_smoking_started'] = np.subtract(data['age'],data['years_smoking'])

# In[211]:
print(len(data[data['age_when_smoking_started'] <= 15]),
      len(data[data['age_when_smoking_started'] <= 10]),
      len(data[data['age_when_smoking_started'] <= 5]))

# In[212]:
####
#   In an ideal situation, smoking started at an age of 3 should be treated as an incorrect entry.
#   For this analysis, we are not disturbing the positive data points ('age_when_smoking_started') and we will impute the negative data points with their group means.
####

# In[213]:
age_list = data.age_label.unique().tolist()

for i in age_list:
    X = data[data['age_label'] == i].years_smoking.mean()
    data['years_smoking'] = np.where(data['age_when_smoking_started'] > 0, data['years_smoking'], round(X,0))
    data['years_smoking'] = data['years_smoking'].astype(int)

# In[214]:
# Discarding the records from the final prediction data set.
df = data[0:800].reset_index(drop=True)

# In[215]:
plt.figure(figsize=(10, 7))

# Scatter plot with zeta disease cases and no zeta disease cases
plt.scatter(df.weight[df.zeta_disease==1], df.years_smoking[df.zeta_disease==1], color ="red")
plt.scatter(df.weight[df.zeta_disease==0], df.years_smoking[df.zeta_disease==0], color="green")

# Plot labels
plt.title("Zeta disease with respect to weight and smoking habits of the person")
plt.xlabel("weight of the person")
plt.ylabel("Years of smoking by the person")
plt.legend(["Zeta disease", "No Zeta disease"]);

# In[216]:
####
#  1. As we can see from the above plot, people weighing over 200 lbs have smoking habits for 20 years and more.
#  2. Lot of non-smokers weigh less than 200 lbs.
####

# In[217]:
#### Correlation matrix
correlation_matrix = df.corr()

fig, ax = plt.subplots(figsize=(12, 10))
ax = sns.heatmap(correlation_matrix ,annot=True,linewidths=0.5,fmt=".2f");

# In[218]:

df[df.columns[0:]].corr()['zeta_disease'][:].sort_values()

# In[219]:
for i in cat_columns:
    df[i] = df[i].astype('object')
    
df.dtypes

# In[235]:
df_actual_set = data[0:800].reset_index(drop=True)
df_pred_set = data[800:820].reset_index(drop=True)

# In[236]:
X = df_actual_set.iloc[:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14]]
Y = df_actual_set.iloc[:,8]

# In[237]:
X_pred = df_pred_set.iloc[:,[0,1,2,3,4,5,6,7,9,10,11,12,13,14]]

# In[238]:
# Create dummy variables for categorical data

X = pd.get_dummies(X)
X_pred = pd.get_dummies(X_pred)

# In[239]:
#### Train Test split
from sklearn.model_selection import train_test_split
X_train,X_test,Y_train,Y_test = train_test_split(X,Y,test_size=0.25,random_state=0)

# In[240]:
#### Scaling using mean and std
scaler_ = StandardScaler()
X_train = scaler_.fit_transform(X_train)
X_test = scaler_.fit_transform(X_test)
X_pred = scaler_.fit_transform(X_pred)

# In[241]:
# LOGISTIC REGRESSION
# In[242]:
classifier_LR = LogisticRegression(C=1.0, class_weight="balanced", dual=False, fit_intercept=True,
                                   intercept_scaling=1, l1_ratio=None, max_iter=1000,
                                   multi_class='auto', n_jobs=None, penalty='l1',
                                   random_state=0, solver='saga', tol=0.001, verbose=0,
                                   warm_start=False)

# In[250]:
classifier_LR.fit(X_train, Y_train)

# In[251]:
Y_pred_LR = classifier_LR.predict(X_test)

# In[252]:
# Confusion matrix
matrix_LR = confusion_matrix(Y_test, Y_pred_LR)
matrix_LR

# In[253]:
# Print accuracy , precision and recall
print(balanced_accuracy_score(Y_test, Y_pred_LR), 
      average_precision_score(Y_test, Y_pred_LR), 
      recall_score(Y_test, Y_pred_LR))

# In[256]:
# RANDOM FOREST CLASSIFIER
# In[257]:
classifier_RF = RandomForestClassifier(bootstrap=True, class_weight='balanced',
                       criterion='entropy', max_depth=None, max_features='auto',
                       max_leaf_nodes=None, min_impurity_decrease=0.0,
                       min_impurity_split=None, min_samples_leaf=1,
                       min_samples_split=2, min_weight_fraction_leaf=0.0,
                       n_estimators=10, n_jobs=None, oob_score=False,
                       random_state=1, verbose=0, warm_start=False)


# In[258]:
classifier_RF.fit(X_train, Y_train)


# In[259]:
Y_pred_RF = classifier_RF.predict(X_test)


# In[260]:
matrix_RF = confusion_matrix(Y_test, Y_pred_RF)
matrix_RF


# In[261]:
print(balanced_accuracy_score(Y_test, Y_pred_RF), 
      average_precision_score(Y_test, Y_pred_RF), 
      recall_score(Y_test, Y_pred_RF))


# In[264]:
# IMBALANCED ENSEMBLE -- RANDOM FOREST CLASSIFIER
# In[265]:


from imblearn.ensemble import BalancedRandomForestClassifier

classification_balanced_RF = BalancedRandomForestClassifier(bootstrap=True, class_weight='balanced',
                                                            criterion='entropy', max_depth=None, max_features='auto',
                                                            max_leaf_nodes=None, min_impurity_decrease=0.0,
                                                            min_samples_leaf=2, min_samples_split=2,
                                                            min_weight_fraction_leaf=0.0, n_estimators=10, n_jobs=1,
                                                            oob_score=False, random_state=1, replacement=False,
                                                            sampling_strategy='auto', verbose=0, warm_start=False)


# In[266]:


classification_balanced_RF.fit(X_train, Y_train)


# In[267]:


Y_pred_IBRF = classification_balanced_RF.predict(X_test)


# In[268]:


# Balanced accuracy, Precision and Recall

print(balanced_accuracy_score(Y_test, Y_pred_IBRF), 
      average_precision_score(Y_test, Y_pred_IBRF), 
      recall_score(Y_test, Y_pred_IBRF))


# In[269]:


# Confusion matrix

matrix_BRF = confusion_matrix(Y_test, Y_pred_IBRF)
matrix_BRF


# In[58]:


# SMOTE technique


# In[272]:


smote = SMOTE(random_state=1, sampling_strategy=1.0)
X_train_smote, Y_train_smote = smote.fit_resample(X_train, Y_train)
Counter(Y_train_smote)


# In[273]:


# LOGISTIC REGRESSION WITH SMOTE


# In[274]:


classifier_LR_smote = LogisticRegression(C=1.0, class_weight="balanced", dual=False, fit_intercept=True,
                                         intercept_scaling=1, l1_ratio=None, max_iter=1000,
                                         multi_class='auto', n_jobs=None, penalty='l1',
                                         random_state=1, solver='saga', tol=0.001, verbose=0,
                                         warm_start=False)


# In[275]:


classifier_LR_smote.fit(X_train_smote, Y_train_smote)


# In[276]:


Y_pred_LR_smote = classifier_LR_smote.predict(X_test)


# In[277]:


# Confusion matrix

matrix_LR_smote = confusion_matrix(Y_test, Y_pred_LR_smote)
matrix_LR_smote


# In[278]:


# Balanced accuracy, Precision and Recall

print(balanced_accuracy_score(Y_test, Y_pred_LR_smote), 
      average_precision_score(Y_test, Y_pred_LR_smote), 
      recall_score(Y_test, Y_pred_LR_smote))


# In[279]:


# Calculate Importance of a feature
importance = classifier_LR_smote.coef_[0]
#  Feature Importance summary
for i,v in enumerate(importance):
    print('Feature: %0d, Score: %.5f' % (i,v))
# plot feature importance
plt.bar([x for x in range(len(importance))], importance)
plt.show()


# In[280]:


# LOGISTIC REGRESSION WITH ClusterCentroids


# In[281]:


from imblearn.under_sampling import ClusterCentroids

cc = ClusterCentroids(random_state=1)
X_train_cc, Y_train_cc = cc.fit_resample(X_train, Y_train)
Counter(Y_train_cc)


# In[282]:


# Train the Logistic Regression model using the resampled data
cluster_model = LogisticRegression(solver='saga', random_state=1, max_iter=1000)
cluster_model.fit(X_train_cc, Y_train_cc)


# In[283]:


Y_pred_LR_cc = cluster_model.predict(X_test)


# In[284]:


# Confusion matrix

confusion_matrix(Y_test, Y_pred_LR_cc)


# In[285]:


# Balanced accuracy, Precision and Recall

print(balanced_accuracy_score(Y_test, Y_pred_LR_cc), 
      average_precision_score(Y_test, Y_pred_LR_cc), 
      recall_score(Y_test, Y_pred_LR_cc))


# In[286]:


# XGBOOST CLASSIFIER


# In[287]:


# Identify best value for scale_ps_weight to adjust imbalance.
model = XGBClassifier(base_score=0.53, booster='gbtree', colsample_bylevel=1,
              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
              importance_type='gain', interaction_constraints='',eval_metric = 'error',
              learning_rate=0.1, max_delta_step=0, max_depth=5,
              min_child_weight=1, monotone_constraints='()',
              n_estimators=100, n_jobs=2, num_parallel_tree=1,
              objective='binary:logistic', random_state=1, reg_alpha=0,
              reg_lambda=1, scale_pos_weight=10, subsample=1,
              tree_method='exact', use_label_encoder=False,
              validate_parameters=1)

# Grid
weights = list(range(0,100,5))
param_grid = dict(scale_pos_weight=weights)

# Evaluation
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=2, random_state=1)

# Grid search
grid = GridSearchCV(estimator=model, param_grid=param_grid, n_jobs=-1, cv=cv, scoring='roc_auc')
grid_result = grid.fit(X_train, Y_train)

# Resulting configuration
print(grid_result.best_score_, grid_result.best_params_)

means_ = grid_result.cv_results_['mean_test_score']
stdv_ = grid_result.cv_results_['std_test_score']
params_ = grid_result.cv_results_['params']
for mean, stdev, param in zip(means_, stdv_, params_):
    print("%f (%f) with: %r" % (mean, stdev, param))


# In[293]:


classifier_XG = XGBClassifier(base_score=0.53, booster='gbtree', colsample_bylevel=1,
                              colsample_bynode=1, colsample_bytree=1, gamma=0, gpu_id=-1,
                              importance_type='gain', interaction_constraints=''
                              ,eval_metric = 'error',learning_rate=0.2, max_delta_step=0,
                              max_depth=5,min_child_weight=1, monotone_constraints='()',
                              n_estimators=100, n_jobs=2, num_parallel_tree=1,
                              objective='binary:logistic', random_state=1, reg_alpha=0,
                              reg_lambda=1, scale_pos_weight=5, subsample=1,
                              tree_method='exact', use_label_encoder=False,
                              validate_parameters=1)


# In[294]:


classifier_XG.fit(X_train, Y_train)


# In[295]:


Y_pred_XG = classifier_XG.predict(X_test)


# In[296]:


# Balanced accuracy, Precision and Recall

print(balanced_accuracy_score(Y_test, Y_pred_XG), 
      average_precision_score(Y_test, Y_pred_XG), 
      recall_score(Y_test, Y_pred_XG))


# In[297]:


# Confusion matrix

matrix_XG = confusion_matrix(Y_test, Y_pred_XG)
matrix_XG


# In[299]:


####
#   Using Logistic regression with Cluster Centroids for predictions.
####


# In[300]:


Y_pred = cluster_model.predict(X_pred)


# In[301]:


Y_pred

