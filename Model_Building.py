#!/usr/bin/env python
# coding: utf-8

# In[60]:



#1. Importing data & libraries

# Importing Libraries

import numpy as np
import pandas as pd
from numpy import mean
from numpy import std

import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')
import seaborn as sns




from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import OneHotEncoder , LabelEncoder ,normalize
from scipy import stats

import warnings
warnings.filterwarnings('ignore')

#for displaying 500 results in pandas dataframe
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)

import itertools

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve
import warnings


# In[61]:


# Modeling Dataframe
df = pd.read_excel(r"Modelling data.xlsx")
df.head(10)


# In[ ]:


#Is credit history, i.e. new loans in last six months, loans defaulted in last six months, 
##time since first loan, etc., a significant factor in estimating probability of loan defaulters?
#checking the correlation matrix for this


# In[62]:


sns.heatmap(df[['PRI_NO_ACCTS','PRI_ACTV_ACCTS','PRI_OVERDUE_ACCTS','PRI_CURR_BAL',
            'PRI_SANCT_AMT','PRI_DISBRSD_AMT','PRI_INSTAL_AMT', 'loan_default']].corr(),annot=True)


# In[ ]:


# The primary and secondary account details are not related to loan default probability.


# In[63]:


sns.heatmap(df[['NEW_ACCT_LAST_SIX_MNTHS','LOANS_DEFAULTED_LAST_SIX_MNTHS'
            ,'AVG_ACCT_AGE','NO_OF_INQUIRY','loan_default']].corr(),annot=True)


# In[ ]:


# Separating the dependent and independent varibles


# In[64]:


y=df['loan_default']
X=df.drop("loan_default",axis=1)


# In[65]:


y


# In[66]:


X.columns


# In[67]:


from sklearn.model_selection import train_test_split,KFold,cross_val_score
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=42)


# In[68]:


X_train.value_counts(normalize= True)
y_test.value_counts(normalize=True)


# In[69]:


print("Size of X",X.shape)
print("Size of y",y.shape)
print("Size of X_train",X_train.shape)
print("Size of y_train",y_train.shape)


# In[70]:


y_train_plot=pd.DataFrame(y_train,columns=['loan_default'])
y_test_plot=pd.DataFrame(y_test,columns=['loan_default'])

defaulters_train=y_train_plot['loan_default'].sum()
non_defaulters_train=len(y_train_plot)-y_train_plot['loan_default'].sum()
total_train=len(y_train_plot)

defaulters_test=y_test_plot['loan_default'].sum()
non_defaulters_test=len(y_test_plot)-y_test_plot['loan_default'].sum()
total_test=len(y_test_plot)


# In[71]:


#Graph
my_pal = {0: 'black', 1: 'red'}

plt.figure(figsize = (6, 3))
ax = sns.countplot(x = 'loan_default', data = y_train_plot, palette = my_pal)
plt.title('X_Train Class Distribution')
plt.show()


# In[72]:



#Graph
my_pal = {0: 'black', 1: 'red'}

plt.figure(figsize = (6, 3))
ax = sns.countplot(x = 'loan_default', data = y_test_plot, palette = my_pal)
plt.title('X_Test Class Distribution')
plt.show()


#  ## FEATURE SELECTION

# In[73]:


columnsToDelete = ['UniqueID','MobileNo_Avl_Flag','Current_pincode_ID',
                   'Employee_code_ID','State_ID','branch_id','manufacturer_id',
                   'supplier_id','DOB','DisbursalDate','NO_OF_INQUIRY']


# In[74]:


## BEFORE DELETING THE COLUMNS
print("Size AFTER Deleting the Features",len(X_train.columns))

## DROPING THE COLUMNS FROM THE DATA FRAME
X_train=X_train.drop(X_train[columnsToDelete],axis=1)

## AFTER DROPPING THE COLUMNS
print("Size AFTER Deleting the Features",len(X_train.columns))


# In[75]:


numericalTypes=['disbursed_amount', 'asset_cost', 'ltv', 
 'CNS_SCORE', 'PRI_NO_ACCTS', 'PRI_ACTV_ACCTS', 'PRI_OVERDUE_ACCTS',
'PRI_CURR_BAL', 'PRI_SANCT_AMT', 'PRI_DISBRSD_AMT', 'SEC_NO_ACCTS', 'SEC_ACTV_ACCTS', 
'SEC_OVERDUE_ACCTS','SEC_CURR_BAL', 'SEC_SANCTIONED_AMT', 'SEC_DISBURSED_AMT', 'PRI_INSTAL_AMT', 
'SEC_INSTAL_AMT', 'NEW_ACCT_LAST_SIX_MNTHS', 'LOANS_DEFAULTED_LAST_SIX_MNTHS', 'AVG_ACCT_AGE', 
'CRED_HIST_LEN', 'Age', 'AgeatDisbursal']
print(len(numericalTypes))

categoricalTypes= [ 'Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag','Emp_Type', 'CNS_SCORE_DESC']
print(len(categoricalTypes))


# In[76]:


## Creating New dataframe for Numerical and Categorical 
X_train_numerical=X_train[numericalTypes].copy()
X_test_numerical=X_test[numericalTypes].copy()


# In[77]:


#Combining the Primary and Secondry information of accounts
X_train_numerical.loc[:,'No_of_Accounts'] = X_train_numerical['PRI_NO_ACCTS'] + X_train_numerical['SEC_NO_ACCTS']
X_train_numerical.loc[:,'PRI_Inactive_accounts'] = X_train_numerical['PRI_NO_ACCTS'] - X_train_numerical['PRI_ACTV_ACCTS']
X_train_numerical.loc[:,'SEC_Inactive_accounts'] = X_train_numerical['SEC_NO_ACCTS'] - X_train_numerical['SEC_ACTV_ACCTS']
X_train_numerical.loc[:,'Total_Inactive_accounts'] = X_train_numerical['PRI_Inactive_accounts'] + X_train_numerical['SEC_Inactive_accounts']
X_train_numerical.loc[:,'Total_Overdue_Accounts'] = X_train_numerical['PRI_OVERDUE_ACCTS'] + X_train_numerical['SEC_OVERDUE_ACCTS']
X_train_numerical.loc[:,'Total_Current_Balance'] = X_train_numerical['PRI_CURR_BAL'] + X_train_numerical['SEC_CURR_BAL']
X_train_numerical.loc[:,'Total_Sanctioned_Amount'] = X_train_numerical['PRI_SANCT_AMT'] + X_train_numerical['SEC_SANCTIONED_AMT']
X_train_numerical.loc[:,'Total_Disbursed_Amount'] = X_train_numerical['PRI_DISBRSD_AMT'] + X_train_numerical['SEC_DISBURSED_AMT']
X_train_numerical.loc[:,'Total_Installment'] = X_train_numerical['PRI_INSTAL_AMT'] + X_train_numerical['SEC_INSTAL_AMT']



X_test_numerical.loc[:,'No_of_Accounts'] = X_test_numerical['PRI_NO_ACCTS'] + X_test_numerical['SEC_NO_ACCTS']
X_test_numerical.loc[:,'PRI_Inactive_accounts'] = X_test_numerical['PRI_NO_ACCTS'] - X_test_numerical['PRI_ACTV_ACCTS']
X_test_numerical.loc[:,'SEC_Inactive_accounts'] = X_test_numerical['SEC_NO_ACCTS'] - X_test_numerical['SEC_ACTV_ACCTS']
X_test_numerical.loc[:,'Total_Inactive_accounts'] = X_test_numerical['PRI_Inactive_accounts'] + X_test_numerical['SEC_Inactive_accounts']
X_test_numerical.loc[:,'Total_Overdue_Accounts'] = X_test_numerical['PRI_OVERDUE_ACCTS'] + X_test_numerical['SEC_OVERDUE_ACCTS']
X_test_numerical.loc[:,'Total_Current_Balance'] = X_test_numerical['PRI_CURR_BAL'] + X_test_numerical['SEC_CURR_BAL']
X_test_numerical.loc[:,'Total_Sanctioned_Amount'] = X_test_numerical['PRI_SANCT_AMT'] + X_test_numerical['SEC_SANCTIONED_AMT']
X_test_numerical.loc[:,'Total_Disbursed_Amount'] = X_test_numerical['PRI_DISBRSD_AMT'] + X_test_numerical['SEC_DISBURSED_AMT']
X_test_numerical.loc[:,'Total_Installment'] = X_test_numerical['PRI_INSTAL_AMT'] + X_test_numerical['SEC_INSTAL_AMT']


# In[78]:


X_test_numerical.columns


# In[79]:


X_train_numerical.columns


# In[80]:


print("Shape of X_train_numerical: ",X_train_numerical.shape)
print("Shape of X_test_numerical: ",X_test_numerical.shape)


# In[81]:


X_train_numerical=X_train_numerical.drop(['PRI_NO_ACCTS','SEC_NO_ACCTS','PRI_ACTV_ACCTS','SEC_ACTV_ACCTS','PRI_CURR_BAL','SEC_CURR_BAL','PRI_Inactive_accounts','SEC_Inactive_accounts','PRI_SANCT_AMT','SEC_SANCTIONED_AMT','PRI_DISBRSD_AMT','SEC_DISBURSED_AMT','PRI_OVERDUE_ACCTS','SEC_OVERDUE_ACCTS','PRI_INSTAL_AMT','SEC_INSTAL_AMT'],axis=1)


# In[82]:


X_test_numerical=X_test_numerical.drop(['PRI_NO_ACCTS','SEC_NO_ACCTS',
            'PRI_ACTV_ACCTS','SEC_ACTV_ACCTS',
            'PRI_CURR_BAL','SEC_CURR_BAL',
            'PRI_Inactive_accounts','SEC_Inactive_accounts',
            'PRI_SANCT_AMT','SEC_SANCTIONED_AMT',
            'PRI_DISBRSD_AMT','SEC_DISBURSED_AMT',
            'PRI_OVERDUE_ACCTS','SEC_OVERDUE_ACCTS',
            'PRI_INSTAL_AMT','SEC_INSTAL_AMT'],axis=1)


# In[83]:


X_test_numerical.columns


# In[84]:


X_train_numerical.columns


# In[85]:


print("After Droping: Shape of X_train_numerical: ",X_train_numerical.shape)
print("After Droping: Shape of X_test_numerical: ",X_test_numerical.shape)


# STANDARDIZING THE TRAIN AND TEST DATA

# In[86]:



scaler = StandardScaler()
scaler.fit(X_train_numerical)
X_train_numerical_std = scaler.transform(X_train_numerical)
X_test_numerical_std = scaler.transform(X_test_numerical)


# In[87]:


## Type of Returned Data
type(X_train_numerical_std)


# In[88]:


#Converting the ndarray to Pandas DataFrame with Column Names
X_train_numerical_std=pd.DataFrame(X_train_numerical_std,columns=['disbursed_amount', 'asset_cost', 'ltv', 'CNS_SCORE', 'NEW_ACCT_LAST_SIX_MNTHS',
                                                                  'LOANS_DEFAULTED_LAST_SIX_MNTHS', 'AVG_ACCT_AGE', 'CRED_HIST_LEN', 'Age', 'AgeatDisbursal', 
                                                                  'No_of_Accounts', 'Total_Inactive_accounts', 'Total_Overdue_Accounts', 'Total_Current_Balance', 
                                                                  'Total_Sanctioned_Amount', 'Total_Disbursed_Amount', 'Total_Installment'])

X_test_numerical_std=pd.DataFrame(X_test_numerical_std,columns=['disbursed_amount', 'asset_cost', 'ltv', 'CNS_SCORE', 'NEW_ACCT_LAST_SIX_MNTHS',
                                                                  'LOANS_DEFAULTED_LAST_SIX_MNTHS', 'AVG_ACCT_AGE', 'CRED_HIST_LEN', 'Age', 'AgeatDisbursal', 
                                                                  'No_of_Accounts', 'Total_Inactive_accounts', 'Total_Overdue_Accounts', 'Total_Current_Balance', 
                                                                  'Total_Sanctioned_Amount', 'Total_Disbursed_Amount', 'Total_Installment'])


# In[89]:


# Checking the values after converting
X_train_numerical_std


# In[90]:


print("Shape of Standardized X_train: ",X_train_numerical_std.shape)
print("Shape of Standardized X_test: ",X_test_numerical_std.shape)


# ENCODING THE CATEGORICAL VARIABLE USING ONE HOT ENCODER

# In[91]:


X_train_categorical=X_train[categoricalTypes].copy()
X_test_categorical=X_test[categoricalTypes].copy()


# In[92]:


X_train_categorical.columns


# In[93]:



onehot_encoder = OneHotEncoder(sparse=False)
X_train_categorical_encoded = onehot_encoder.fit(X_train_categorical)
X_train_categorical_encoded = onehot_encoder.transform(X_train_categorical) 
X_test_categorical_encoded = onehot_encoder.transform(X_test_categorical)


# In[94]:


# Checking the Encoded Data
X_train_categorical_encoded


# In[95]:


print("Shape of X_train after One Hot Encoding: ",X_train_categorical_encoded.shape)


# In[96]:


type(X_train_categorical_encoded)


# In[97]:


onehot_encoder.get_feature_names(['Aadhar_flag', 'PAN_flag', 'VoterID_flag', 'Driving_flag', 'Passport_flag', 'Emp_Type', 'CNS_SCORE_DESC'])


# In[98]:


## Adding the Obtained feature names into LIST
encodedCatColumnNames=['Aadhar_flag_0', 'Aadhar_flag_1', 'PAN_flag_0', 'PAN_flag_1',
       'VoterID_flag_0', 'VoterID_flag_1', 'Driving_flag_0',
       'Driving_flag_1', 'Passport_flag_0', 'Passport_flag_1',
       'Emp_Type_Salaried', 'Emp_Type_Self employed', 'Emp_Type_missing',
       'CNS_SCORE_DESC_0', 'CNS_SCORE_DESC_1', 'CNS_SCORE_DESC_2',
       'CNS_SCORE_DESC_3', 'CNS_SCORE_DESC_4', 'CNS_SCORE_DESC_5',
       'CNS_SCORE_DESC_6', 'CNS_SCORE_DESC_7', 'CNS_SCORE_DESC_8',
       'CNS_SCORE_DESC_9', 'CNS_SCORE_DESC_10', 'CNS_SCORE_DESC_11',
       'CNS_SCORE_DESC_12', 'CNS_SCORE_DESC_13', 'CNS_SCORE_DESC_14',
       'CNS_SCORE_DESC_15', 'CNS_SCORE_DESC_16', 'CNS_SCORE_DESC_17',
       'CNS_SCORE_DESC_18', 'CNS_SCORE_DESC_19']


# Converting ndarray to Pandas Data Frame

# In[99]:


X_train_categorical_encoded=pd.DataFrame(X_train_categorical_encoded,columns=['Aadhar_flag_0', 'Aadhar_flag_1', 'PAN_flag_0', 'PAN_flag_1',
       'VoterID_flag_0', 'VoterID_flag_1', 'Driving_flag_0',
       'Driving_flag_1', 'Passport_flag_0', 'Passport_flag_1',
       'Emp_Type_Salaried', 'Emp_Type_Self employed', 'Emp_Type_missing',
       'CNS_SCORE_DESC_0', 'CNS_SCORE_DESC_1', 'CNS_SCORE_DESC_2',
       'CNS_SCORE_DESC_3', 'CNS_SCORE_DESC_4', 'CNS_SCORE_DESC_5',
       'CNS_SCORE_DESC_6', 'CNS_SCORE_DESC_7', 'CNS_SCORE_DESC_8',
       'CNS_SCORE_DESC_9', 'CNS_SCORE_DESC_10', 'CNS_SCORE_DESC_11',
       'CNS_SCORE_DESC_12', 'CNS_SCORE_DESC_13', 'CNS_SCORE_DESC_14',
       'CNS_SCORE_DESC_15', 'CNS_SCORE_DESC_16', 'CNS_SCORE_DESC_17',
       'CNS_SCORE_DESC_18', 'CNS_SCORE_DESC_19'])

X_test_categorical_encoded=pd.DataFrame(X_test_categorical_encoded,columns=['Aadhar_flag_0', 'Aadhar_flag_1', 'PAN_flag_0', 'PAN_flag_1',
       'VoterID_flag_0', 'VoterID_flag_1', 'Driving_flag_0',
       'Driving_flag_1', 'Passport_flag_0', 'Passport_flag_1',
       'Emp_Type_Salaried', 'Emp_Type_Self employed', 'Emp_Type_missing',
       'CNS_SCORE_DESC_0', 'CNS_SCORE_DESC_1', 'CNS_SCORE_DESC_2',
       'CNS_SCORE_DESC_3', 'CNS_SCORE_DESC_4', 'CNS_SCORE_DESC_5',
       'CNS_SCORE_DESC_6', 'CNS_SCORE_DESC_7', 'CNS_SCORE_DESC_8',
       'CNS_SCORE_DESC_9', 'CNS_SCORE_DESC_10', 'CNS_SCORE_DESC_11',
       'CNS_SCORE_DESC_12', 'CNS_SCORE_DESC_13', 'CNS_SCORE_DESC_14',
       'CNS_SCORE_DESC_15', 'CNS_SCORE_DESC_16', 'CNS_SCORE_DESC_17',
       'CNS_SCORE_DESC_18', 'CNS_SCORE_DESC_19'])


# In[100]:


print("Shape of Encoded X_train Categorical: ",X_train_categorical_encoded.shape)
print("Shape of Encoded X_test Categorical: ",X_test_categorical_encoded.shape)


# In[101]:


X_test_categorical_encoded


# In[102]:


X_train_merged = pd.concat([X_train_numerical_std,X_train_categorical_encoded], axis=1)
X_test_merged = pd.concat([X_test_numerical_std,X_test_categorical_encoded], axis=1)
X_train_merged.columns


# In[103]:



print("Shape of X_train Merged: ",X_train_merged.shape)
print("Shape of X_test Merged: ",X_test_merged.shape)


# In[104]:


print("Shape of y_train : ",y_train.shape)
print("Shape of y_test : ",y_test.shape)


# In[105]:


from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.metrics import confusion_matrix, accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import precision_score, recall_score, confusion_matrix, classification_report, roc_curve, precision_recall_curve


# In[106]:


from sklearn.metrics import roc_auc_score,accuracy_score,precision_score,recall_score,f1_score
from sklearn.metrics import confusion_matrix, roc_curve, auc


# In[107]:


lr = LogisticRegression(C=5.0,class_weight="balanced")


# In[108]:


# This function plots the confusion matrices given y_i, y_i_hat.
def plot_confusion_matrix(test_y, predict_y):
    C = confusion_matrix(test_y, predict_y)
    A =(((C.T)/(C.sum(axis=1))).T)
    B =(C/C.sum(axis=0))
    plt.figure(figsize=(20,4))
    
    labels = [1,2]
    # representing A in heatmap format
    cmap=sns.light_palette("blue")
    plt.subplot(1, 3, 1)
    sns.heatmap(C, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Confusion matrix")
    
    plt.subplot(1, 3, 2)
    sns.heatmap(B, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Precision matrix")
    
    plt.subplot(1, 3, 3)
    # representing B in heatmap format
    sns.heatmap(A, annot=True, cmap=cmap, fmt=".3f", xticklabels=labels, yticklabels=labels)
    plt.xlabel('Predicted Class')
    plt.ylabel('Original Class')
    plt.title("Recall matrix")
    
    plt.show()


# In[109]:


accuracy = {}
roc_r = {}

def train_model(model):
    # Checking accuracy
    model = model.fit(X_train_merged, y_train)
    pred = model.predict(X_test_merged)
    acc = accuracy_score(y_test, pred)*100
    accuracy[model] = acc
    print('accuracy_score',acc)
    print('precision_score',precision_score(y_test, pred)*100)
    print('recall_score',recall_score(y_test, pred)*100)
    print('f1_score',f1_score(y_test, pred)*100)
    roc_score = roc_auc_score(y_test, pred)*100
    roc_r[model] = roc_score
    print('roc_auc_score',roc_score)
    
    # confusion matrix
    print('confusion_matrix')
    plot_confusion_matrix(y_test,pred)
#     print(pd.DataFrame(confusion_matrix(y_test, pred)))
    fpr, tpr, threshold = roc_curve(y_test, pred)
    roc_auc = auc(fpr, tpr)*100

    plt.title('Receiver Operating Characteristic')
    plt.plot(fpr, tpr, 'b', label = 'AUC = %0.2f' % roc_auc)
    plt.legend(loc = 'lower right')
    plt.plot([0, 1], [0, 1],'r--')
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.ylabel('True Positive Rate')
    plt.xlabel('False Positive Rate')
    plt.rcParams["figure.figsize"] = [7,7]
#     plt.figure(figsize=(2,3))
#     plt.figure(figsize=(3,4))
    plt.show()


# In[110]:


train_model(lr)


# In[ ]:





# In[111]:


#from sklearn.neighbors import KNeighborsClassifier


# In[112]:


#knn = KNeighborsClassifier(weights='distance', algorithm='auto', n_neighbors=15)


# In[113]:


#train_model(knn)


# ## Dealing with Imbalanced data

# In[114]:


pip install imblearn -U-


# In[115]:


pip install scikit-learn==0.23.1


# In[116]:


#from imblearn import under_sampling 
#from imblearn import over_sampling
from imblearn.over_sampling import SMOTE


# In[117]:


# setting up testing and training sets
smote = SMOTE()
X_train_smote, y_train_smote = smote.fit_resample(X_train_merged.astype('float'),y_train)


# In[118]:


from collections import Counter
print("Before SMOTE :" , Counter(y_train))
print("After SMOTE :" , Counter(y_train_smote))


# In[119]:


model=LogisticRegression()
model.fit(X_train_smote,y_train_smote)
y_predict =model.predict(X_test_merged)
acc=print(accuracy_score(y_test,y_predict))
pd.crosstab(y_test,y_predict)


# In[120]:


acc = accuracy_score(y_test, y_predict)*100
accuracy[model] = acc
print('accuracy_score',acc)
print('precision_score',precision_score(y_test, y_predict)*100)
print('recall_score',recall_score(y_test, y_predict)*100)
print('f1_score',f1_score(y_test, y_predict)*100)


# In[ ]:





# In[ ]:




