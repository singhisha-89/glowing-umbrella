#!/usr/bin/env python
# coding: utf-8

# In[90]:



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


# In[91]:


# Dataframe
df = pd.read_excel(r"data_loan.xlsx")
df.head(10)


# In[92]:


df.columns


# In[93]:


#Changing the names of the columns
# This is one of the ways in which we can do it
## CHANGING TO UPPER CASE
###dd1.columns=dd1.columns.str.upper()
### REPLACING "." with "_" in column names
#dd1.columns=dd1.columns.str.replace(".","_")


# In[94]:



df.rename(columns={   'PERFORM_CNS.SCORE':'CNS_SCORE',
                      'Date.of.Birth':'DOB',
                      'Employment.Type':'Emp_Type',
                      'PERFORM_CNS.SCORE':'CNS_SCORE',
                      'PERFORM_CNS.SCORE.DESCRIPTION':'CNS_SCORE_DESC', 
                      'PRI.NO.OF.ACCTS':'PRI_NO_ACCTS',
                      'PRI.ACTIVE.ACCTS':'PRI_ACTV_ACCTS',
                      'PRI.OVERDUE.ACCTS':'PRI_OVERDUE_ACCTS', 
                      'PRI.CURRENT.BALANCE': 'PRI_CURR_BAL', 
                      'PRI.SANCTIONED.AMOUNT':'PRI_SANCT_AMT',
                      'PRI.DISBURSED.AMOUNT': 'PRI_DISBRSD_AMT',
                      'SEC.NO.OF.ACCTS':'SEC_NO_ACCTS', 
                      'SEC.ACTIVE.ACCTS': 'SEC_ACTV_ACCTS', 
                      'SEC.OVERDUE.ACCTS':'SEC_OVERDUE_ACCTS',
                      'SEC.CURRENT.BALANCE':'SEC_CURR_BAL',
                      'SEC.SANCTIONED.AMOUNT':'SEC_SANCTIONED_AMT',
                      'SEC.DISBURSED.AMOUNT':'SEC_DISBURSED_AMT',
                      'PRIMARY.INSTAL.AMT':'PRI_INSTAL_AMT',
                      'SEC.INSTAL.AMT':'SEC_INSTAL_AMT',
                      'NEW.ACCTS.IN.LAST.SIX.MONTHS':'NEW_ACCT_LAST_SIX_MNTHS',
                      'DELINQUENT.ACCTS.IN.LAST.SIX.MONTHS':'LOANS_DEFAULTED_LAST_SIX_MNTHS',
                      'AVERAGE.ACCT.AGE':'AVG_ACCT_AGE',
                      'CREDIT.HISTORY.LENGTH':'CRED_HIST_LEN',
                      'NO.OF_INQUIRIES':'NO_OF_INQUIRY'},inplace=True)


# In[95]:


df.info()


# In[96]:


# Provide the statistical description of the quantitative data variables
df.describe()


# In[97]:


df.isnull().sum(axis=0)


# In[98]:


df['Emp_Type'].value_counts()


# In[99]:


df['Emp_Type'] .unique()


# In[100]:


df['Emp_Type'].fillna('missing', inplace = True)
df['Emp_Type'] .unique()


# In[101]:


df.isnull().sum(axis=0)


# In[102]:


# How is the target variable distributed overall?
sns.countplot(df['Emp_Type'])
plt.title("Emp_Type")
plt.show()


# In[103]:


# Creating plot
df.groupby(by='Emp_Type')['loan_default'].sum().plot.pie(autopct="%.1f%%")


# defaulters belonging to self employed catagory are more than the other two catagories i.e. 'Salaried' and 'missing'

# In[104]:


# Finding te duplicates if there are any
df.duplicated().unique


# In[105]:


df.columns


# In[106]:


for col in ['UniqueID','branch_id','supplier_id', 'manufacturer_id', 'Current_pincode_ID', 
       'Emp_Type','State_ID', 'Employee_code_ID',
       'MobileNo_Avl_Flag', 'Aadhar_flag', 'PAN_flag', 'VoterID_flag',
       'Driving_flag', 'Passport_flag','CNS_SCORE_DESC','loan_default'
       ]: df[col] = df[col].astype('category')
   


# In[107]:


# Changing date variables type to datetime
df['DOB'] =  pd.to_datetime(df['DOB'], format='%d-%m-%Y')
df['DisbursalDate'] =  pd.to_datetime(df['DisbursalDate'], format='%d-%m-%Y')


# In[108]:


df.head()


# In[109]:


df['loan_default'].value_counts()
# class_df =df.groupby('loan_default').count()['UniqueID'].reset_index().sort_values(by='UniqueID',ascending=False)
# class_df.style.background_gradient(cmap='winter')


# In[110]:



#Graph
my_pal = {0: 'blue', 1: 'red'}

plt.figure(figsize = (12, 6))
ax = sns.countplot(x = 'loan_default', data = df, palette = my_pal)
plt.title('Class Distribution')
plt.show()

# As we can see that the data is highly imbalaced and we will to handle it later to remove the error. 
#It can be done by various methods like SMOTE, upsampling, downsampling etc 


# In[111]:


df['AVG_ACCT_AGE']


# In[112]:


import re
df['AVG_ACCT_AGE']=df['AVG_ACCT_AGE'].map(lambda x : re.sub("[^0-9]+"," ",x))
df['AVG_ACCT_AGE']


# In[113]:


df['AVG_ACCT_AGE']=df['AVG_ACCT_AGE'].str.split(" ",expand=True)[0].astype(int)*12 + df['AVG_ACCT_AGE'].str.split(" ",expand=True)[1].astype(int)
df['AVG_ACCT_AGE']


# In[114]:


import re
df['CRED_HIST_LEN']= df['CRED_HIST_LEN'].map(lambda x : re.sub("[^0-9]+"," ",x))


# In[115]:



df['CRED_HIST_LEN']= df['CRED_HIST_LEN'].str.split(" ",expand=True)[0].astype(int)*12 + df['CRED_HIST_LEN'].str.split(" ",expand=True)[1].astype(int)
df['CRED_HIST_LEN']


# In[116]:


df.info()


# 
# def duration(dur):
#     yrs = int(dur.split(' ')[0].replace('yrs',''))
#     mon = int(dur.split(' ')[1].replace('mon',''))
#     return yrs*12+mon

# def age(dur):
#     yr = int(dur.split('-')[2])
#     if yr >=0 and yr<=19:
#         return yr+2000
#     else:
#          return yr+1900
# 
# df['DOB'] = df['DOB'].apply(age)
# df['DisbursalDate'] = df['DisbursalDate'].apply(age)
# df['Age']=df['DisbursalDate']-df['DOB']
# df=df.drop(['DisbursalDate','DOB'],axis=1)

# In[117]:


### Creating the age column of customers and doing the analysis for it against the loan_default


# In[118]:


df['Age']=2021- pd.DatetimeIndex(df['DOB']).year


# In[119]:


df['Age'].unique


# In[120]:


sns.scatterplot(x='loan_default',y='Age',data=df)


# In[121]:



sns.distplot(df['Age'], color = 'red')
plt.title('Distribution of Age')


# In[122]:


sns.catplot(data=df,kind='count',x='Age',hue='loan_default')


# In[123]:


df['AgeatDisbursal'] = ((df['DisbursalDate'] - df['DOB'])/365).dt.days


# In[124]:


sns.distplot(df['AgeatDisbursal'], color = 'red')
plt.title('Age on Disbursal Date')


# In[125]:


df.select_dtypes(include=['category']).nunique()

# We will proceed to drop those variables that have too many categories. These will only add noise to the model.
# df=df.drop(['BRANCH_ID','CURRENT_PINCODE_ID','EMPLOYEE_CODE_ID','SUPPLIER_ID'],axis=1)


# # EDA

# ### Univariate Analysis

# In[126]:


plt.figure(figsize=(15,5))
sns.countplot(df['manufacturer_id'])
plt.title("manufacturer_id")
plt.show()


# In[127]:


plt.figure(figsize=(15,5))
sns.countplot(df['State_ID'])
plt.title("State_ID")
plt.show()


# In[128]:


df['NEW_ACCT_LAST_SIX_MNTHS'].value_counts(normalize=100).head() 


# In[129]:


sns.distplot(df['NEW_ACCT_LAST_SIX_MNTHS'], color = 'red')
plt.title('NEW_ACCT_LAST_SIX_MNTHS')


# In[130]:


plt.figure(figsize=(15,5))
sns.countplot(df['LOANS_DEFAULTED_LAST_SIX_MNTHS'])
plt.title("LOANS_DEFAULTED_LAST_SIX_MNTHS ")
plt.show()


# In[131]:


# Do customer who make higher no. of enquiries end up being higher risk candidates?
plt.figure(figsize=(15,5))
sns.countplot(df['NO_OF_INQUIRY'])
plt.title("NO_OF_INQUIRY")
plt.show()


# In[132]:


#Based on the definition doesnt seem like it might affect the loan_defaulter intuitively.
#Based on the correlation coefficient too it seems like its not affecting defaulters list.


# In[133]:


df[['loan_default','NO_OF_INQUIRY']].corr()


# In[134]:


plt.figure(figsize=(15,5))
sns.countplot(df['CNS_SCORE_DESC'])
plt.title("CNS_SCORE_DESC")
plt.xticks(rotation=90)
plt.show()


# In[135]:


df['CNS_SCORE_DESC'].value_counts(normalize=100)
# Nearly 50% of the data has no CNS score description available


# In[136]:


df.columns


# In[137]:


fig, ax = plt.subplots(1,4)
sns.countplot(df['PRI_ACTV_ACCTS'], ax=ax[0])
sns.countplot(df['SEC_NO_ACCTS'], ax=ax[1])
sns.countplot(df['PRI_OVERDUE_ACCTS'], ax=ax[2])
sns.countplot(df['SEC_OVERDUE_ACCTS'], ax=ax[3])


# In[138]:


#What type of ID was presented by most of the customers as proof?
# Clearly by looking at the values we can say that maximum people use Adhar card as the ID

print(df['Aadhar_flag'].value_counts())
print(df['PAN_flag'].value_counts())
print(df['VoterID_flag'].value_counts())
print(df['Driving_flag'].value_counts())
print(df['Passport_flag'].value_counts())


# #Study the distribution of the target variable across the various 
# #categories such as branch, city, state, branch, supplier, manufacturer, etc. 
# 
# # Bivariate Analysis

# In[139]:


supplier_loan =pd.crosstab(df['branch_id'],df['loan_default'])
supplier_loan


# In[140]:



chi_sq, p_value, deg_freedom, exp_freq = stats.chi2_contingency(supplier_loan)
print('Chi Square Statistics',chi_sq)
print('p-value',p_value)
print('Degree of freedom',deg_freedom)


# In[141]:


#sns.barplot(x='State_ID',y='loan_default',data=df,estimator=np.sum)


# In[142]:


#sns.barplot(x=,y='loan_default',data=df,estimator=np.sum)
sns.catplot(data=df,kind='count',x='Current_pincode_ID',hue='loan_default')


# In[143]:


supplier_loan= pd.crosstab(df['supplier_id'],df['loan_default'])
supplier_loan


# In[144]:


chi_sq, p_value, deg_freedom, exp_freq = stats.chi2_contingency(supplier_loan)
print('Chi Square Statistics',chi_sq)
print('p-value',p_value)
print('Degree of freedom',deg_freedom)


# In[145]:


# sns.barplot(x='manufacturer_id',y='loan_default',data=df,estimator=np.sum)


# 3. Performing EDA and Modelling:
# Study the credit bureau score distribution. How is the distribution for defaulters vs non-defaulters? Explore in detail.
# 
# Explore the primary and secondary account details. Is the information in some way related to loan default probability ?
# 
# Is there a difference between the sanctioned and disbursed amount of primary & secondary loans. Study the difference by providing apt statistics and graphs.
# 
# Do customer who make higher no. of enquiries end up being higher risk candidates? 
# 
# Is credit history, i.e. new loans in last six months, loans defaulted in last six months, time since first loan, etc., a significant factor in estimating probability of loan defaulters?
# 
# Perform logistic regression modelling, predict the outcome for the test data, and validate the results using the confusion matrix.

# In[146]:


#sns.barplot(x='NO_OF_INQUIRY',y='loan_default',data=df,estimator=np.sum)


# ###'PERFORM_CNS.SCORE', 'PERFORM_CNS.SCORE.DESCRIPTION'
# 

# In[147]:


df[['CNS_SCORE', 'CNS_SCORE_DESC']]


# In[148]:


#Study the credit bureau score distribution. How is the distribution for defaulters vs non-defaulters? Explore in detail.


# In[149]:


df['CNS_SCORE'].nunique()
df['CNS_SCORE'].unique()
df['CNS_SCORE'].value_counts()


# In[150]:


sns.distplot(df[df['loan_default']==0]['CNS_SCORE'],hist=False,color='r')
sns.distplot(df[df['loan_default']==1]['CNS_SCORE'],hist=False)


# In[151]:


#Taking values where Beareau score is not 0. This is to eliminate the 0 peak. 
# We will only consider those people who have taken the loan. 
# 0 Bureau score implies that the loan has not been taken yet.


# In[152]:


df2= pd.DataFrame(df[df['CNS_SCORE']!=0])


# In[153]:


sns.distplot(df2[df2['loan_default']==0]['CNS_SCORE'],hist=False,color='r')
sns.distplot(df2[df2['loan_default']==1]['CNS_SCORE'],hist=False)


# In[154]:


#The Bureau score for the people who did not default lies in the range 600 to 800.


# In[155]:


sns.distplot(df['CNS_SCORE'])


# In[156]:


#sns.distplot(df['CNS_SCORE_DESC'])


# In[157]:


def cns_score(score):
    if score<100:
        return 0
    elif (score>=100) & (score<200):
        return 1
    elif (score>=200) & (score<300):
        return 2
    elif (score>=300) & (score<400):
        return 3
    elif (score>=400) & (score<500):
        return 4
    elif (score>=500) & (score<600):
        return 5
    elif (score>=600) & (score <700):
        return 6
    elif (score>=700) & (score <800):
        return 7
    elif (score>=800) & (score <900):
        return 8
    elif (score>=900) & (score <1000):
        return 9
    else:
        return 10


# In[158]:



cns_score(1004)


# In[159]:


df['CNS_SCORE'].map(lambda x:cns_score(x)).value_counts()


# In[160]:


df['CNS_SCORE']=df['CNS_SCORE'].map(lambda x:cns_score(x))


# In[161]:


df[ 'CNS_SCORE_DESC'].value_counts()


# In[162]:


from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()


# In[163]:


df['CNS_SCORE_DESC']=le.fit_transform(df['CNS_SCORE_DESC'])


# In[164]:


df['CNS_SCORE_DESC'].value_counts()


# In[165]:


for i in ['CNS_SCORE', 'CNS_SCORE_DESC']:
    print('Feature:',i)
    chi_sq, p_value, deg_freedom, exp_freq = stats.chi2_contingency(pd.crosstab(df[i],df['loan_default']))
    print('Chi Square Statistics',chi_sq)
    print('p-value',p_value)
    print('Degree of freedom',deg_freedom)
    print()


# In[166]:


primary= df.loc[:,[ 'PRI_NO_ACCTS', 'PRI_ACTV_ACCTS', 'PRI_OVERDUE_ACCTS', 'PRI_CURR_BAL', 'PRI_SANCT_AMT', 'PRI_DISBRSD_AMT']]


# In[167]:


primary.describe()


# In[168]:


### Primary No of accounts,Secondary No of accounts
#Is there a difference between the sanctioned and disbursed amount of primary & secondary loans.
#Study the difference by providing apt statistics and graphs


# In[169]:


fig,axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(df['PRI_NO_ACCTS'],ax=axes[0],color='r')
sns.distplot(df['SEC_NO_ACCTS'],ax=axes[1])
plt.show()


# In[170]:


### Primary Active accounts,Secondary Active accounts
fig,axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(df['PRI_ACTV_ACCTS'],ax=axes[0],color='r')
sns.distplot(df['SEC_ACTV_ACCTS'],ax=axes[1])
plt.show()


# In[171]:



fig,axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(df['PRI_OVERDUE_ACCTS'],ax=axes[0],color='r')
sns.distplot(df['SEC_OVERDUE_ACCTS'],ax=axes[1])
plt.show()


# In[172]:


fig,axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(df['PRI_CURR_BAL'],ax=axes[0],color='r')
sns.distplot(df['SEC_CURR_BAL'],ax=axes[1])
plt.show()


# In[173]:


fig,axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(df['PRI_SANCT_AMT'],ax=axes[0],color='r')
sns.distplot(df['SEC_SANCTIONED_AMT'],ax=axes[1])
plt.show()


# In[174]:


print(df['PRI_SANCT_AMT'].min())
print(df['SEC_SANCTIONED_AMT'].min())

print(df['PRI_SANCT_AMT'].mean())
print(df['SEC_SANCTIONED_AMT'].mean())

print(df['PRI_SANCT_AMT'].max())
print(df['SEC_SANCTIONED_AMT'].max())

print(df['PRI_SANCT_AMT'].std())
print(df['SEC_SANCTIONED_AMT'].std())


# In[175]:


fig,axes = plt.subplots(1,2,figsize=(15,5))
sns.distplot(df['PRI_DISBRSD_AMT'],ax=axes[0],color='r')
sns.distplot(df['SEC_DISBURSED_AMT'],ax=axes[1])
plt.show()


# In[176]:


print(df['PRI_DISBRSD_AMT'].min())
print(df['SEC_DISBURSED_AMT'].min())

print(df['PRI_DISBRSD_AMT'].mean())
print(df['SEC_DISBURSED_AMT'].mean())

print(df['PRI_DISBRSD_AMT'].max())
print(df['SEC_DISBURSED_AMT'].max())

print(df['PRI_DISBRSD_AMT'].std())
print(df['SEC_DISBURSED_AMT'].std())

