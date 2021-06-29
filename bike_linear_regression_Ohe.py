#!/usr/bin/env python
# coding: utf-8

# # Bike Sharing Dataset Linear Modeling
# 
# + Based on Bike Sharing dataset from [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
# + This notebook is based upon the hourly data file, i.e. hour.csv
# + This notebook showcases linear modeling using linear regression

# ### Problem Statement
# Given the Bike Sharing dataset with hourly level information of bikes along with weather and other attributes, model a system which can predict the bike count.

# ## Import required packages

# In[229]:


get_ipython().run_line_magic('matplotlib', 'inline')
# data manuipulation
import numpy as np
import pandas as pd

# modeling utilities
import scipy.stats as stats
from sklearn import metrics
from sklearn import preprocessing
from sklearn import  linear_model
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_predict

# plotting libraries
import matplotlib.pyplot as plt
import seaborn as sn


sn.set_style('whitegrid')
sn.set_context('talk')
params = {'legend.fontsize': 'x-large',
          'figure.figsize': (30, 10),
          'axes.labelsize': 'x-large',
          'axes.titlesize':'x-large',
          'xtick.labelsize':'x-large',
          'ytick.labelsize':'x-large'}

plt.rcParams.update(params)


# ## Load Dataset

# In[230]:


hour_df = pd.read_csv('hour.csv')
print("Shape of dataset::{}".format(hour_df.shape))


# In[231]:


hour_df.head()


# In[232]:


print("Shape of dataset::{}".format(hour_df.shape))


# In[233]:


#from sklearn.preprocessing import OneHotEncoder
#ohe = OneHotEncoder()
#y = ohe.fit_transform(y).toarray()


# ## Preprocessing
# + Standarize column names
# + Typecast attributes
# + Encode Categoricals using One Hot Encoding

# ### Standarize Column Names

# In[234]:


hour_df.rename(columns={'instant':'rec_id',
                        'dteday':'datetime',
                        'holiday':'is_holiday',
                        'workingday':'is_workingday',
                        'weathersit':'weather_condition',
                        'hum':'humidity',
                        'mnth':'month',
                        'cnt':'total_count',
                        'hr':'hour',
                        'yr':'year'},inplace=True)


# In[235]:


#Datatypes before conversion
hour_df.dtypes


# ### Typecast Attributes

# In[236]:


# date time conversion
hour_df['datetime'] = pd.to_datetime(hour_df.datetime)

# categorical variables
hour_df['season'] = hour_df.season.astype('category')
hour_df['is_holiday'] = hour_df.is_holiday.astype('category')
hour_df['weekday'] = hour_df.weekday.astype('category')
hour_df['weather_condition'] = hour_df.weather_condition.astype('category')
hour_df['is_workingday'] = hour_df.is_workingday.astype('category')
hour_df['month'] = hour_df.month.astype('category')
hour_df['year'] = hour_df.year.astype('category')
hour_df['hour'] = hour_df.hour.astype('category')


# In[237]:


#Datatypes after conversion
hour_df.dtypes


# 
# ### Encode Categoricals (One Hot Encoding)

# In[238]:


def fit_transform_ohe(df,col_name):
    """This function performs one hot encoding for the specified
        column.

    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        col_name: the column to be one hot encoded

    Returns:
        tuple: label_encoder, one_hot_encoder, transformed column as pandas Series

    """
    # label encode the column
    le = preprocessing.LabelEncoder()
    le_labels = le.fit_transform(df[col_name])
    df[col_name+'_label'] = le_labels
    
    # one hot encoding
    ohe = preprocessing.OneHotEncoder()
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    
    return le,ohe,features_df

# given label encoder and one hot encoder objects, 
# encode attribute to ohe
def transform_ohe(df,le,ohe,col_name):
    """This function performs one hot encoding for the specified
        column using the specified encoder objects.

    Args:
        df(pandas.DataFrame): the data frame containing the mentioned column name
        le(Label Encoder): the label encoder object used to fit label encoding
        ohe(One Hot Encoder): the onen hot encoder object used to fit one hot encoding
        col_name: the column to be one hot encoded

    Returns:
        tuple: transformed column as pandas Series

    """
    # label encode
    col_labels = le.transform(df[col_name])
    df[col_name+'_label'] = col_labels
    
    # ohe 
    feature_arr = ohe.fit_transform(df[[col_name+'_label']]).toarray()
    feature_labels = [col_name+'_'+str(cls_label) for cls_label in le.classes_]
    features_df = pd.DataFrame(feature_arr, columns=feature_labels)
    
    return features_df


# # Single selections using iloc and DataFrame
# 
# # Rows:
# 
# data.iloc[0]    # first row of data frame 
# 
# data.iloc[1]    # second row of data frame 
# 
# data.iloc[-1]   # last row of data frame
# 
# # Columns:
# 
# data.iloc[:,0]  # first column of data frame 
# 
# data.iloc[:,1]  # second column of data frame 
# 
# data.iloc[:,-1] # last column of data frame

# 
# # Multiple row and column selections using iloc and DataFrame
# 
# data.iloc[0:5] # first five rows of dataframe
# 
# data.iloc[:, 0:2] # first two columns of data frame with all rows
# 
# data.iloc[[0,3,6,24], [0,5,6]] # 1st, 4th, 7th, 25th row + 1st 6th 7th columns.
# 
# data.iloc[0:5, 5:8] # first 5 rows and 5th, 6th, 7th columns of data frame.

# In[239]:


hour_df.head()


# In[240]:


hour_df.shape


# ## Train-Test Split

# In[241]:


#hour_df.iloc[:,0:-3]     #Seperates last 3 cols
# hour_df.iloc[:,-1]      #Seperates last col


X, X_test, y, y_test = train_test_split(hour_df.iloc[:,0:-3], hour_df.iloc[:,-1], 
                                                    test_size=0.33, random_state=42)

X.reset_index(inplace=True)
y = y.reset_index()

X_test.reset_index(inplace=True)
y_test = y_test.reset_index()

print("Training set::{}{}".format(X.shape,y.shape))
print("Testing set::{}".format(X_test.shape))


# In[242]:


X.head()


# In[243]:


y.head()


# In[244]:


cat_attr_list = ['season','is_holiday',
                 'weather_condition','is_workingday',
                 'hour','weekday','month','year']
numeric_feature_cols = ['temp','humidity','windspeed','hour','weekday','month','year']
#one hot to be done for below features
subset_cat_features =  ['season','is_holiday','weather_condition','is_workingday']


# In[245]:


encoded_attr_list = []
for col in cat_attr_list:
        return_obj = fit_transform_ohe(X,col)
        encoded_attr_list.append({'label_enc':return_obj[0],
                              'ohe_enc':return_obj[1],
                              'feature_df':return_obj[2],
                              'col_name':col})


# In[246]:


feature_df_list = [X[numeric_feature_cols]]
feature_df_list.extend([enc['feature_df']                         for enc in encoded_attr_list                         if enc['col_name'] in subset_cat_features])

train_df_new = pd.concat(feature_df_list, axis=1)
print("Shape::{}".format(train_df_new.shape))


# In[247]:


#feature_df_list


# In[248]:


train_df_new.head()


# In[249]:


print('Before ohe Shape is ::', hour_df.shape)   # Original set
print('After ohe Shape is ::', train_df_new.shape) #Training set after dropping some features


# In[250]:


#Catagorical features won't appear below
train_df_new.describe()


# ## Linear Regression

# In[251]:


# reshape (-1,1) creates new array with just one column numpy array  
X = train_df_new
y= y.total_count.values.reshape(-1,1)

lin_reg = linear_model.LinearRegression()


# ### Cross Validation

# In[252]:


#for training data
predicted = cross_val_predict(lin_reg, X, y, cv=10)

fig, ax = plt.subplots()
ax.scatter(y, y-predicted)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residual')
plt.show()


# In[253]:


r2_scores = cross_val_score(lin_reg, X, y, cv=10)
mse_scores = cross_val_score(lin_reg, X, y, cv=10,scoring='neg_mean_squared_error')


# In[254]:


fig, ax = plt.subplots()
ax.plot([i for i in range(len(r2_scores))],r2_scores,lw=2)
ax.set_xlabel('Iteration')
ax.set_ylabel('R-Squared')
ax.title.set_text("Cross Validation Scores, Avg:{}".format(np.average(r2_scores)))
plt.show()


# In[255]:


lin_reg.fit(X,y)


# ## Test Dataset Performance

# In[256]:


test_encoded_attr_list = []
for enc in encoded_attr_list:
    col_name = enc['col_name']
    le = enc['label_enc']
    ohe = enc['ohe_enc']
    test_encoded_attr_list.append({'feature_df':transform_ohe(X_test,
                                                              le,ohe,
                                                              col_name),
                                   'col_name':col_name})
    
    
test_feature_df_list = [X_test[numeric_feature_cols]]
test_feature_df_list.extend([enc['feature_df']                              for enc in test_encoded_attr_list                              if enc['col_name'] in subset_cat_features])

test_df_new = pd.concat(test_feature_df_list, axis=1) 
print("Shape::{}".format(test_df_new.shape))


# In[257]:


X_test = test_df_new
y_test = y_test.total_count.values.reshape(-1,1)

y_pred = lin_reg.predict(X_test)

residuals = y_test-y_pred


# In[258]:


r2_score = lin_reg.score(X_test,y_test)
print("R-squared::{}".format(r2_score))
print("MSE: %.2f"
      % metrics.mean_squared_error(y_test, y_pred))


# In[259]:


fig, ax = plt.subplots()
ax.scatter(y_test, residuals)
ax.axhline(lw=2,color='black')
ax.set_xlabel('Observed')
ax.set_ylabel('Residuals')
ax.title.set_text("Residual Plot with R-Squared={}".format(np.average(r2_score)))
plt.show()


# ## Stats Models

# In[260]:


import statsmodels.api as sm

# Set the independent variable
X = X.values.tolist()

# This handles the intercept. 
# Statsmodel takes 0 intercept by default
X = sm.add_constant(X)

X_test = X_test.values.tolist()
X_test = sm.add_constant(X_test)


# Build OLS model
model = sm.OLS(y, X)
results = model.fit()

# Get the predicted values for dependent variable
pred_y = results.predict(X_test)

# View Model stats
print(results.summary())


# In[261]:


plt.scatter(pred_y,y_test)


# In[ ]:





# In[ ]:




