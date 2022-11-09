# UPLOADING FILES
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import scipy.stats as stats
train = pd.read_csv("C:/Users/ankro/OneDrive/Desktop/rstrain.csv")
test = pd.read_csv("C:/Users/ankro/OneDrive/Desktop/rstest.csv")
pd.set_option('display.max_columns',None)

# Data Exploraty Analysis
train.sample(10)

from pandas_profiling import ProfileReport
prof = ProfileReport(train)
prof.to_file(output_file= "C:/Users/ankro/OneDrive/Desktop/output.html")

pd.set_option('display.max_rows',None)
train.info()
test.info()
train.describe()
train.describe(include=[np.object])

#Handling Missing data
missing_train = train.isnull().sum()
missing_test = test.isnull().sum()
missing_train = pd.DataFrame({'NaN_count_train': missing_train, 'NaN_percentage_train': missing_train / len(train)})
missing_test = pd.DataFrame({'NaN_count_test': missing_test, 'NaN_percentage_test': missing_test / len(test)})
missing=pd.concat([missing_train,missing_test],axis=1)
missing

# Dublicate value in dataset
print(train.duplicated().sum()) # any dublicate in test data is not any problem bcz we will predict the outcome

# Removing the column having high amount of missing value and rows having lowest amount of missing term
for j in train.columns:
    if ((train[j].isnull().sum()/len(train))>=.45): # setting threshold value to eleminate columns
        train.drop(columns=j,axis=1,inplace=True)# eliminating columns
    elif ((train[j].isnull().sum()/len(train))<=.03): # setting threshold value to eleminate rows
        train.dropna(subset=[j],inplace=True) #eliminating rows
#Repeating above step for test dataset also

# Left over columns having missing values

series=train.isnull().sum()[train.isnull().sum()>0].index
series

for i in series:
    plt.figure(figsize=(15,5))
    sns.distplot(train[i].value_counts(dropna=False),color='green',hist=True)
    sns.distplot(train[i].value_counts(dropna=True),color='red',hist=True)
    plt.show()
    
# Replacing the null value with mean/meadian/mode and observing the graph

train['new_LotFrontage']=train['LotFrontage'].fillna(train['LotFrontage'].mean()) # median, mode
sns.distplot(train['LotFrontage'],color='green')
sns.distplot(train['new_LotFrontage'],color='red')

# Using KNN Imputer to impute the missing value in the dataset LotFrontage columns

from sklearn.impute import KNNImputer
knnimputer= KNNImputer(n_neighbors=3,add_indicator=True)
train['new_LotFrontage']=knnimputer.fit_transform(train[['LotFrontage','LotArea','TotalBsmtSF','1stFlrSF']])
plt.figure(figsize=(15,5))
sns.distplot(train['LotFrontage'],color='green',hist=False)
sns.distplot(train['new_LotFrontage'],color='red',hist=False)
# as compare to mean median KNN inputer give less distorted graph hence use KNN imputer for filling missing data
# Repeating same for test dataset

#'GarageType', 'GarageYrBlt', 'GarageFinish','GarageQual','GarageCond' all the missing value belong to the same row and give detail of same attribute of house hence we can replace them with 0/'NA
train['GarageType'].fillna('NA',inplace=True)
train['GarageYrBlt'].fillna(0,inplace=True)
train['GarageFinish'].fillna('NA',inplace=True)
train['GarageQual'].fillna('NA',inplace=True)
train['GarageCond'].fillna('NA',inplace=True)
train.drop('LotFrontage',inplace=True,axis=1)
# Repeating same for test dataset

# Handling categroy data
df_obj=train.select_dtypes(include =['object'])
df_obj.describe()

# Looking the content of the data and deciding weather it is nominal or category data type

for i in df_obj.columns:
    print(i)
    print(train[i].value_counts()/len(train)*100)
    
# Here we can see that some of the feature ['Street','LandContour','Condition2','RoofMatl','Heating',] in the dataset has most repeating value >95% as compare to another so we will drop that columns

train.drop(['Street','LandContour','Condition2','RoofMatl','Heating'],axis=1,inplace=True)
# Repeating same for test dataset
df_obj.drop(['Street','LandContour','Condition2','RoofMatl','Heating'],axis=1,inplace=True)

#Target Encoding
from sklearn.preprocessing import OrdinalEncoder
ord_enc=OrdinalEncoder()
train[df_obj.columns]=ord_enc.fit_transform(train[df_obj.columns])
# Repeating same for test dataset

# Outlier Data Handling
# In housing sector any outlier is having a direct affect on the total cost price of the house. so we need to understand the outlier befor treating them

for i in train.columns:
    if i=='SalePrice':
        continue
    plt.figure(figsize=(18,5))
    plt.subplot(1,3,1)
    sns.distplot(train[i])
    plt.subplot(1,3,2)
    sns.boxplot(train[i])
    plt.subplot(1,3,3)
    stats.probplot(train[i],dist='norm',plot=plt)
    print(train[i].skew())
    plt.show()
 
# FEAATURE SELECTION
# Finding The strong correlation between the feature 
sel_corr=train.corr()[(train.corr()>0.5) | (train.corr()<-0.5)]['SalePrice'].dropna()
print(sel_corr)
train[sel_corr.index].describe()

# Finding weak colinarity among the selected feature
temp=train[sel_corr.index].corr().sort_values('SalePrice',ascending=False)
plt.figure(figsize=(12,8))
sns.heatmap(data=temp,vmin=-1,vmax=1,annot=True)

for i in range((temp.shape[0])):
    for j in range(temp.shape[1]):
        if (temp.iloc[i,j]<0.2) and (temp.iloc[i,j]>-0.2):
            print(temp.index[i],temp.columns[j],temp.iloc[i,j])
col = ['OverallQual', 'YearBuilt', 'YearRemodAdd','GrLivA']

# Selecting the top 4 feature from the list and after feature scaling  running multiple algorithum to find the answer
            
# Feature scaling

from sklearn.preprocessing import StandardScaler
scaler=StandardScaler()
scaler.fit(train[['OverallQual', 'YearBuilt', 'YearRemodAdd','GrLivArea']])
x_train[['OverallQual', 'YearBuilt', 'YearRemodAdd','GrLivArea']]=pd.DataFrame(scaler.transform(train[['OverallQual', 'YearBuilt', 'YearRemodAdd','GrLivArea']]),columns=col)
# Repeating above transformation for test dataset
y_train=train['SalePrice']
print(x_train.shape,y_train.shape) # checking the shape of x,y variable set

# Feature modeling
from sklearn.linear_model import LinearRegression
LR = LinearRegression()
LR.fit(x_train,y_train) 

# Prediction on the test dataset
predict = LR.predict(test[['OverallQual', 'YearBuilt', 'YearRemodAdd','GrLivArea']])
predict= pd.concat([test['Id'],pd.Series(predict)],axis=1)
predict.to_csv("C:/Users/ankro/OneDrive/Desktop/housing_price.csv",index=False)


