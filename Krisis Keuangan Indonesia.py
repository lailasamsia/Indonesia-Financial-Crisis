# # Import Library

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# # Load Data
df = pd.read_csv('data.csv', sep =';')


# # Review Data
df
df.info()
df.shape
describe=df.describe()

# # Data Cleaning
# ## Checking Duplicates Data
df.drop_duplicates(inplace = True)

# ## Checking Missing Values
df.isnull().sum()

plt.figure(figsize=(10,10))
df['Krisis'].value_counts().plot.pie(shadow=True, explode = [0,0.1],autopct='%1.2f%%')
plt.title("Persentase Krisis dan Tidak Krisis")
plt.legend()
plt.show()

plt.figure(figsize=(15,10))
sns.heatmap(data=df.drop(columns="Id",axis=1).corr(), cmap = 'Wistia',annot=True)
plt.show()

plt.figure(figsize=(20,15))
sns.pairplot(data=df.drop(columns=["Id","Krisis"],axis=1))
plt.show()

# ## Outlier Detection
def outlier(sample):
    Q1=sample.quantile(0.25)
    Q3=sample.quantile(0.75)
    IQR=Q3-Q1
    lower_range = Q1 -(1.5 * IQR)
    upper_range = Q3 +(1.5 * IQR)
    number_outlier=len(sample[sample>upper_range])+len(sample[sample<lower_range])
    print("Number of Outlier {}".format(number_outlier))
    if number_outlier>0:
        print("Outlier observation row:")
    else:
        pass
    for i in range(len(sample)):
        if sample[i]<lower_range: 
            print(i)
        elif sample[i]>upper_range:
            print(i)
        else:
            pass

# ### 
outlier(df['Ekspor'])
outlier(df['Cadangan Devisa'])
outlier(df['IHSG'])
outlier(df['Selisih Pinjaman dan Simpanan'])
outlier(df['Suku Bunga Simpanan Riil'])
outlier(df['Selisih BI Rate Riil dan FED Rate Riil'])
sns.boxplot(df['Selisih BI Rate Riil dan FED Rate Riil'])
outlier(df['Simpanan bank '])
outlier(df['Nilai Tukar Riil'])
outlier(df['Nilai Tukar Perdagangan'])
sns.boxplot(df['Nilai Tukar Perdagangan'])
outlier(df['M1'])
outlier(df['M2/Cadangan Devisa'])
outlier(df['M2M'])
sns.boxplot(df['M2M'])

Q1=df['M2M'].quantile(0.25)
Q3=df['M2M'].quantile(0.75)
IQR=Q3-Q1
lower_range = Q1 -(1.5 * IQR)
upper_range = Q3 +(1.5 * IQR)
df.loc[(df['M2M']>upper_range),:]
#Replace outlier observations with upper bound and lower bound
df.loc[(df['M2M']>upper_range),'M2M']=upper_range
df.loc[(df['M2M']<lower_range),'M2M']=lower_range
outlier(df['M2M'])

#Separating categorical and numerical columns
Id_col     = ['Id']
target_col = ['Krisis']
num_cols   = [x for x in df.columns if x not in target_col + Id_col]

# # Data Partition
from sklearn.model_selection import train_test_split

##partition data into data training and data testing
train,test = train_test_split(df,test_size = .20 ,random_state = 112)
    
##seperating dependent and independent variables on training and testing data
cols    = [i for i in df.columns if i not in Id_col + target_col]
train_X = train[cols]
train_Y = train[target_col]
test_X  = test[cols]
test_Y  = test[target_col]


# # SMOTE
from imblearn.over_sampling import SMOTE

#handle imbalance class using oversampling minority class with smote method
os = SMOTE(sampling_strategy='minority',random_state = 123,k_neighbors=5)
train_smote_X,train_smote_Y = os.fit_resample(train_X,train_Y)
train_smote_X = pd.DataFrame(data = train_smote_X,columns=cols)
train_smote_Y = pd.DataFrame(data = train_smote_Y,columns=target_col)

#target column value count
train_Y['Krisis'].value_counts()
#Proportion after smote
train_smote_Y['Krisis'].value_counts()


# # Logistic Regression
from sklearn.linear_model import LogisticRegression
classifier = LogisticRegression(random_state = 0, solver = 'liblinear',
                                penalty = 'l1')
classifier.fit(train_smote_X, train_smote_Y)
pred_lg = classifier.predict(test_X)

from sklearn.metrics import classification_report
target_names = ['No','Yes']
print(classification_report(test_Y, pred_lg, target_names=target_names))

from sklearn.metrics import accuracy_score
print("Accuracy for Random Forest on CV data: ",accuracy_score(test_Y,pred_lg))

dp = pd.read_csv('predict.csv', sep=';')
pred_lg_new = classifier.predict(dp.drop(columns=['Id'], axis=1))

dp['Krisis'] = pred_lg_new
submission1 = dp.loc[:, ['Id', 'Krisis']]

from sklearn.metrics import confusion_matrix
CF_lg=confusion_matrix(test_Y, pred_lg)
CF_lg

from sklearn.metrics import roc_auc_score
roc_auc_score(test_Y,pred_lg)
#submission.to_csv('submission.csv', index=False, header=True)


# # Random Forest
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
rfc=RandomForestClassifier(random_state=123)
param_grid = { 
    'n_estimators': [200,500,1000],
    'max_features': ['auto', 'log2'],
    'criterion' :['entropy','gini']
}

CV_rfc = GridSearchCV(estimator=rfc, param_grid=param_grid, cv= 3)
CV_rfc.fit(train_smote_X, train_smote_Y)

CV_rfc.best_params_

CV_rfc.best_score_

pred_rf=CV_rfc.predict(test_X)

print("Accuracy for Random Forest on CV data: ",accuracy_score(test_Y,pred_rf))

from sklearn.metrics import confusion_matrix
CF_rf=confusion_matrix(test_Y, pred_rf)
CF_rf

from sklearn.metrics import classification_report

target_names = ['No','Yes']
print(classification_report(test_Y, pred_rf, target_names=target_names))

from sklearn.metrics import roc_auc_score
roc_auc_score(test_Y,pred_rf)

dp = pd.read_csv('predict.csv', sep=';')
pred_rf_new = CV_rfc.predict(dp.drop(columns=['Id'],axis=1))

dp['Krisis'] = pred_rf_new
submission2 = dp.loc[:, ['Id', 'Krisis']]
submission2
#submission2.to_csv('submission2.csv', index=False, header=True)

