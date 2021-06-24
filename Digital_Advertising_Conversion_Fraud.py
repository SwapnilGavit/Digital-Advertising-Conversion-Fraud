
# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# Reading Training Data set
Data =pd.read_csv(r'Data\Training Data.csv')

print('Shape of Data:',Data.shape)
Data.head()


# Reading Testing Data set
Test_Data =pd.read_csv(r'Data\Test Data.csv')

print('Shape of Traning Data:',Test_Data.shape)


# Check the Column Names
Data.columns

# Check the values of Conversion fraud
Data['conversion_fraud'].value_counts()

# Descriptive Statistics
Data.describe()

# Descriptive Statistics for categorical data
Data.describe(include='object')


# Check Missing Values in features
Data.isnull().sum()


# Sort rows with Maximum Missing Values
Data.isnull().sum(axis=1).sort_values(ascending=False)


# Check how many rows have missing values more than 16
print(Data.shape)
Data[Data.isnull().sum(axis=1)>16].shape


# Delete rows with more than 16 missing values rows
print('Shape before deleting rows',Data.shape)
Data=Data[Data.isnull().sum(axis=1)<=16]
print('Shape after deleting rows',Data.shape)


# column in which all the values are missing
Data.isnull().all(axis=0).sum()


# Check the percetage of missing values in columns
x= Data.isnull().sum()
y= (Data.isnull().sum()/Data.shape[0])*100
z={'No. of Missing Values':x,'Percentage of Missing Values':y}
df= pd.DataFrame(z,columns=['No. of Missing Values','Percentage of Missing Values'])
df.sort_values(by='Percentage of Missing Values',ascending=False)


# Store the Record id column
record_id = Test_Data['record_id']

# record_id column is only present in the Test Data So we drop it
Test_Data.drop(['record_id'],axis=1,inplace=True)


#  Add a Flag to Identify the Train and Test Data Set
Data['Type']=1
Test_Data['Type']= 0

print('Shape of Train Data',Data.shape)
print('Shape of Test Data',Test_Data.shape)

# Concat the Data sets
Full_Data=pd.concat([Data,Test_Data],axis=0)
print('Shape of Full Data',Full_Data.shape)



# Create a list of columns which has more than 50% missing values
remove_col_lst=df[df['Percentage of Missing Values']>=50.0].index


# Delete the Columns which has more than 50% missing values
print('Shape of before ',Full_Data.shape)

Full_Data.drop(remove_col_lst,axis=1,inplace=True)

print('Shape of after ',Full_Data.shape)


# Check dtypes of column which has missing values
Full_Data.dtypes[Full_Data.isnull().any()]

# Conversion column is our Target column 
# So convert it to numeric column befor droping object dtypes
# Check the unique values for conversion to numeric
Full_Data['conversion_fraud'].unique()

Full_Data['conversion_fraud'].replace((True,False),(1,0),inplace=True)


# Drop the object type columns from data 
Full_Data.drop(Full_Data.select_dtypes('object').columns,axis=1,inplace=True)

Full_Data.isnull().sum()


# Fill the values of data

Full_Data['cityGrpDimId_cr'].fillna(Full_Data['cityGrpDimId_cr'].mode()[0],inplace=True)

Full_Data['stateGrpDimId_cr'].fillna(Full_Data['stateGrpDimId_cr'].mode()[0],inplace=True)

Full_Data['clickTimeInMillis_cr'].fillna(Full_Data['clickTimeInMillis_cr'].mode()[0],inplace=True)

Full_Data['v_cr'].fillna(Full_Data['v_cr'].mode()[0],inplace=True)

Full_Data['templateid_cr'].fillna(Full_Data['templateid_cr'].mean(),inplace=True)

Full_Data['clickbid_cr'].fillna(Full_Data['clickbid_cr'].mode()[0],inplace=True)


Full_Data.isnull().sum().sum()

# 455 values are converstion fraud values of Test data whih we wanted to find

# Normalize the data befor fitting to ML model
# Lets import min_max_scale for that
from sklearn.preprocessing import minmax_scale

# Stor column names
cloumn_names=Full_Data.columns

scaled_Full_Data = minmax_scale(Full_Data, feature_range=(0,1))

# Convert to Pandas DataFrame and add Columns names
scaled_Full_Data=pd.DataFrame(scaled_Full_Data,columns=cloumn_names)
scaled_Full_Data.head()

# Seperate the Train and Test Data Sets using flages created earlier
Data_Modified= scaled_Full_Data[scaled_Full_Data['Type']==1]
Test_Modified= scaled_Full_Data[scaled_Full_Data['Type']==0]

print('Shape of Train Data',Data_Modified.shape)
print('Shape of Test Data',Test_Modified.shape)


# Split the Target column from the Data
X = Data_Modified.drop(['conversion_fraud'],axis=1)
Y = Data_Modified['conversion_fraud']

# Check the Shape of X and Y
print('Shape of X:', X.shape)
print('Shape of Y:', Y.shape)


# It is very important to resample the data, as the Target class is Highly imbalanced.
# Here We are going to use Over Sampling Technique to resample the data.

# Import the SMOTE algorithm to do the same.
from imblearn.over_sampling import SMOTE
x_res, y_res = SMOTE().fit_resample(X, Y)

# Check Shape of X and Y after resampling it
print('Shape of X:',x_res.shape)
print('Shape of X:',y_res.shape)
print('\n')
# Check the value counts of target variable
print("Before Resambling :")
print(Y.value_counts())
print("\n After Resambling :")
print(y_res.value_counts())


from sklearn.model_selection import train_test_split

x_train, x_test, y_train, y_test = train_test_split(x_res,y_res, test_size=0.20, random_state=0,shuffle= True, stratify=y_res)

print("Shape of the x Train :", x_train.shape)
print("Shape of the y Train :", y_train.shape)
print("Shape of the x Test :", x_test.shape)
print("Shape of the y Test :", y_test.shape)


# Import accuracy_score to check the accuracy of Model
from sklearn.metrics import accuracy_score


# KNN Classifier
from sklearn.neighbors import KNeighborsClassifier

kNN = KNeighborsClassifier(n_neighbors = 10, metric = 'minkowski', p = 1)

kNN.fit(x_train,y_train)

y_pred_KNN = kNN.predict(x_test)

print('Accuracy : %s '%'{0:.2%}'.format(accuracy_score(y_test, y_pred_KNN)))


# DecisionTreeClassifier
from sklearn.tree import DecisionTreeClassifier

Dt = DecisionTreeClassifier()

Dt.fit(x_train,y_train)

y_pred_Dt = Dt.predict(x_test)

print('Accuracy : %s '%'{0:.2%}'.format(accuracy_score(y_test, y_pred_Dt)))


# Gradient Boosting Classifier
# Import Grdient Boosting Classifier from sklearn
from sklearn.ensemble import GradientBoostingClassifier

# Create  ML Model and fit the training data
GBC = GradientBoostingClassifier(learning_rate=0.2, max_depth=4, n_estimators=200, random_state=25)
GBC.fit(x_train, y_train)

# Predict Output and Store it 
y_pred_GBC= GBC.predict(x_test)

# Accuracy
from sklearn.metrics import accuracy_score
print('Accuracy : %s '%'{0:.2%}'.format(accuracy_score(y_test, y_pred_GBC)))

# Accuracy of Gradient Boost classifier is more than Dicision Tree and KNN
# So we choose the Gradient Boost Classifer as a ML model

# Confusion Matrix

# Analyze the Performance of Gradient Boosting using Confusion matrix
from sklearn.metrics import confusion_matrix
Conf_Matrix = confusion_matrix(y_test, y_pred_GBC)

# Visualize the Confusion Matrix using heatmap
plt.rcParams['figure.figsize'] = (3,3)
sns.heatmap(Conf_Matrix, annot = True, fmt = '.8g',center=30,cmap="rocket")
plt.show()



# Check the Classification report for performance analysis

from sklearn.metrics import  classification_report
Class_Report = classification_report(y_test, y_pred_GBC)
print(Class_Report)



# Cross Validation

# import cross validation function
from sklearn.model_selection import cross_val_score

scores = cross_val_score(GBC, x_train, y_train, cv=10)
print(scores)
print('\n Cross-Validation Score :%s '%'{0:.2%}'.format(scores.mean()))



Test_Modified.head()

# predict the converstion fraud for test data set and store it
result= GBC.predict(Test_Modified.drop(['conversion_fraud'],axis=1))


# Convert to pandas DataFrame
result=pd.DataFrame(result,columns={'conversion_fraud'})


result['conversion_fraud'].replace((1,0),(True,False),inplace=True)

# Set Index as Loan_ID which was stored in loan_id earlier
result.set_index(record_id, inplace=True)

result.head()


# Store the Final result
result.to_csv(r'C:Outputs\Result_Submission.csv')
