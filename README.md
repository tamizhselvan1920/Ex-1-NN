<H3>ENTER YOUR NAME : Tamizhselvan</H3>
<H3>ENTER YOUR REGISTER NO : 212222230158</H3>
<H3>EX. NO.1</H3>
<H3>DATE : 22.8.24</H3>
<H1 ALIGN =CENTER> Introduction to Kaggle and Data preprocessing</H1>

## AIM:

To perform Data preprocessing in a data set downloaded from Kaggle

## EQUIPMENTS REQUIRED:
Hardware – PCs
Anaconda – Python 3.7 Installation / Google Colab /Jupiter Notebook

## RELATED THEORETICAL CONCEPT:

**Kaggle :**
Kaggle, a subsidiary of Google LLC, is an online community of data scientists and machine learning practitioners. Kaggle allows users to find and publish data sets, explore and build models in a web-based data-science environment, work with other data scientists and machine learning engineers, and enter competitions to solve data science challenges.

**Data Preprocessing:**

Pre-processing refers to the transformations applied to our data before feeding it to the algorithm. Data Preprocessing is a technique that is used to convert the raw data into a clean data set. In other words, whenever the data is gathered from different sources it is collected in raw format which is not feasible for the analysis.
Data Preprocessing is the process of making data suitable for use while training a machine learning model. The dataset initially provided for training might not be in a ready-to-use state, for e.g. it might not be formatted properly, or may contain missing or null values.Solving all these problems using various methods is called Data Preprocessing, using a properly processed dataset while training will not only make life easier for you but also increase the efficiency and accuracy of your model.

**Need of Data Preprocessing :**

For achieving better results from the applied model in Machine Learning projects the format of the data has to be in a proper manner. Some specified Machine Learning model needs information in a specified format, for example, Random Forest algorithm does not support null values, therefore to execute random forest algorithm null values have to be managed from the original raw data set.
Another aspect is that the data set should be formatted in such a way that more than one Machine Learning and Deep Learning algorithm are executed in one data set, and best out of them is chosen.


## ALGORITHM:
STEP 1:Importing the libraries<BR>
STEP 2:Importing the dataset<BR>
STEP 3:Taking care of missing data<BR>
STEP 4:Encoding categorical data<BR>
STEP 5:Normalizing the data<BR>
STEP 6:Splitting the data into test and train<BR>

##  PROGRAM:
```py
import pandas as pd
import io
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
df=pd.read_csv('Churn_Modelling.csv')
df.head()
df.info()
df.isnull().sum()
df.duplicated().sum()
scaler=StandardScaler()
columns = ['CreditScore', 'Age', 'Tenure', 'Balance', 'NumOfProducts', 'EstimatedSalary']
df[columns] = scaler.fit_transform(df[columns])
df.head()
print("Normalized Data:\n",df.head())
X=df.iloc[:,:-1].values
Y=df.iloc[:,-1].values
print("INPUT(X)\n",X)
print("OUTPUT(Y\n)",Y)
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
```
## OUTPUT:
### DATA HEAD
![Screenshot 2024-08-24 083827](https://github.com/user-attachments/assets/b0955751-48fb-42d2-8d2d-905caf3d02f2)

### DATA INFO
![Screenshot 2024-08-24 084229](https://github.com/user-attachments/assets/ced85c42-e3f4-4c98-b1c5-d8311419994d)

### NULL VALUES
![Screenshot 2024-08-24 084331](https://github.com/user-attachments/assets/705eed9e-a1d8-4da3-9576-be280874de46)

### NORMALIZED DATA
![Screenshot 2024-08-24 084551](https://github.com/user-attachments/assets/b31ac0bf-244b-4e53-9f85-6013d5faee70)

### INPUT DATA
![Screenshot 2024-08-24 084703](https://github.com/user-attachments/assets/36b9f4c1-d8ae-4dd7-b824-0a09de907451)

### OUTPUT DATA
![Screenshot 2024-08-24 084826](https://github.com/user-attachments/assets/cea4dfa0-ff1a-4f8c-b3df-1e489f91eecc)

### TEST AND TRAINING DATA'
![Screenshot 2024-08-24 084955](https://github.com/user-attachments/assets/4970e41a-45fa-495b-9ce9-1fb83e71ff92)

## RESULT:
Thus, Implementation of Data Preprocessing is done in python  using a data set downloaded from Kaggle.


