# Implementation-of-Decision-Tree-Classifier-Model-for-Predicting-Employee-Churn

## AIM:
To write a program to implement the Decision Tree Classifier Model for Predicting Employee Churn.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Load the employee dataset and display basic information, including null values and class distribution of the `left` column.
2. Encode the categorical `salary` column using Label Encoding.
3. Define the features (`X`) and target (`y`) by selecting relevant columns.
4. Split the data into training and testing sets (80-20 split).
5. Initialize a Decision Tree Classifier with the entropy criterion and train it on the training data.
6. Predict the target values for the test set.
7. Calculate and display the model's accuracy.
8. Compute and display the confusion matrix for the predictions.
9. Predict the `left` status for a new employee sample.

## Program:
```
/*
Program to implement the Decision Tree Classifier Model for Predicting Employee Churn.
Developed by: Tanushree A
RegisterNumber:212223100057  
*/
import pandas as pd
data=pd.read_csv("Employee.csv")
data.head()
data.info()
data.isnull().sum()
data["left"].value_counts()
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
data["salary"]=le.fit_transform(data["salary"])
data.head()
X=data[["satisfaction_level","last_evaluation","number_project","average_montly_hours","time_spend_company","Work_accident","promotion_last_5years","salary"]]
X.head()
Y=data["left"]
from sklearn.model_selection import train_test_split
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=100)
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(criterion="entropy")
dt.fit(X_train,Y_train)
Y_pred=dt.predict(X_test)
from sklearn.metrics import accuracy_score
acc=accuracy_score(Y_test,Y_pred)
print(acc)
dt.predict([[0.5,0.8,9,260,6,0,1,2]])
```

## Output:
## Accuracy score
![image](https://github.com/user-attachments/assets/6e566360-4215-44b9-8447-7bc70e589ddc)

## predicted value
![image](https://github.com/user-attachments/assets/d97e4c84-771b-49df-ae2c-485dad4d04fd)



## Result:
Thus the program to implement the  Decision Tree Classifier Model for Predicting Employee Churn is written and verified using python programming.
