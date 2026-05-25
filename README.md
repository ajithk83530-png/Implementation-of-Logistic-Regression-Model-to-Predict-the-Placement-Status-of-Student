# Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
```
1.Load the dataset, drop unnecessary columns, and encode categorical variables.
2.Define the features (X) and target variable (y).
3.Split the data into training and testing sets.
4.Train the logistic regression model, make predictions, and evaluate using accuracy and other
```

## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: Ajithkumar J
RegisterNumber: 212225040015 
*/
```
~~~python
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

data = pd.read_csv(r"C:\Users\admin\Desktop\Placement_Data.csv")
data.head()

datal = data.copy()
datal = datal.drop(["sl_no", "salary"], axis=1)
datal.head()

print("Missing values:\n", datal.isnull().sum())
print("Duplicate rows:", datal.duplicated().sum())

le = LabelEncoder()
datal["gender"] = le.fit_transform(datal["gender"])
datal["ssc_b"] = le.fit_transform(datal["ssc_b"])
datal["hsc_b"] = le.fit_transform(datal["hsc_b"])
datal["hsc_s"] = le.fit_transform(datal["hsc_s"])
datal["degree_t"] = le.fit_transform(datal["degree_t"])
datal["workex"] = le.fit_transform(datal["workex"])
datal["specialisation"] = le.fit_transform(datal["specialisation"])
datal["status"] = le.fit_transform(datal["status"])

x = datal.iloc[:, :-1]
y = datal["status"]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=0)

lr = LogisticRegression(solver="liblinear")
lr.fit(x_train, y_train)

y_pred = lr.predict(x_test)

accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

confusion = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", confusion)

classification_report_output = classification_report(y_test, y_pred)
print("Classification Report:\n", classification_report_output)
~~~

## Output:

HEAD:

<img width="1279" height="330" alt="image" src="https://github.com/user-attachments/assets/dc08f245-6c1f-44d8-9d71-bb8bc40b5e92" />


COPY:

<img width="1141" height="345" alt="image" src="https://github.com/user-attachments/assets/b17eeaca-88c7-40e3-8630-e7b42d0e7252" />

FIT TRANSFORM:

<img width="1122" height="707" alt="image" src="https://github.com/user-attachments/assets/6bcbd8d2-7077-44fe-a868-f4988f400e2f" />

LOGISTIC REGRESSION:

<img width="1231" height="309" alt="image" src="https://github.com/user-attachments/assets/ea7d1575-058d-4849-8ae7-d07fbd39245c" />

ACCURACY SCORE:

<img width="1225" height="169" alt="image" src="https://github.com/user-attachments/assets/38c66ca3-4212-4bc1-8c9e-ba748d634f92" />

CONFUSION MATRIX:

<img width="1229" height="203" alt="image" src="https://github.com/user-attachments/assets/f3458181-bcc3-44ac-a9b9-41b71e87a328" />

CLASSIFICATION REPORTS AND PREDICTIONS:

<img width="1217" height="524" alt="image" src="https://github.com/user-attachments/assets/04edd7b3-8a43-4167-b766-8a7decadc4f9" />





## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
