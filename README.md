# EX-04 Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student

## AIM:
To write a program to implement the the Logistic Regression Model to Predict the Placement Status of Student.

## Equipments Required:
1. Hardware – PCs
2. Anaconda – Python 3.7 Installation / Jupyter notebook

## Algorithm
1. Import the required packages and print the present data.
2. Print the placement data and salary data.
3. Find the null and duplicate values.
4. Using logistic regression find the predicted values of accuracy , confusion matrices.
5. Display the results
## Program:
```
/*
Program to implement the the Logistic Regression Model to Predict the Placement Status of Student.
Developed by: VIGNESH KUMARAN N S
RegisterNumber: 212222230171
*/
```

```
import pandas as pd
df = pd.read_csv("Placement_Data.csv")
df.head()
df.isnull().sum()
df1 = df.copy()
df1.duplicated().sum()

from sklearn.preprocessing import LabelEncoder
le=LabelEncoder()
df1["gender"]=le.fit_transform(df1["gender"])
df1["ssc_b"]=le.fit_transform(df1["ssc_b"])
df1["hsc_b"]=le.fit_transform(df1["hsc_b"])
df1["hsc_s"]=le.fit_transform(df1["hsc_s"])
df1["degree_t"]=le.fit_transform(df1["degree_t"])
df1["workex"]=le.fit_transform(df1["workex"])
df1["specialisation"]=le.fit_transform(df1["specialisation"])
df1["status"]=le.fit_transform(df1["status"])
df1

x = df1.iloc[:, : -1]
y = df1["status"]

from sklearn.model_selection import train_test_split
x_train, x_test ,y_train ,y_test = train_test_split(x, y, test_size = 0.2, random_state=34)

from sklearn.linear_model import LogisticRegression
model = LogisticRegression(solver = "liblinear")
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

from sklearn.metrics import accuracy_score,confusion_matrix,classification_report
accuracy = accuracy_score(y_test, y_pred)
confusion = confusion_matrix(y_test, y_pred)
cr = classification_report(y_test ,y_pred)

print("Accuracy score:",accuracy)
print("\nConfusion matrix:\n",confusion)
print("\nClassification Report:\n",cr)

model.predict([[1,80,1,90,1,1,90,1,0,85,1,85]])```

## Output:
# DATASET
![data](https://github.com/BALA291/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120717501/afa1d61b-fdb7-48f5-93eb-56e3106844ef)

# DROPPING SALARY
![drop](https://github.com/BALA291/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120717501/71e31cd3-3741-4378-bd58-3c3311e3fa5d)

# NULL VALUES
![nullvalue](https://github.com/BALA291/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120717501/2b18f041-2284-44ed-8d39-97b214066ac7)

# X AND Y VALUES
![x and y values](https://github.com/BALA291/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120717501/b85f1380-b0ac-4537-9dd8-1e7b00533232)

# ACCURACY, CONFUSION MATRIX AND CLASSIFICATION REPORT
![accuracy con](https://github.com/BALA291/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120717501/05b2a75d-6e15-4d4c-aaa1-c04013a8f268)

# PREDICTED VALUES
![pred](https://github.com/BALA291/Implementation-of-Logistic-Regression-Model-to-Predict-the-Placement-Status-of-Student/assets/120717501/abf41d04-e39f-44c7-b745-c8d98687c033)


## Result:
Thus the program to implement the the Logistic Regression Model to Predict the Placement Status of Student is written and verified using python programming.
