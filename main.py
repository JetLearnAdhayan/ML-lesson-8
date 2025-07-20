#random forest algorithim
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

data = pd.read_csv("student-mat.csv")

print(data.head())
print(data.info())

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

data["school"] = label_encoder.fit_transform(data["school"])
data["sex"] = label_encoder.fit_transform(data["sex"])
data["famsize"] = label_encoder.fit_transform(data["famsize"])
data["Pstatus"] = label_encoder.fit_transform(data["Pstatus"])
data["Mjob"] = label_encoder.fit_transform(data["Mjob"])
data["Fjob"] = label_encoder.fit_transform(data["Fjob"])
data["reason"] = label_encoder.fit_transform(data["reason"])
data["guardian"] = label_encoder.fit_transform(data["guardian"])
data["schoolsup"] = label_encoder.fit_transform(data["schoolsup"])
data["famsup"] = label_encoder.fit_transform(data["famsup"])
data["paid"] = label_encoder.fit_transform(data["paid"])
data["activities"] = label_encoder.fit_transform(data["activities"])
data["nursery"] = label_encoder.fit_transform(data["nursery"])
data["higher"] = label_encoder.fit_transform(data["higher"])
data["internet"] = label_encoder.fit_transform(data["internet"])
data["romantic"] = label_encoder.fit_transform(data["romantic"])

data.drop("G1", axis = 1,inplace = True)
data.drop("G2",axis=1,inplace=True)

X = data[["school", "sex", "age", "famsize", "Pstatus", "Medu", "Fedu", "Mjob", "Fjob", "reason", "guardian", "traveltime", "studytime", "failures", "schoolsup", "famsup", "paid", "activities", "nursery", "higher", "internet", "romantic", "famrel", "freetime", "goout", "Dalc", "Walc", "health", "absences"]]
y = data["G3"]

from sklearn.model_selection import train_test_split

X_test,X_train,y_test,y_train = train_test_split(X,y,test_size=0.2,random_state=2)

from sklearn.ensemble import RandomForestClassifier
classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

acc = accuracy_score(y_test,y_pred)
print(acc*100)

matrix = confusion_matrix(y_test,y_pred)

sb.heatmap(matrix,annot=True,fmt="d")
plt.title("Confusion_matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
