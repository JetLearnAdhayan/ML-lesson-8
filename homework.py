import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

# Read CSV with NO header, then set your custom column names
data = pd.read_csv("adult.csv", header=None)
data.columns = ["age", "workclass", "Id", "education", "educational-num",
                "marital-status", "occupation", "relationship", "race",
                "gender", "capital-gain", "capital-loss", "hours-per-week",
                "native-country", "income"]

print(data.head())
print(data.info())

from sklearn.preprocessing import LabelEncoder

label_encoder = LabelEncoder()

data["workclass"] = label_encoder.fit_transform(data["workclass"])
data["education"] = label_encoder.fit_transform(data["education"])
data["marital-status"] = label_encoder.fit_transform(data["marital-status"])
data["occupation"] = label_encoder.fit_transform(data["occupation"])
data["relationship"] = label_encoder.fit_transform(data["relationship"])
data["race"] = label_encoder.fit_transform(data["race"])
data["gender"] = label_encoder.fit_transform(data["gender"])
data["native-country"] = label_encoder.fit_transform(data["native-country"])
data["income"] = label_encoder.fit_transform(data["income"])

X = data[["age", "workclass", "Id", "education", "educational-num",
          "marital-status", "occupation", "relationship", "race",
          "gender", "capital-gain", "capital-loss", "hours-per-week",
          "native-country"]]

y = data["income"]

from sklearn.model_selection import train_test_split

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=2)


from sklearn.tree import DecisionTreeClassifier

classifier = DecisionTreeClassifier(criterion="entropy", random_state=0)
classifier.fit(X_train, y_train)

y_pred = classifier.predict(X_test)

from sklearn.metrics import confusion_matrix, accuracy_score

acc = accuracy_score(y_test,y_pred)
print("Decision Tree Accuracy:", acc*100)

matrix = confusion_matrix(y_test,y_pred)

sb.heatmap(matrix,annot=True,fmt="d")
plt.title("Decision Tree Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()

from sklearn.ensemble import RandomForestClassifier

classifier = RandomForestClassifier(n_estimators=100)
classifier.fit(X_train,y_train)

y_pred = classifier.predict(X_test)

acc = accuracy_score(y_test,y_pred)
print("Random Forest Accuracy:", acc*100)

matrix = confusion_matrix(y_test,y_pred)

sb.heatmap(matrix,annot=True,fmt="d")
plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.show()
