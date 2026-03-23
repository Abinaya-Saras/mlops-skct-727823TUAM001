# Abinaya Saras - Roll No 727823TUAM001

from datetime import datetime
from sklearn.datasets import load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

print("Roll No: 727823TUAM001", datetime.now())

data = load_diabetes()
X = data.data
y = data.target

X_train, X_test, y_train, y_test = train_test_split(X, y)

model = LinearRegression()
model.fit(X_train, y_train)

print("Training completed")
