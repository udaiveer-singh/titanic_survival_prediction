import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# 1. Load the dataset
df=pd.read_csv('titanic_survival_prediction\Titanic-Dataset.csv')

# 2. preprocessing
df['Age'].fillna(df['Age'].median(), inplace=True)
df['Embarked'].fillna(df["Embarked"].mode()[0],inplace=True)
df['Fare'].fillna(df['Fare'].median(),inplace=True)

# 3. convert categories to numericals
le=LabelEncoder()
df["Sex"]=le.fit_transform(df["Sex"])
df["Embarked"]=le.fit_transform(df["Embarked"])

# 4. feature Selections
features=["Pclass","Sex","Age","SibSp","Parch","Fare","Embarked"]
X=df[features]
y=df["Survived"]

#5. Split into train and test
X_train,X_test,Y_train,Y_test=train_test_split(X,y,test_size=0.2,random_state=42)

#6. train the model
model=LogisticRegression(max_iter=1000)
model.fit(X_train,Y_train)

# make predictions
Y_pred=model.predict(X_test)

# accuracy
print("accuracy :",accuracy_score(Y_test,Y_pred))
