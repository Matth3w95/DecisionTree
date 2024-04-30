import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
#Load the dataset
data = pd.read_csv('data/bank-additional-full.csv', sep=';')

#Preprocessing
# this removes the target variable and encodes the categorical variables
X = pd.get_dummies(data.drop('y', axis=1))

# Now this part gets the target variable and converts no into 0 and yes into 1 it encodes it for machne learning
y = data['y'].apply(lambda x: 1 if x == 'yes' else 0)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


#initialise and train the classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

#Predicting the test set results
y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))


