import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt

# Load the dataset
data = pd.read_csv('data/bank-additional-full.csv', sep=';')

# Preprocessing
# this removes the target variable and encodes the categorical variables
X = pd.get_dummies(data.drop('y', axis=1))

# Now this part gets the target variable and converts no into 0 and yes into 1 it encodes it for machne learning
y = data['y'].apply(lambda x: 1 if x == 'yes' else 0)

smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X, y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# initialise and train the classifier
model = DecisionTreeClassifier(random_state=42)
model.fit(X_train, y_train)

# Feature Importances
# model.feature_importances_ method that takes the model extracts the features that are most significant in decreasing gini impurity
# this is assinged to the importances vaureble
# feature_names are used to label these important features
importances = model.feature_importances_  # array
feature_names = X.columns

# sorts the importances array in descending order so most important to last
indices = np.argsort(importances)[::-1]
plt.figure()
plt.title("Feature importances")

# geneerates a sequence of numbers equal to the number of features
plt.bar(range(X.shape[1]), importances[indices])
# labels the feature na mes
plt.xticks(range(X.shape[1]), feature_names[indices], rotation=90)
plt.show()
# Remove less important features if required
# this creates a list of feature names that have an importance greater than 0.01, essentially
# a filtering criterion to remove features that contribute very little to the model;s predicton capability.

important_features = [feature for feature, importance in zip(feature_names, importances) if importance > 0.01]
# refining feature sets, updates X train adn Xtest to include only the selected important features. This refinement
# means subsequwnt model trianing and evaluations will use a ptentiall rediced set of feaures which can lead to
# faster training times and less overfitting
X_train = X_train[important_features]
X_test = X_test[important_features]
model.fit(X_train, y_train)
# Predicting the test set results
y_pred = model.predict(X_test)

print("Accuracy: ", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
