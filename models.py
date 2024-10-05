# where the machine learning models are defined and trained
# models.py

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

def train_logistic_regression(X_train, X_test, y_train, y_test):
    # Initialize and train the model
    model = LogisticRegression()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Logistic Regression Results:")
    print(classification_report(y_test, y_pred))

def train_random_forest(X_train, X_test, y_train, y_test):
    # Initialize and train the model
    model = RandomForestClassifier()
    model.fit(X_train, y_train)

    # Predict and evaluate
    y_pred = model.predict(X_test)
    print("Random Forest Results:")
    print(classification_report(y_test, y_pred))
