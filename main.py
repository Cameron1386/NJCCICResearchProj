# We will load data, preprocess it, and run our machine learning models here
# main.py

from preprocessing import load_and_clean_data
from models import train_logistic_regression, train_random_forest

def main():
    # Step 1: Load and Clean Data
    X_train, X_test, y_train, y_test = load_and_clean_data('data/NJCCIC Dataset - Training Dataset.csv')
    
    # Step 2: Train and Evaluate Logistic Regression
    train_logistic_regression(X_train, X_test, y_train, y_test)
    
    # Step 3: Train and Evaluate Random Forest
    train_random_forest(X_train, X_test, y_train, y_test)

if __name__ == '__main__':
    main()
