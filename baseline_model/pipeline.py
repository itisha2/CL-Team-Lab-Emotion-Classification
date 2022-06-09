"""
Training and Evaluation
"""

from load_data import *
from baseline_classifier import *
from evaluation import *


def main(train_labels, input_data_path):
    """
    Step 1: Load training data
    """

    isear_train_df = load_train_data(data_path=input_data_path, train_labels=train_labels)
    print("1. Training Data Loaded: Shape = ", isear_train_df.shape)

    """
    Step 2: Load test data
    """
    isear_test_df = load_test_data(data_path=input_data_path, train_labels=train_labels)
    print("2. Test Data Loaded: Shape = ", isear_test_df.shape)

    """
    Step 3: Fit Classifier
    """
    nbObj = NaiveBayesClassifier()
    nbObj.fit(isear_train_df)
    print("3. Classifier fit to the data!")

    """
    Step 4: Call predict
    """
    predictions = nbObj.predict(isear_test_df["X"])
    print("4. Got predictions!")
    """
    Step 5: Evaluation
    """
    evalObj = evaluation(y_actual=isear_test_df["Y"], y_pred=predictions, labels=train_labels)
    print("5. View results: \n")
    return evalObj.main()[1]
