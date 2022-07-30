
"""
Multi-layer perceptron is used for training the overlapping region detector and soft decision maker

The Multi-layer perceptron training class performs the following:
    1.) performs smote analysis on the input data (output from BERT), to handle class imbalance
    2.) Peforms PCA to reduce the number of dimensions from 7 to 3.
    3.) performs standarisation
    4.) split the train and test data
    5.) applies the multi-layer perceptron
    6.) performs evaluation
"""



from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import SMOTE
import pandas as pd
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings("ignore")


def mlp_classifier(X, y):
    """

    :param X: sorted list of confidence values from BERT
    :param y: binary label, whether bert model or humans agree or disgaree
    :return: pca, standardiser and the classification model
    """

    """Performing SMOTE Analysis"""

    oversample = SMOTE()
    X, y = oversample.fit_resample(X, y)
    print("Data Distribution after SMOTE : \n")
    print(y.value_counts())
    print()
    """Performing PCA"""

    pca = PCA(n_components=3)
    reduced_data = pca.fit_transform(X)
    reduced_data_df = pd.DataFrame(reduced_data, columns=["dim1", "dim2", "dim3"])
    print("PCA variance of 3 dimensions : ", pca.explained_variance_ratio_)
    print()
    """Performing data standardisation"""

    data_col = reduced_data_df.columns
    scalar = StandardScaler()

    scalar.fit(reduced_data_df)
    # transform data
    standard_data = scalar.transform(reduced_data_df)
    standard_df = pd.DataFrame(standard_data, columns=data_col)

    """
    Data Splitting
    """
    X_train, X_test, y_train, y_test = train_test_split(standard_df, y, test_size = 0.2,
                                                        random_state = 42, stratify = y)
    print("Shape train and test data, after train_test_split : ")
    print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)
    print()
    """Model training"""

    from sklearn.neural_network import MLPClassifier
    clf = MLPClassifier(max_iter=100, random_state=42, activation="tanh", solver="adam", hidden_layer_sizes=(100, ),
    learning_rate="adaptive")
    clf.fit(X_train, y_train)

    """Predictions"""

    y_pred_train = clf.predict(standard_df)
    y_pred_train_proba = clf.predict_proba(standard_df)

    # evaluate predictions
    accuracy = accuracy_score(y, y_pred_train)
    print("Train Accuracy: %.2f%%" % (accuracy * 100.0))
    print(classification_report(y, y_pred_train))

    print(confusion_matrix(y, y_pred_train))
    print()
    print()

    # make predictions for test data
    y_pred = clf.predict(X_test)
    y_pred_prob = clf.predict_proba(X_test)

    top_df_test = X_test
    top_df_test["yhat"] = y_pred
    top_df_test[["y_pred_prob_1", "y_pred_prob_2"]] = y_pred_prob
    top_df_test["ytrue"] = y_test
    top_match = []
    for ind, row in top_df_test.iterrows():
        if row["ytrue"] == row["yhat"]:
            top_match.append(1)
        else:
            top_match.append(0)
    top_df_test["match"] = top_match
    # evaluate predictions
    accuracy = accuracy_score(y_test, y_pred)
    print("Test Accuracy: %.2f%%" % (accuracy * 100.0))
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))
    print("_________________________________________________________________________")
    print()
    return top_df_test, pca, scalar, clf

